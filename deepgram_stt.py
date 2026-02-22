"""Deepgram streaming STT with VAD cost-gating.

Replaces ContinuousSTT. Same external interface:
- Produces TranscriptSegment objects into a TranscriptBuffer
- on_segment/on_stats callbacks
- set_playing_audio() for playback suppression
- start()/stop() lifecycle

Key difference: STT inference is cloud-based (Deepgram WebSocket),
not local (Whisper). Silero VAD runs locally to gate what audio
reaches Deepgram (cost saving, not latency saving).

Architecture:
- Audio capture thread records from AEC source (or default mic)
- VAD gates connection lifecycle (active/idle/sleep), NOT per-chunk filtering
- Deepgram SDK manages WebSocket thread, fires callbacks
- Callbacks push transcripts to asyncio loop via call_soon_threadsafe
- KeepAlive messages sent during silence to prevent 10-second timeout
- Reconnection with exponential backoff on connection failures

Audio format: 24kHz 16-bit mono PCM (linear16), no resampling needed.
Silero VAD receives 24kHz->16kHz resampled audio (same as ContinuousSTT).
"""

import asyncio
import threading
import time
from pathlib import Path

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets.listen_v1_control_message import (
    ListenV1ControlMessage,
)

from transcript_buffer import TranscriptBuffer, TranscriptSegment, is_hallucination

# Audio settings (match live_session.py)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4096  # bytes per read (~85ms at 24kHz 16-bit mono)
BYTES_PER_SAMPLE = 2

# VAD settings
VAD_THRESHOLD = 0.5

# Deepgram connection lifecycle
KEEPALIVE_INTERVAL = 5.0      # seconds between KeepAlive when idle
IDLE_TIMEOUT = 10.0           # seconds of silence -> idle mode (KeepAlive only)
SLEEP_TIMEOUT = 60.0          # seconds of idle -> disconnect entirely
RECONNECT_DELAY_BASE = 1.0    # exponential backoff base
RECONNECT_MAX_DELAY = 30.0    # backoff cap
MAX_RECONNECT_ATTEMPTS = 10
PLAYBACK_COOLDOWN = 0.3       # seconds after TTS ends before accepting transcripts

# Stats reporting
STATS_INTERVAL_SECONDS = 5.0


class DeepgramSTT:
    """VAD-gated Deepgram streaming STT producing TranscriptSegments.

    Architecture:
    - Audio capture: own pasimple capture thread (same pattern as ContinuousSTT)
    - VAD: Silero ONNX (CPU, <1ms per chunk) gates connection lifecycle
    - Deepgram: WebSocket streaming, Nova-3 model
    - Output: TranscriptSegment -> TranscriptBuffer, callbacks, transcript_q

    Args:
        api_key: Deepgram API key
        transcript_buffer: TranscriptBuffer to append segments to
        transcript_q: asyncio.Queue for pipeline integration (STT stage reads this)
        aec_device_name: PipeWire AEC source name (None = default mic)
        on_segment: callback(TranscriptSegment) for each new segment
        on_stats: callback(dict) for periodic stats
        on_unavailable: callback() when max reconnect attempts exhausted
    """

    def __init__(self, api_key, transcript_buffer, transcript_q=None,
                 aec_device_name=None, on_segment=None, on_stats=None,
                 on_unavailable=None):
        self._api_key = api_key
        self._transcript_buffer = transcript_buffer
        self._transcript_q = transcript_q
        self._aec_device_name = aec_device_name
        self._on_segment = on_segment
        self._on_stats = on_stats
        self._on_unavailable = on_unavailable

        self._running = False
        self._stop_event = threading.Event()

        # Playback suppression (same interface as ContinuousSTT)
        self._playing_audio = False
        self._playback_end_time = 0.0

        # Deepgram connection
        self._dg_connection = None
        self._dg_context = None  # context manager from connect()
        self._connected = False
        self._reconnect_attempts = 0

        # VAD (reuse Silero ONNX)
        self._vad_model = None
        self._vad_state = None

        # Transcript accumulation (between is_final segments until speech_final)
        self._accumulated_finals = []

        # Audio forwarding queue (capture thread -> async forwarder)
        self._audio_q = asyncio.Queue(maxsize=200)

        # KeepAlive tracking
        self._last_audio_sent_time = 0.0

        # Stats
        self._segment_count = 0
        self._hallucination_count = 0
        self._latencies = []
        self._last_stats_time = 0.0
        self._speech_start_time = 0.0

    # ── Public interface (matches ContinuousSTT) ──────────────────

    @property
    def running(self):
        """Whether the DeepgramSTT loop is active."""
        return self._running

    @property
    def stats(self):
        """Return current stats dict."""
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies else 0.0
        )
        return {
            'segment_count': self._segment_count,
            'hallucination_count': self._hallucination_count,
            'avg_latency_ms': avg_latency,
            'buffer_depth': len(self._transcript_buffer),
            'connected': self._connected,
            'reconnect_attempts': self._reconnect_attempts,
        }

    async def start(self):
        """Begin capture + Deepgram streaming loop."""
        self._running = True
        self._stop_event.clear()
        self._last_stats_time = time.time()

        self._load_vad_model()

        # Start audio capture thread
        device_name = self._resolve_device()
        loop = asyncio.get_event_loop()
        capture_thread = threading.Thread(
            target=self._capture_thread, args=(device_name, loop), daemon=True
        )
        capture_thread.start()

        print("DeepgramSTT: Started", flush=True)

        try:
            await self._connection_loop()
        except Exception as e:
            print(f"DeepgramSTT: Fatal error: {e}", flush=True)
        finally:
            self._running = False
            print("DeepgramSTT: Stopped", flush=True)

    def stop(self):
        """Signal graceful shutdown."""
        self._running = False
        self._stop_event.set()

    def set_playing_audio(self, playing):
        """Called by LiveSession during TTS playback.

        During playback, audio is still sent to Deepgram (for KeepAlive and
        so the connection stays open), but transcripts are suppressed to
        avoid transcribing the AI's own speech.

        NOTE: Unlike ContinuousSTT which discards audio during playback,
        DeepgramSTT keeps streaming audio. Deepgram handles echo better
        than local Whisper, and PipeWire AEC removes the AI's voice
        from the mic signal. We suppress the TRANSCRIPT output, not the
        audio input.
        """
        self._playing_audio = playing
        if not playing:
            self._playback_end_time = time.time()

    # ── Connection lifecycle ──────────────────────────────────────

    async def _connection_loop(self):
        """Manage Deepgram WebSocket connection with reconnection."""
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                if not self._running:
                    break
                self._connected = False
                self._reconnect_attempts += 1

                if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    self._check_reconnect_exhausted()
                    break

                delay = min(
                    RECONNECT_DELAY_BASE * (2 ** (self._reconnect_attempts - 1)),
                    RECONNECT_MAX_DELAY
                )
                print(f"DeepgramSTT: Connection failed ({e}), "
                      f"retry {self._reconnect_attempts} in {delay:.1f}s", flush=True)
                await asyncio.sleep(delay)

    async def _connect_and_stream(self):
        """Connect to Deepgram, register callbacks, forward audio."""
        client = DeepgramClient(api_key=self._api_key)

        # Connect using the sync client (SDK manages its own WebSocket thread)
        self._dg_context = client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=str(SAMPLE_RATE),
            channels="1",
            interim_results="true",
            endpointing="300",
            utterance_end_ms="1000",
            smart_format="true",
            vad_events="true",
            punctuate="true",
            language="en",
        )

        # Enter context manager to get the connection object
        dg_connection = self._dg_context.__enter__()

        # Register event handlers
        dg_connection.on(EventType.OPEN, self._on_open)
        dg_connection.on(EventType.MESSAGE, self._on_message)
        dg_connection.on(EventType.CLOSE, self._on_close)
        dg_connection.on(EventType.ERROR, self._on_error)

        self._dg_connection = dg_connection

        # Start the WebSocket listener (SDK manages internally)
        dg_connection.start_listening()

        self._connected = True
        self._reconnect_attempts = 0
        print("DeepgramSTT: Connected to Deepgram Nova-3", flush=True)

        try:
            await self._audio_forward_loop()
        finally:
            try:
                # Send Finalize before closing to get last transcript
                dg_connection.send_control(
                    ListenV1ControlMessage(type="Finalize")
                )
                # Exit context manager
                self._dg_context.__exit__(None, None, None)
            except Exception:
                pass
            self._connected = False
            self._dg_connection = None
            self._dg_context = None

    # ── Deepgram event handlers ───────────────────────────────────

    def _on_open(self, *args, **kwargs):
        """Handle WebSocket open event."""
        print("DeepgramSTT: WebSocket opened", flush=True)

    def _on_message(self, result, *args, **kwargs):
        """Handle Deepgram transcription results.

        Deepgram sends three types of results:
        1. interim (is_final=False): partial/speculative transcript
        2. final (is_final=True, speech_final=False): confirmed text chunk
        3. speech_final (is_final=True, speech_final=True): end of utterance

        We accumulate is_final chunks and flush on speech_final.
        """
        try:
            channel = result.channel
            alternatives = channel.alternatives
            if not alternatives:
                return

            transcript = alternatives[0].transcript.strip()
            if not transcript:
                return

            is_final = getattr(result, 'is_final', False)
            speech_final = getattr(result, 'speech_final', False)

            if is_final:
                self._accumulated_finals.append(transcript)

                if speech_final:
                    # Flush accumulated text as one segment
                    full_text = " ".join(self._accumulated_finals).strip()
                    self._accumulated_finals.clear()

                    if full_text:
                        self._emit_transcript(full_text)

        except Exception as e:
            print(f"DeepgramSTT: Message handling error: {e}", flush=True)

    def _on_close(self, *args, **kwargs):
        """Handle WebSocket close event."""
        print("DeepgramSTT: WebSocket closed", flush=True)
        self._connected = False

    def _on_error(self, error, *args, **kwargs):
        """Handle WebSocket error event."""
        print(f"DeepgramSTT: Error: {error}", flush=True)

    # ── Transcript emission ───────────────────────────────────────

    def _emit_transcript(self, text):
        """Process a complete transcript and emit to all consumers."""
        # Playback suppression: discard transcripts of AI's own speech
        if self._playing_audio:
            return
        if time.time() - self._playback_end_time < PLAYBACK_COOLDOWN:
            return

        # Hallucination filter (reuse existing)
        if is_hallucination(text):
            self._hallucination_count += 1
            print(f"DeepgramSTT: Rejected hallucination: \"{text}\"", flush=True)
            return

        # Create segment
        segment = TranscriptSegment(
            text=text,
            timestamp=time.time(),
            source="user",
        )

        # Append to transcript buffer (shared with decision model)
        self._transcript_buffer.append(segment)
        self._segment_count += 1
        print(f"STT [deepgram]: {text}", flush=True)

        # Callback for live_session event emission
        if self._on_segment:
            self._on_segment(segment)

        # Pipeline integration: put on transcript queue for STT stage
        if self._transcript_q:
            try:
                self._transcript_q.put_nowait(segment)
            except asyncio.QueueFull:
                pass  # Drop rather than block

    # ── KeepAlive management ──────────────────────────────────────

    def _maybe_send_keepalive(self):
        """Send KeepAlive if no audio has been sent recently."""
        if not self._dg_connection or not self._connected:
            return
        if time.time() - self._last_audio_sent_time < KEEPALIVE_INTERVAL:
            return

        try:
            self._dg_connection.send_control(
                ListenV1ControlMessage(type="KeepAlive")
            )
            self._last_audio_sent_time = time.time()
        except Exception:
            pass

    # ── Audio sending ─────────────────────────────────────────────

    def _try_send_audio(self, audio_data):
        """Attempt to send audio data to Deepgram. Handle errors."""
        if not self._dg_connection or not self._connected:
            return

        try:
            self._dg_connection.send_media(audio_data)
            self._last_audio_sent_time = time.time()
        except Exception as e:
            print(f"DeepgramSTT: Send error: {e}", flush=True)
            self._connected = False

    # ── Reconnection exhaustion ───────────────────────────────────

    def _check_reconnect_exhausted(self):
        """Check if reconnect attempts are exhausted and fire callback."""
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print(f"DeepgramSTT: Exhausted {MAX_RECONNECT_ATTEMPTS} "
                  f"reconnect attempts", flush=True)
            if self._on_unavailable:
                self._on_unavailable()

    # ── Audio forwarding ──────────────────────────────────────────

    async def _audio_forward_loop(self):
        """Read audio from capture queue, VAD-gate, forward to Deepgram.

        VAD gates CONNECTION LIFECYCLE, not per-chunk audio filtering:
        - When speech detected: stream audio to Deepgram
        - When silence: send audio to Deepgram (preserves endpointing context)
        - When extended silence (>KEEPALIVE_INTERVAL): send KeepAlive instead
        - When very long silence (>SLEEP_TIMEOUT): disconnect
        """
        last_speech_time = time.time()

        while self._running and self._connected:
            try:
                audio_data = self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                self._maybe_send_keepalive()
                self._maybe_emit_stats()
                continue

            # Run VAD on chunk (for lifecycle decisions, not filtering)
            vad_prob = self._run_vad(audio_data)

            if vad_prob > VAD_THRESHOLD:
                last_speech_time = time.time()

            # Always send audio during active period (preserves Deepgram endpointing)
            time_since_speech = time.time() - last_speech_time

            if time_since_speech < IDLE_TIMEOUT:
                # ACTIVE: stream all audio (speech + silence between words)
                self._try_send_audio(audio_data)
            elif time_since_speech < SLEEP_TIMEOUT:
                # IDLE: send KeepAlive instead of audio (save cost)
                self._maybe_send_keepalive()
            else:
                # SLEEP: disconnect entirely (zero cost)
                print("DeepgramSTT: Sleep timeout, disconnecting", flush=True)
                break

            self._maybe_emit_stats()

    # ── Audio capture (same pattern as ContinuousSTT) ─────────────

    def _capture_thread(self, device_name, loop):
        """Record audio in a daemon thread, push to async queue."""
        import pasimple

        while not self._stop_event.is_set():
            try:
                with pasimple.PaSimple(
                    pasimple.PA_STREAM_RECORD,
                    pasimple.PA_SAMPLE_S16LE,
                    CHANNELS, SAMPLE_RATE,
                    app_name='push-to-talk-deepgram',
                    device_name=device_name,
                ) as pa:
                    while not self._stop_event.is_set():
                        data = pa.read(CHUNK_SIZE)

                        def _enqueue(d=data):
                            try:
                                self._audio_q.put_nowait(d)
                            except asyncio.QueueFull:
                                pass

                        loop.call_soon_threadsafe(_enqueue)
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"DeepgramSTT: Capture error: {e}, reconnecting...",
                          flush=True)
                    time.sleep(1)

    # ── VAD (same as ContinuousSTT) ───────────────────────────────

    def _load_vad_model(self):
        """Load Silero VAD ONNX model for speech detection."""
        model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
        if not model_path.exists():
            print(f"DeepgramSTT: VAD model not found at {model_path}", flush=True)
            return False

        try:
            import onnxruntime
            import numpy as np
            self._vad_model = onnxruntime.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            self._vad_state = {
                'state': np.zeros((2, 1, 128), dtype=np.float32),
                'sr': np.array(16000, dtype=np.int64),
                'context': np.zeros(64, dtype=np.float32),
            }
            print("DeepgramSTT: Silero VAD loaded", flush=True)
            return True
        except Exception as e:
            print(f"DeepgramSTT: Failed to load VAD: {e}", flush=True)
            return False

    def _run_vad(self, audio_bytes):
        """Run VAD inference on audio chunk, return max speech probability.

        Resamples 24kHz audio to 16kHz for Silero VAD (same as ContinuousSTT).
        """
        if not self._vad_model:
            return 0.0

        import numpy as np

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16kHz (from 24kHz: take 2 of every 3 samples)
        if SAMPLE_RATE == 24000:
            indices = np.arange(0, len(samples), 1.5).astype(int)
            indices = indices[indices < len(samples)]
            samples = samples[indices]

        context = self._vad_state['context']
        max_prob = 0.0

        try:
            for i in range(0, len(samples) - 511, 512):
                window = samples[i:i + 512]
                input_data = np.concatenate([context, window]).reshape(1, -1)

                ort_inputs = {
                    'input': input_data,
                    'state': self._vad_state['state'],
                    'sr': self._vad_state['sr']
                }
                ort_outputs = self._vad_model.run(None, ort_inputs)
                prob = ort_outputs[0].item()
                self._vad_state['state'] = ort_outputs[1]
                context = window[-64:]

                if prob > max_prob:
                    max_prob = prob

            self._vad_state['context'] = context
            return max_prob
        except Exception as e:
            print(f"DeepgramSTT VAD error: {e}", flush=True)
            return 0.0

    def _resolve_device(self):
        """Try AEC device, fall back to default mic (None)."""
        if not self._aec_device_name:
            return None

        try:
            import pasimple
            test_pa = pasimple.PaSimple(
                pasimple.PA_STREAM_RECORD,
                pasimple.PA_SAMPLE_S16LE,
                CHANNELS, SAMPLE_RATE,
                app_name='push-to-talk-deepgram',
                device_name=self._aec_device_name,
            )
            test_pa.read(CHUNK_SIZE)
            del test_pa
            print(f"DeepgramSTT: Using echo-cancelled source "
                  f"'{self._aec_device_name}'", flush=True)
            return self._aec_device_name
        except Exception as e:
            print(f"DeepgramSTT: AEC source '{self._aec_device_name}' "
                  f"not available ({e}), using default mic", flush=True)
            return None

    def _maybe_emit_stats(self):
        """Emit stats periodically via callback."""
        now = time.time()
        if now - self._last_stats_time >= STATS_INTERVAL_SECONDS:
            self._last_stats_time = now
            if self._on_stats:
                self._on_stats(self.stats)
