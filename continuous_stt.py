"""Continuous Speech-to-Text with VAD-gated Whisper transcription.

Provides ContinuousSTT: a class that captures audio continuously (no PTT),
gates it through Silero VAD, and transcribes speech segments with
faster-whisper (distil-large-v3). Clean transcripts are appended to
a TranscriptBuffer.

Key behaviors:
- Audio capture thread records from AEC source (or default mic) at 24kHz 16-bit mono
- Each chunk (~85ms) runs through Silero VAD (CPU, <1ms)
- Speech accumulates; silence gap (0.8s) triggers Whisper transcription
- Safety cap: force transcription after MAX_BUFFER_SECONDS (10s)
- Hallucination filter rejects known Whisper artifacts
- Stats emitted periodically (segment_count, hallucination_rate, latency, VRAM)
"""

import asyncio
import threading
import time
from pathlib import Path

from transcript_buffer import TranscriptBuffer, TranscriptSegment, is_hallucination

# Audio settings (match live_session.py)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4096  # bytes per read (~85ms at 24kHz 16-bit mono)
BYTES_PER_SAMPLE = 2

# VAD settings
VAD_THRESHOLD = 0.5
SILENCE_CHUNKS_THRESHOLD = 10  # ~850ms of silence at 85ms/chunk

# Buffer limits
MAX_BUFFER_SECONDS = 10
MAX_BUFFER_CHUNKS = int(MAX_BUFFER_SECONDS / (CHUNK_SIZE / (SAMPLE_RATE * BYTES_PER_SAMPLE)))  # ~118
MIN_BUFFER_CHUNKS = 6  # ~510ms minimum speech

# Whisper model
WHISPER_MODEL_NAME = "distil-large-v3"

# Stats reporting interval
STATS_INTERVAL_SECONDS = 5.0


class ContinuousSTT:
    """VAD-gated Whisper loop producing TranscriptSegments.

    Args:
        transcript_buffer: TranscriptBuffer to append clean segments to
        vram_monitor: optional VRAMMonitor for GPU health checks
        aec_device_name: PipeWire AEC source name (None = default mic)
        on_segment: callback(TranscriptSegment) for each new segment
        on_stats: callback(dict) for periodic stats reporting
    """

    def __init__(self, transcript_buffer, vram_monitor=None,
                 aec_device_name=None, on_segment=None, on_stats=None):
        self._transcript_buffer = transcript_buffer
        self._vram_monitor = vram_monitor
        self._aec_device_name = aec_device_name
        self._on_segment = on_segment
        self._on_stats = on_stats

        self._running = False
        self._stop_event = threading.Event()
        self._playing_audio = False  # Set by LiveSession during TTS playback
        self._playback_end_time = 0.0  # Cooldown after TTS ends
        self._PLAYBACK_COOLDOWN = 0.5  # Seconds to wait after TTS before processing

        # Audio queue for capture thread -> processing loop
        self._audio_q = asyncio.Queue(maxsize=200)

        # VAD state
        self._vad_model = None
        self._vad_state = None

        # Whisper model (lazy loaded)
        self._whisper_model = None

        # Speech buffer
        self._audio_buffer = bytearray()
        self._chunks_in_buffer = 0
        self._silence_chunks = 0
        self._speech_detected = False

        # Stats tracking
        self._segment_count = 0
        self._hallucination_count = 0
        self._whisper_latencies = []
        self._last_stats_time = 0.0

    @property
    def running(self):
        """Whether the continuous STT loop is active."""
        return self._running

    @property
    def stats(self):
        """Return current stats dict."""
        avg_latency = (
            sum(self._whisper_latencies) / len(self._whisper_latencies)
            if self._whisper_latencies else 0.0
        )
        result = {
            'segment_count': self._segment_count,
            'hallucination_count': self._hallucination_count,
            'avg_latency_ms': avg_latency,
            'buffer_depth': len(self._transcript_buffer),
        }
        if self._vram_monitor:
            try:
                result['vram_level'] = self._vram_monitor.check()
                vram_stats = self._vram_monitor.get_stats()
                result['vram_used_mb'] = vram_stats['used_mb']
                result['vram_utilization_pct'] = vram_stats['utilization_pct']
            except Exception:
                result['vram_level'] = 'unknown'
        return result

    def _resolve_device(self):
        """Try AEC device, fall back to default mic (None).

        Returns the device_name string to use with pasimple, or None for default.
        """
        if not self._aec_device_name:
            return None

        try:
            import pasimple
            test_pa = pasimple.PaSimple(
                pasimple.PA_STREAM_RECORD,
                pasimple.PA_SAMPLE_S16LE,
                CHANNELS, SAMPLE_RATE,
                app_name='push-to-talk-cstt',
                device_name=self._aec_device_name,
            )
            test_pa.read(CHUNK_SIZE)
            del test_pa
            print(f"ContinuousSTT: Using echo-cancelled source '{self._aec_device_name}'", flush=True)
            return self._aec_device_name
        except Exception as e:
            print(f"ContinuousSTT: AEC source '{self._aec_device_name}' not available ({e}), using default mic", flush=True)
            return None

    def _load_vad_model(self):
        """Load Silero VAD ONNX model for speech detection."""
        model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
        if not model_path.exists():
            print(f"ContinuousSTT: VAD model not found at {model_path}", flush=True)
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
            print("ContinuousSTT: Silero VAD loaded", flush=True)
            return True
        except Exception as e:
            print(f"ContinuousSTT: Failed to load VAD: {e}", flush=True)
            return False

    def _run_vad(self, audio_bytes):
        """Run VAD inference on audio chunk, return max speech probability.

        Replicates the same VAD processing as live_session.py _run_vad.
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
            print(f"ContinuousSTT VAD error: {e}", flush=True)
            return 0.0

    def _whisper_transcribe(self, pcm_data):
        """Transcribe PCM audio using faster-whisper (blocking, run in executor).

        Uses distil-large-v3 with int8_float16 quantization. Applies multi-layer
        segment filtering (no_speech_prob, avg_logprob, compression_ratio).

        Returns transcribed text or None.
        """
        try:
            import numpy as np

            if not self._whisper_model:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel(
                    WHISPER_MODEL_NAME, device="cuda", compute_type="int8_float16"
                )

            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            segments_gen, info = self._whisper_model.transcribe(
                samples, language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
            )

            kept = []
            for s in segments_gen:
                if s.no_speech_prob >= 0.6:
                    continue
                if s.avg_logprob < -1.0:
                    continue
                if s.compression_ratio > 2.4:
                    continue
                kept.append(s.text.strip())

            text = " ".join(kept).strip()
            return text if text else None

        except Exception as e:
            print(f"ContinuousSTT Whisper error: {e}", flush=True)
            return None

    async def start(self):
        """Begin the continuous capture + VAD + transcription loop."""
        self._running = True
        self._stop_event.clear()
        self._last_stats_time = time.time()

        # Load VAD model
        self._load_vad_model()

        # Resolve audio device
        device_name = self._resolve_device()

        # Start capture thread
        loop = asyncio.get_event_loop()
        capture_thread = threading.Thread(
            target=self._capture_thread, args=(device_name, loop), daemon=True
        )
        capture_thread.start()

        print("ContinuousSTT: Started", flush=True)

        # Main processing loop
        try:
            await self._process_loop()
        except Exception as e:
            print(f"ContinuousSTT: Loop error: {e}", flush=True)
        finally:
            self._running = False
            print("ContinuousSTT: Stopped", flush=True)

    def stop(self):
        """Signal the loop to exit gracefully."""
        self._running = False
        self._stop_event.set()

    def set_playing_audio(self, playing):
        """Called by LiveSession when TTS playback starts/stops.

        When playing, audio chunks are discarded to prevent transcribing
        the AI's own speech. A brief cooldown after playback ends avoids
        catching the tail end of TTS output.
        """
        self._playing_audio = playing
        if not playing:
            self._playback_end_time = time.time()

    def _capture_thread(self, device_name, loop):
        """Record audio in a daemon thread, push to async queue."""
        import pasimple

        while not self._stop_event.is_set():
            try:
                with pasimple.PaSimple(
                    pasimple.PA_STREAM_RECORD,
                    pasimple.PA_SAMPLE_S16LE,
                    CHANNELS, SAMPLE_RATE,
                    app_name='push-to-talk-cstt',
                    device_name=device_name,
                ) as pa:
                    while not self._stop_event.is_set():
                        data = pa.read(CHUNK_SIZE)

                        def _enqueue(d=data):
                            try:
                                self._audio_q.put_nowait(d)
                            except asyncio.QueueFull:
                                pass  # Drop frame rather than block

                        loop.call_soon_threadsafe(_enqueue)
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"ContinuousSTT: Capture error: {e}, reconnecting...", flush=True)
                    time.sleep(1)

    async def _process_loop(self):
        """Main VAD + transcription loop."""
        while self._running:
            try:
                audio_data = self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                self._maybe_emit_stats()
                continue

            self._process_chunk(audio_data)
            self._maybe_emit_stats()

    async def _process_loop_once_for_test(self, num_chunks):
        """Process exactly num_chunks chunks for testing (synchronous drain)."""
        for _ in range(num_chunks):
            try:
                audio_data = self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._process_chunk(audio_data)

    def _process_chunk(self, audio_data):
        """Process a single audio chunk through VAD and manage speech buffer."""
        # Suppress during TTS playback + cooldown to avoid transcribing AI speech
        if self._playing_audio:
            self._reset_buffer()
            return
        if time.time() - self._playback_end_time < self._PLAYBACK_COOLDOWN:
            self._reset_buffer()
            return

        vad_prob = self._run_vad(audio_data)

        if vad_prob > VAD_THRESHOLD:
            # Speech detected
            self._audio_buffer.extend(audio_data)
            self._chunks_in_buffer += 1
            self._silence_chunks = 0
            self._speech_detected = True
        else:
            # Silence
            if self._speech_detected:
                # Still accumulating post-speech silence
                self._audio_buffer.extend(audio_data)
                self._chunks_in_buffer += 1
                self._silence_chunks += 1

                if self._silence_chunks >= SILENCE_CHUNKS_THRESHOLD:
                    # End of utterance -- transcribe
                    if self._chunks_in_buffer >= MIN_BUFFER_CHUNKS:
                        self._do_transcribe()
                    self._reset_buffer()

        # Safety cap: force transcription after max buffer
        if self._chunks_in_buffer >= MAX_BUFFER_CHUNKS:
            self._do_transcribe()
            self._reset_buffer()

    def _do_transcribe(self):
        """Transcribe the current audio buffer and process the result."""
        if not self._audio_buffer:
            return

        pcm_data = bytes(self._audio_buffer)
        t0 = time.time()
        transcript = self._whisper_transcribe(pcm_data)
        latency_ms = (time.time() - t0) * 1000
        self._whisper_latencies.append(latency_ms)

        # Keep only recent latencies for averaging
        if len(self._whisper_latencies) > 100:
            self._whisper_latencies = self._whisper_latencies[-50:]

        if transcript and not is_hallucination(transcript):
            segment = TranscriptSegment(
                text=transcript,
                timestamp=time.time(),
                source="user",
            )
            self._transcript_buffer.append(segment)
            self._segment_count += 1
            print(f"STT [continuous]: {transcript}", flush=True)

            if self._on_segment:
                self._on_segment(segment)
        elif transcript:
            self._hallucination_count += 1
            print(f"STT [continuous]: Rejected hallucination: \"{transcript}\"", flush=True)

    def _reset_buffer(self):
        """Reset the speech accumulation state."""
        self._audio_buffer.clear()
        self._chunks_in_buffer = 0
        self._silence_chunks = 0
        self._speech_detected = False

    def _maybe_emit_stats(self):
        """Emit stats periodically via callback."""
        now = time.time()
        if now - self._last_stats_time >= STATS_INTERVAL_SECONDS:
            self._last_stats_time = now
            if self._on_stats:
                self._on_stats(self.stats)
