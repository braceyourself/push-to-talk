# Architecture: Deepgram Streaming STT + Local Decision Model Integration

**Domain:** Always-on voice assistant -- Deepgram streaming replaces local Whisper for STT
**Researched:** 2026-02-22
**Overall Confidence:** HIGH (codebase fully read, Deepgram SDK docs verified, existing patterns mapped)

## Executive Summary

This document maps how Deepgram streaming STT integrates with the existing push-to-talk pipeline. The pivot replaces local Whisper batch-transcribe (~1.5-3s latency) with Deepgram Nova-3 WebSocket streaming (~150ms word-level results). The integration touches three files significantly: `continuous_stt.py` is replaced by a new `deepgram_stt.py`, `live_session.py` rewires stage 2 to consume Deepgram output, and `transcript_buffer.py` gains minor additions. Everything downstream (LLM stage, StreamComposer, playback, event bus) is unchanged.

The critical architectural insight: the existing `ContinuousSTT` class already has the right interface -- it produces `TranscriptSegment` objects into a `TranscriptBuffer`. The new `DeepgramSTT` class keeps this exact interface but replaces the internals (Whisper inference with Deepgram WebSocket, local VAD-gated silence detection with Deepgram's server-side endpointing). The `_stt_stage()` method in `live_session.py` also has a parallel code path that does VAD + Whisper -- this gets replaced by consuming `DeepgramSTT` output.

The second insight: Deepgram's `speech_final` flag maps directly to the existing `END_OF_UTTERANCE` frame type. When Deepgram signals `speech_final=True`, we emit `END_OF_UTTERANCE` + `TRANSCRIPT` into `_stt_out_q`, exactly as the Whisper path does today. The LLM stage does not know or care that the transcript came from Deepgram instead of Whisper.

## Current Architecture (What Exists Today)

```
 PIPELINE STAGES (asyncio.gather in run(), live_session.py:2951-2959)
 =====================================================================

 Stage 1: _audio_capture_stage()     [daemon thread -> asyncio Queue]
           pasimple.read(4096) at 24kHz 16-bit mono
           -> PipelineFrame(AUDIO_RAW) -> _audio_in_q (maxsize=100)

 Stage 2: _stt_stage()               [asyncio coroutine]
           Reads _audio_in_q
           VAD + silence detection + energy thresholds
           Whisper transcription (run_in_executor)
           -> PipelineFrame(END_OF_UTTERANCE) -> _stt_out_q
           -> PipelineFrame(TRANSCRIPT, data=text) -> _stt_out_q

 Stage 3: _llm_stage()               [asyncio coroutine]
           Reads _stt_out_q
           Sends transcript to Claude CLI subprocess
           Streams response text -> StreamComposer

 Stage 4: _composer.run()            [asyncio coroutine]
           Receives AudioSegment(TTS_SENTENCE)
           Calls Piper TTS for each sentence
           -> PipelineFrame(TTS_AUDIO) -> _audio_out_q

 Stage 5: _playback_stage()          [asyncio coroutine]
           Reads _audio_out_q
           Writes PCM to PyAudio callback stream

 Parallel: _continuous_stt.start()   [daemon thread + asyncio processing]
            Separate pasimple capture
            VAD + Whisper -> TranscriptSegment -> TranscriptBuffer
            (Phase 12 always-on layer, currently runs alongside)

 Parallel: interrupt_loop()          [asyncio coroutine, 50ms poll]
 Parallel: _sse_server_stage()       [asyncio TCP server]
```

### Key Files and Integration Points

| File | Lines | Role |
|------|-------|------|
| `live_session.py` | ~3050 | Pipeline orchestrator, all 5 stages |
| `continuous_stt.py` | ~437 | ContinuousSTT class (BEING REPLACED) |
| `transcript_buffer.py` | ~168 | TranscriptSegment, TranscriptBuffer, is_hallucination() (KEEPING) |
| `pipeline_frames.py` | ~29 | FrameType enum, PipelineFrame dataclass (KEEPING) |
| `event_bus.py` | ~335 | JSONL event bus (KEEPING) |
| `stream_composer.py` | ~80+ | Unified audio output queue (KEEPING) |
| `vram_monitor.py` | ~50+ | GPU VRAM monitoring (KEEPING, less critical with Whisper gone) |
| `openai_realtime.py` | ~439 | Reference WebSocket patterns (READ-ONLY reference) |

### Current Data Flow for a Single Utterance

```
User speaks "What time is it?"
  |
  v
_audio_capture_stage (thread) -> PipelineFrame(AUDIO_RAW, data=4096 bytes)
  |                                          |
  v                                          v
_audio_in_q ----+-----------------------+---> _stt_stage()
                |                       |
                |  (muted? skip)        |  (stt_gated? skip, but run VAD for barge-in)
                |                       |
                |                       v
                |              RMS silence detection + energy thresholds
                |              Accumulate until silence (0.8s) or safety cap (10s)
                |                       |
                |                       v
                |              Whisper transcribe (run_in_executor, 500ms-2s)
                |              Hallucination filter
                |                       |
                |                       v
                |              PipelineFrame(END_OF_UTTERANCE) -> _stt_out_q
                |              PipelineFrame(TRANSCRIPT, "What time is it?") -> _stt_out_q
                |
                v
        _continuous_stt (separate capture, separate VAD, separate Whisper)
        -> TranscriptSegment -> TranscriptBuffer (Phase 12)
```

### Current ContinuousSTT Interface (continuous_stt.py)

This is the interface we must replicate:

```python
class ContinuousSTT:
    def __init__(self, transcript_buffer, vram_monitor=None,
                 aec_device_name=None, on_segment=None, on_stats=None):
        # transcript_buffer: TranscriptBuffer to append segments to
        # on_segment: callback(TranscriptSegment) for each new segment
        # on_stats: callback(dict) for periodic stats

    async def start(self):       # Begin capture + transcription loop
    def stop(self):              # Signal graceful shutdown
    def set_playing_audio(self, playing):  # Gate during TTS playback + cooldown

    @property
    def running(self) -> bool:
    @property
    def stats(self) -> dict:
```

Used by `live_session.py`:
- Line 2930: `self._continuous_stt = ContinuousSTT(transcript_buffer=..., on_segment=..., on_stats=...)`
- Line 2959: `self._continuous_stt.start()` (in asyncio.gather)
- Line 2642-2643: `self._continuous_stt.set_playing_audio(False)` (on playback end)
- Line 2674-2675: `self._continuous_stt.set_playing_audio(True)` (on playback start)
- Line 2728-2729: `self._continuous_stt.set_playing_audio(False)` (on barge-in)
- Line 2788-2789: `self._continuous_stt.set_playing_audio(False)` (post barge-in)
- Line 2970-2971: `self._continuous_stt.stop()` (cleanup)

## Target Architecture (After Deepgram Integration)

```
 PIPELINE STAGES (modified)
 =====================================================================

 Stage 1: _audio_capture_stage()     [UNCHANGED]
           pasimple.read(4096) at 24kHz 16-bit mono
           -> PipelineFrame(AUDIO_RAW) -> _audio_in_q

 Stage 2: _stt_stage()               [REWRITTEN - consumes DeepgramSTT output]
           No longer does VAD/silence/Whisper internally
           Reads from deepgram_transcript_q (fed by DeepgramSTT callbacks)
           Emits same END_OF_UTTERANCE + TRANSCRIPT frames
           -> _stt_out_q (SAME interface to LLM stage)

 Stage 3: _llm_stage()               [UNCHANGED]
 Stage 4: _composer.run()            [UNCHANGED]
 Stage 5: _playback_stage()          [UNCHANGED]

 Parallel: _deepgram_stt.start()     [NEW - replaces _continuous_stt.start()]
            DeepgramSTT class
            Reads audio from _audio_in_q (shared with Stage 2)
              OR has its own pasimple capture (see design choice below)
            VAD gates what reaches Deepgram WebSocket
            Deepgram callbacks -> TranscriptSegment -> TranscriptBuffer
            Deepgram callbacks -> _deepgram_transcript_q -> STT stage

 Parallel: interrupt_loop()          [UNCHANGED]
 Parallel: _sse_server_stage()       [UNCHANGED]

 REMOVED: _continuous_stt.start() (Whisper-based)
 REMOVED: _stt_whisper_fallback() method (lines 1406-1457)
 REMOVED: _whisper_transcribe() method (lines 1459-1517)
```

### Target Data Flow

```
User speaks "What time is it?"
  |
  v
_audio_capture_stage (thread) -> PipelineFrame(AUDIO_RAW, data=4096 bytes)
  |
  v
_audio_in_q (maxsize=100)
  |
  +--> _stt_stage() [NEW: thin consumer, reads deepgram_transcript_q]
  |        |
  |        |  Reads from _deepgram_transcript_q (populated by DeepgramSTT)
  |        |  Checks _stt_gated (suppress during playback)
  |        |  Checks muted
  |        |  Emits END_OF_UTTERANCE + TRANSCRIPT on speech_final
  |        |
  |        v
  |    _stt_out_q -> _llm_stage() [UNCHANGED from here]
  |
  +--> DeepgramSTT._audio_forwarder() [NEW]
           |
           |  Reads AUDIO_RAW frames from _audio_in_q (or own capture)
           |  Runs Silero VAD on each chunk
           |  If speech: sends raw PCM to Deepgram WebSocket
           |  If silence: sends KeepAlive every 5s
           |
           v
       Deepgram WebSocket (wss://api.deepgram.com/v1/listen)
           |  model=nova-3, encoding=linear16, sample_rate=24000
           |  interim_results=true, endpointing=300, utterance_end_ms=1000
           |  smart_format=true, vad_events=true
           |
           v
       Deepgram Callbacks:
           on_message -> parse is_final, speech_final
               interim (is_final=false): log/display only
               final (is_final=true): accumulate text
               speech_final (is_final=true, speech_final=true):
                   -> flush accumulated -> TranscriptSegment
                   -> TranscriptBuffer.append()
                   -> _deepgram_transcript_q.put()
                   -> on_segment callback
           on_utterance_end -> secondary end-of-speech signal
           on_speech_started -> emit event bus "stt_start"
           on_error -> log, trigger reconnect
           on_close -> reconnect with backoff
```

## New Component: DeepgramSTT (deepgram_stt.py)

### Class Design

```python
"""Deepgram streaming STT with VAD cost-gating.

Replaces ContinuousSTT. Same external interface:
- Produces TranscriptSegment objects into a TranscriptBuffer
- on_segment/on_stats callbacks
- set_playing_audio() for playback suppression
- start()/stop() lifecycle

Key difference: STT inference is cloud-based (Deepgram WebSocket),
not local (Whisper). Silero VAD runs locally to gate what audio
reaches Deepgram (cost saving, not latency saving).
"""

import asyncio
import time
import threading
from pathlib import Path
from dataclasses import dataclass

from deepgram import DeepgramClient, AsyncDeepgramClient
from deepgram.core.events import EventType

from transcript_buffer import TranscriptBuffer, TranscriptSegment, is_hallucination

# Audio settings (match live_session.py)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4096  # ~85ms at 24kHz 16-bit mono
BYTES_PER_SAMPLE = 2

# VAD settings
VAD_THRESHOLD = 0.5

# Deepgram connection settings
KEEPALIVE_INTERVAL = 5.0  # seconds between KeepAlive when no speech
RECONNECT_DELAY_BASE = 1.0  # seconds, doubles on each retry
RECONNECT_MAX_DELAY = 30.0
MAX_RECONNECT_ATTEMPTS = 10

# Stats reporting
STATS_INTERVAL_SECONDS = 5.0


class DeepgramSTT:
    """VAD-gated Deepgram streaming STT producing TranscriptSegments.

    Architecture:
    - Audio capture: reads from shared _audio_in_q OR own pasimple capture
    - VAD: Silero ONNX (CPU, <1ms per chunk) gates audio to Deepgram
    - Deepgram: WebSocket streaming, Nova-3 model
    - Output: TranscriptSegment -> TranscriptBuffer, callbacks, _transcript_q

    Args:
        api_key: Deepgram API key
        transcript_buffer: TranscriptBuffer to append segments to
        transcript_q: asyncio.Queue for pipeline integration (STT stage reads this)
        aec_device_name: PipeWire AEC source name (None = default mic)
        on_segment: callback(TranscriptSegment) for each new segment
        on_stats: callback(dict) for periodic stats
    """

    def __init__(self, api_key, transcript_buffer, transcript_q=None,
                 aec_device_name=None, on_segment=None, on_stats=None):
        self._api_key = api_key
        self._transcript_buffer = transcript_buffer
        self._transcript_q = transcript_q  # For pipeline STT stage
        self._aec_device_name = aec_device_name
        self._on_segment = on_segment
        self._on_stats = on_stats

        self._running = False
        self._stop_event = threading.Event()

        # Playback suppression (same interface as ContinuousSTT)
        self._playing_audio = False
        self._playback_end_time = 0.0
        self._PLAYBACK_COOLDOWN = 0.3  # Shorter than Whisper -- Deepgram handles echo better

        # Deepgram connection
        self._dg_connection = None
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
        self._latencies = []  # Time from speech_final to segment emission
        self._last_stats_time = 0.0
        self._speech_start_time = 0.0  # When Deepgram detected speech start

    # ── Public interface (matches ContinuousSTT) ──

    @property
    def running(self):
        return self._running

    @property
    def stats(self):
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
            # Main loop: connect to Deepgram, forward audio, handle reconnects
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

    # ── Connection lifecycle ──

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
                delay = min(
                    RECONNECT_DELAY_BASE * (2 ** (self._reconnect_attempts - 1)),
                    RECONNECT_MAX_DELAY
                )
                print(f"DeepgramSTT: Connection failed ({e}), "
                      f"retry {self._reconnect_attempts} in {delay:.1f}s", flush=True)
                if self._reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    print("DeepgramSTT: Max reconnect attempts reached", flush=True)
                    break
                await asyncio.sleep(delay)

    async def _connect_and_stream(self):
        """Connect to Deepgram, register callbacks, forward audio."""
        client = DeepgramClient(self._api_key)

        # Use synchronous client with context manager
        # (Deepgram SDK manages its own WebSocket thread)
        dg_connection = client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            interim_results=True,
            endpointing=300,            # 300ms pause = endpoint
            utterance_end_ms=1000,      # 1s gap between words = utterance end
            smart_format=True,          # Punctuation, numerals
            vad_events=True,            # Emit speech start/end events
        )

        # Register event handlers
        dg_connection.on(EventType.OPEN, self._on_open)
        dg_connection.on(EventType.MESSAGE, self._on_message)
        dg_connection.on(EventType.CLOSE, self._on_close)
        dg_connection.on(EventType.ERROR, self._on_error)
        # Additional events if available in SDK:
        # dg_connection.on(EventType.SPEECH_STARTED, self._on_speech_started)
        # dg_connection.on(EventType.UTTERANCE_END, self._on_utterance_end)

        self._dg_connection = dg_connection

        # Start the WebSocket listener (SDK manages internally)
        dg_connection.start_listening()

        self._connected = True
        self._reconnect_attempts = 0
        print("DeepgramSTT: Connected to Deepgram Nova-3", flush=True)

        try:
            # Forward audio from capture queue to Deepgram
            await self._audio_forward_loop()
        finally:
            # Clean up connection
            try:
                dg_connection.finish()
            except Exception:
                pass
            self._connected = False
            self._dg_connection = None

    # ── Deepgram event handlers ──

    def _on_open(self, _):
        print("DeepgramSTT: WebSocket opened", flush=True)

    def _on_message(self, result):
        """Handle Deepgram transcription results.

        Deepgram sends three types of results:
        1. interim (is_final=False): partial/speculative transcript
        2. final (is_final=True, speech_final=False): confirmed text chunk
        3. speech_final (is_final=True, speech_final=True): end of utterance

        We accumulate is_final chunks and flush on speech_final,
        matching the pattern already validated in test_live_session.py.
        """
        try:
            channel = result.channel
            alternatives = channel.alternatives
            if not alternatives:
                return

            transcript = alternatives[0].transcript.strip()
            if not transcript:
                return

            is_final = result.is_final
            speech_final = result.speech_final

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

    def _on_close(self, _):
        print("DeepgramSTT: WebSocket closed", flush=True)
        self._connected = False

    def _on_error(self, error):
        print(f"DeepgramSTT: Error: {error}", flush=True)

    # ── Transcript emission ──

    def _emit_transcript(self, text):
        """Process a complete transcript and emit to all consumers."""
        # Playback suppression: discard transcripts of AI's own speech
        if self._playing_audio:
            return
        if time.time() - self._playback_end_time < self._PLAYBACK_COOLDOWN:
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

    # ── Audio forwarding ──

    async def _audio_forward_loop(self):
        """Read audio from capture queue, VAD-gate, forward to Deepgram."""
        while self._running and self._connected:
            try:
                audio_data = self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)

                # Send KeepAlive if no audio sent recently
                if (time.time() - self._last_audio_sent_time > KEEPALIVE_INTERVAL
                        and self._dg_connection):
                    try:
                        self._dg_connection.keep_alive()
                    except Exception:
                        pass
                    self._last_audio_sent_time = time.time()

                self._maybe_emit_stats()
                continue

            # Run VAD on chunk
            vad_prob = self._run_vad(audio_data)

            if vad_prob > VAD_THRESHOLD:
                # Speech detected -- forward to Deepgram
                if self._dg_connection:
                    try:
                        self._dg_connection.send(audio_data)
                        self._last_audio_sent_time = time.time()
                    except Exception as e:
                        print(f"DeepgramSTT: Send error: {e}", flush=True)
                        self._connected = False
                        break
            else:
                # Silence -- send KeepAlive periodically
                if (time.time() - self._last_audio_sent_time > KEEPALIVE_INTERVAL
                        and self._dg_connection):
                    try:
                        self._dg_connection.keep_alive()
                    except Exception:
                        pass
                    self._last_audio_sent_time = time.time()

            self._maybe_emit_stats()

    # ── Audio capture (same pattern as ContinuousSTT) ──

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

    # ── VAD (same as ContinuousSTT) ──

    def _load_vad_model(self):
        """Load Silero VAD ONNX model."""
        # ... identical to ContinuousSTT._load_vad_model() ...

    def _run_vad(self, audio_bytes):
        """Run VAD, return speech probability."""
        # ... identical to ContinuousSTT._run_vad() ...

    def _resolve_device(self):
        """Try AEC device, fall back to default."""
        # ... identical to ContinuousSTT._resolve_device() ...

    def _maybe_emit_stats(self):
        """Emit stats periodically."""
        now = time.time()
        if now - self._last_stats_time >= STATS_INTERVAL_SECONDS:
            self._last_stats_time = now
            if self._on_stats:
                self._on_stats(self.stats)
```

### Design Decisions

**1. Own audio capture vs shared _audio_in_q**

Decision: **Own capture thread** (same pattern as ContinuousSTT).

Rationale: The existing `_audio_capture_stage()` feeds `_audio_in_q` which is consumed by `_stt_stage()`. If DeepgramSTT also reads from `_audio_in_q`, both consumers compete for the same frames (asyncio.Queue is single-consumer). Options:
- (a) Fan-out: duplicate frames to two queues -- adds complexity, memory
- (b) Single consumer: DeepgramSTT reads `_audio_in_q`, feeds `_stt_stage()` via `_deepgram_transcript_q` -- works but couples audio routing
- (c) Own capture: DeepgramSTT has its own pasimple stream -- simple, proven pattern (ContinuousSTT already does this)

Option (c) is simplest. Two pasimple readers from the same PipeWire source works fine -- PipeWire handles multiple clients. The audio capture stage can eventually be removed when the old `_stt_stage()` Whisper path is fully deprecated, but keeping it initially provides a fallback.

**2. VAD gating at the client vs letting Deepgram handle it**

Decision: **VAD gate at client** (Silero, local).

Rationale: Deepgram charges per audio second streamed. Sending silence costs money. Silero VAD runs on CPU in <1ms per chunk. Only stream speech-positive chunks. During silence, send KeepAlive messages (text frames, not audio) to keep the connection open. This is the documented Deepgram best practice for cost optimization.

**3. Transcript suppression during playback**

Decision: **Suppress transcript output, keep streaming audio.**

Rationale: Unlike ContinuousSTT which discards audio chunks during playback (because Whisper would transcribe the AI's speech from the mic), DeepgramSTT keeps streaming audio. Reasons:
- PipeWire AEC removes the AI's voice from the mic signal
- Deepgram handles residual echo better than local Whisper
- Keeping the connection alive avoids reconnect overhead
- We suppress the *transcript output* (not the audio input) when `_playing_audio` is True

**4. Deepgram SDK sync vs async client**

Decision: **Synchronous client** (`DeepgramClient`, not `AsyncDeepgramClient`).

Rationale: The Deepgram SDK's sync client manages its own WebSocket thread internally. The `on_message` callback fires on that thread. We use `loop.call_soon_threadsafe()` to push transcripts to the asyncio event loop (same pattern used for audio capture). This avoids potential async event loop conflicts between the Deepgram SDK's internal async machinery and our existing asyncio loop. The sync client is also better documented in official examples.

Alternative: If the sync client proves problematic, the async client (`AsyncDeepgramClient` with `client.listen.v2.connect()`) can be used. The key difference is `await connection.start_listening()` instead of `connection.start_listening()`.

**5. Audio format: no resampling needed**

Decision: **Send 24kHz 16-bit mono directly.**

Deepgram's streaming API accepts `linear16` encoding at any sample rate specified in the connection parameters. Our capture is already 24kHz 16-bit mono (`pasimple.PA_SAMPLE_S16LE`, `SAMPLE_RATE=24000`). Pass `sample_rate=24000, encoding="linear16", channels=1` in the connection options. No resampling required.

Note: ContinuousSTT resamples to 16kHz for Silero VAD (which expects 16kHz). This resampling stays for VAD but does NOT affect what we send to Deepgram. VAD gets the resampled audio; Deepgram gets the original 24kHz audio.

## Integration with live_session.py

### Changes to __init__() (line 187-316)

```python
# REMOVE:
self._continuous_stt = None     # line 310

# ADD:
self._deepgram_stt = None       # Replaces _continuous_stt
self._deepgram_transcript_q = None  # DeepgramSTT -> STT stage
```

### Changes to run() (line 2890-2962)

```python
# REMOVE (line 2930-2937):
self._continuous_stt = ContinuousSTT(
    transcript_buffer=self._transcript_buffer,
    vram_monitor=self._vram_monitor,
    aec_device_name=...,
    on_segment=self._on_transcript_segment,
    on_stats=self._on_stt_stats,
)

# ADD:
self._deepgram_transcript_q = asyncio.Queue(maxsize=50)
self._deepgram_stt = DeepgramSTT(
    api_key=self.deepgram_api_key,
    transcript_buffer=self._transcript_buffer,
    transcript_q=self._deepgram_transcript_q,
    aec_device_name=self.config.get("aec_device_name", "Echo Cancellation Source")
        if hasattr(self, 'config') and isinstance(getattr(self, 'config', None), dict)
        else "Echo Cancellation Source",
    on_segment=self._on_transcript_segment,
    on_stats=self._on_stt_stats,
)

# CHANGE stages list (line 2951-2959):
stages = [
    self._audio_capture_stage(),       # Keep (may remove later)
    self._stt_stage(),                 # REWRITTEN to consume deepgram_transcript_q
    self._llm_stage(),                 # UNCHANGED
    self._composer.run(),              # UNCHANGED
    self._playback_stage(),            # UNCHANGED
    interrupt_loop(),                  # UNCHANGED
    self._sse_server_stage(),          # UNCHANGED
    self._deepgram_stt.start(),        # NEW: replaces _continuous_stt.start()
]
```

### Changes to _stt_stage() (line 2190-2394)

The entire method is rewritten. Instead of doing VAD + silence detection + Whisper transcription internally, it becomes a thin consumer of DeepgramSTT output.

```python
async def _stt_stage(self):
    """Consume DeepgramSTT transcripts and emit pipeline frames.

    Reads TranscriptSegment objects from _deepgram_transcript_q (populated
    by DeepgramSTT callbacks). Emits END_OF_UTTERANCE + TRANSCRIPT frames
    into _stt_out_q for the LLM stage.

    Handles:
    - _stt_gated: suppress during playback (VAD barge-in still runs in DeepgramSTT)
    - muted: suppress when user mutes
    - _stt_flush_event: immediate flush on mute toggle
    - generation_id: tag frames for interrupt coherence
    """
    print("STT: Using Deepgram streaming", flush=True)

    try:
        while self.running:
            try:
                segment = await asyncio.wait_for(
                    self._deepgram_transcript_q.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                # Check flush event during idle
                if self._stt_flush_event and self._stt_flush_event.is_set():
                    self._stt_flush_event.clear()
                    # Deepgram handles flushing automatically via speech_final
                    # No local buffer to flush -- just clear the event
                continue

            # Gate checks
            if self.muted:
                continue
            if self._stt_gated:
                self._was_stt_gated = True
                continue

            # Gated -> ungated transition (same logic as current)
            if self._was_stt_gated:
                self._was_stt_gated = False
                # Discard any segments that arrived during gating
                # (they are from AI's own speech or post-echo)
                continue

            transcript = segment.text
            self._emit_event("stt_complete", text=transcript[:60],
                             latency_ms=0, rejected=False)  # Latency tracked by DeepgramSTT
            print(f"STT [deepgram]: {transcript}", flush=True)

            await self._stt_out_q.put(PipelineFrame(
                type=FrameType.END_OF_UTTERANCE,
                generation_id=self.generation_id
            ))
            await self._stt_out_q.put(PipelineFrame(
                type=FrameType.TRANSCRIPT,
                generation_id=self.generation_id,
                data=transcript
            ))
            self._post_barge_in = False

    except Exception as e:
        print(f"STT stage error: {e}", flush=True)
```

### Changes to playback/barge-in (multiple locations)

Replace `self._continuous_stt.set_playing_audio(...)` with `self._deepgram_stt.set_playing_audio(...)`:

- Line 2642-2643: `self._deepgram_stt.set_playing_audio(False)`
- Line 2674-2675: `self._deepgram_stt.set_playing_audio(True)`
- Line 2728-2729: `self._deepgram_stt.set_playing_audio(False)`
- Line 2788-2789: `self._deepgram_stt.set_playing_audio(False)`

### Changes to cleanup (line 2970-2971)

```python
# REMOVE:
if self._continuous_stt:
    self._continuous_stt.stop()

# ADD:
if self._deepgram_stt:
    self._deepgram_stt.stop()
```

### Whisper fallback removal

Remove these methods from live_session.py:
- `_stt_whisper_fallback()` (lines 1406-1457)
- `_whisper_transcribe()` (lines 1459-1517)

The CircuitBreaker `_stt_breaker` (line 294) stays -- it can now gate Deepgram failures and trigger a reconnect or graceful degradation.

## Deepgram Connection Configuration

### Connection Parameters

```python
{
    "model": "nova-3",           # Latest, best accuracy
    "encoding": "linear16",      # 16-bit PCM, matches our capture
    "sample_rate": 24000,        # Our native rate, no resampling
    "channels": 1,               # Mono
    "interim_results": True,     # Get partial results for UI feedback
    "endpointing": 300,          # 300ms silence = end of utterance
    "utterance_end_ms": 1000,    # 1s gap between words = utterance end
    "smart_format": True,        # Auto punctuation, numerals
    "vad_events": True,          # Emit speech start/end events
    "language": "en",            # English
}
```

### Parameter Rationale

**`endpointing=300`**: Default is 10ms (too aggressive for conversation). 300-500ms is recommended for conversational AI per Deepgram docs. This controls when `speech_final=True` fires. 300ms means Deepgram considers a 300ms pause to be an endpoint -- the user finished their thought. This replaces the `SILENCE_DURATION_NORMAL = 0.8` in the current Whisper path (which used 800ms). The trade-off: 300ms may fire too early for users who pause mid-sentence. Can tune to 500ms if needed.

**`utterance_end_ms=1000`**: Secondary end-of-speech detection. Looks at word timing gaps across both interim and final results. When no words appear for 1000ms, fires `UtteranceEnd` event. This is a backup to `endpointing` and works better for detecting the end of a multi-sentence utterance.

**`interim_results=True`**: Required for `utterance_end_ms` to work. Also provides real-time partial transcripts for UI feedback (dashboard can show what the user is saying as they speak, before final results arrive).

**`smart_format=True`**: Deepgram adds punctuation, capitalizes sentences, formats numerals. This means the transcript text arriving in the pipeline is already formatted -- no need for post-processing.

**`vad_events=True`**: Deepgram emits `SpeechStarted` events when it detects the beginning of speech in the audio stream. Used for event bus emission ("stt_start" event) to show "listening" status in the dashboard.

### Audio Format: No Conversion Needed

```
Capture: pasimple.PA_SAMPLE_S16LE at 24000 Hz, 1 channel
         = 16-bit signed little-endian PCM, 24kHz mono
         = "linear16" in Deepgram terminology

Deepgram: encoding="linear16", sample_rate=24000, channels=1
         = exact match

Silero VAD: expects 16kHz input
         = resample 24kHz -> 16kHz (take 2 of every 3 samples)
         = same resampling already done in ContinuousSTT._run_vad()
```

No audio format conversion is needed for Deepgram. The raw `pa.read(4096)` output goes directly to `dg_connection.send(audio_data)` as binary bytes.

## Error Handling

### Deepgram Disconnection

```
Detection: on_close callback fires, OR send() raises exception
Response:
  1. Set _connected = False
  2. Log error with context
  3. Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (capped)
  4. Reconnect: new DeepgramClient, new connection
  5. Must send audio within 10 seconds of reconnect (Deepgram timeout)
  6. Reset reconnect counter on successful connection
Recovery:
  - Audio capture continues (buffered in _audio_q)
  - VAD continues (local, no network dependency)
  - Transcript output pauses until reconnected
  - CircuitBreaker tracks failures for dashboard visibility
```

### Audio Buffering During Reconnect

```
Problem: Audio produced during reconnect is lost
Solution:
  - _audio_q has maxsize=200 (~17 seconds of audio)
  - During reconnect, capture thread keeps filling queue
  - On reconnect, drain queued audio to Deepgram
  - If queue fills, oldest frames dropped (acceptable -- user speech
    during a multi-second outage is stale anyway)
Note: Deepgram processes streaming audio at max 1.25x realtime.
  Sending a large buffer all at once may cause delayed transcripts.
  Consider draining at controlled rate or discarding stale audio.
```

### KeepAlive Protocol

```
Deepgram closes connection if no audio or KeepAlive within 10 seconds.
Implementation:
  - Track _last_audio_sent_time
  - In _audio_forward_loop, if no speech for KEEPALIVE_INTERVAL (5s):
    - Call dg_connection.keep_alive() (SDK method)
    - This sends {"type": "KeepAlive"} as a text WebSocket frame
  - CRITICAL: KeepAlive must be TEXT frame, not binary
    The SDK's keep_alive() method handles this correctly
```

### Rate Limits

```
Deepgram rate limits (per account):
  - Concurrent connections: varies by plan (typically 25-100)
  - This system uses 1 connection -- not a concern
  - Audio throughput: max 1.25x realtime -- not a concern (we stream realtime)
  - Cost: ~$0.0044/minute for Nova-3 = ~$0.26/hour
    With VAD gating (only stream speech), actual cost is lower
```

### API Key Rotation

```
Deepgram API keys don't expire automatically.
Key is read from environment or file at startup:
  - $DEEPGRAM_API_KEY env var
  - ~/.config/deepgram/api_key file
  - ~/.deepgram/api_key file
  (See push-to-talk.py get_deepgram_api_key(), line 126-137)

To rotate: update the key source, restart the service.
No in-flight rotation needed -- single long-lived connection.
```

## Thread/Async Model

```
┌──────────────────────────────────────────────────────────────────┐
│                    ASYNCIO EVENT LOOP (main)                      │
│                                                                    │
│  Coroutines:                                                       │
│  - _audio_capture_stage()    reads _audio_in_q (from thread)     │
│  - _stt_stage()              reads _deepgram_transcript_q        │
│  - _llm_stage()              reads _stt_out_q                    │
│  - _composer.run()           reads segment_q                     │
│  - _playback_stage()         reads _audio_out_q                  │
│  - _deepgram_stt.start()     manages Deepgram connection         │
│    ├─ _connection_loop()     reconnect manager                   │
│    └─ _audio_forward_loop()  VAD + send to Deepgram              │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                     DAEMON THREADS                                 │
│                                                                    │
│  Thread 1: Audio capture (live_session._audio_capture_stage)     │
│            pasimple.read() -> loop.call_soon_threadsafe()         │
│            -> PipelineFrame(AUDIO_RAW) -> _audio_in_q            │
│                                                                    │
│  Thread 2: DeepgramSTT._capture_thread()                         │
│            pasimple.read() -> loop.call_soon_threadsafe()         │
│            -> raw bytes -> _audio_q                               │
│                                                                    │
│  Thread 3: Deepgram SDK internal WebSocket thread                │
│            Managed by DeepgramClient.listen.v1.connect()         │
│            Fires callbacks: _on_message, _on_close, _on_error    │
│            Callbacks run on THIS thread -- must not block         │
│            Use loop.call_soon_threadsafe() for asyncio work       │
│                                                                    │
│  Thread 4: PyAudio callback thread (playback)                    │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                     SUBPROCESSES                                   │
│                                                                    │
│  Claude CLI (stdio IPC)                                           │
│  Learner daemon (reads events.jsonl)                              │
│  Clip factory (generates filler clips)                            │
│  Input classifier (Unix socket, heuristic + model2vec)           │
└──────────────────────────────────────────────────────────────────┘
```

### Critical Threading Concern

Deepgram SDK's `on_message` callback fires on the SDK's internal WebSocket thread, NOT on the asyncio event loop. The callback must be fast and non-blocking. It should NOT:
- `await` anything
- Call `asyncio.Queue.put()` (wrong thread)
- Do heavy computation

Instead, use `loop.call_soon_threadsafe()` to schedule work on the event loop:

```python
def _on_message(self, result):
    # Parse on the WebSocket thread (fast)
    transcript = result.channel.alternatives[0].transcript.strip()
    is_final = result.is_final
    speech_final = result.speech_final

    # Schedule emission on the event loop
    if is_final and speech_final and transcript:
        self._loop.call_soon_threadsafe(
            self._emit_transcript_threadsafe, transcript
        )
```

Alternative: use `asyncio.run_coroutine_threadsafe()` if the emission needs to be async (e.g., to put on an asyncio.Queue).

## What Gets Deleted

### Files to Remove

| File | Status | Replacement |
|------|--------|-------------|
| `continuous_stt.py` | DELETE entirely | `deepgram_stt.py` |

### Code to Remove from live_session.py

| Lines | What | Why |
|-------|------|-----|
| 32 | `from continuous_stt import ContinuousSTT` | Replaced by `from deepgram_stt import DeepgramSTT` |
| 310 | `self._continuous_stt = None` | Replaced by `self._deepgram_stt = None` |
| 1406-1457 | `_stt_whisper_fallback()` | Deepgram replaces Whisper |
| 1459-1517 | `_whisper_transcribe()` | Deepgram replaces Whisper |
| 2190-2394 | `_stt_stage()` (entire body) | Rewritten as thin Deepgram consumer |
| 2930-2937 | ContinuousSTT initialization | Replaced by DeepgramSTT initialization |

### Dependencies to Remove from requirements.txt

| Package | Why Remove |
|---------|-----------|
| `faster-whisper` | No longer needed -- Deepgram replaces Whisper |

Note: `onnxruntime` stays (used by Silero VAD). `deepgram-sdk>=3.0` already in requirements.txt.

### Dependencies to Keep

| Package | Why Keep |
|---------|---------|
| `deepgram-sdk>=3.0` | Already listed, needed for DeepgramSTT |
| `onnxruntime>=1.18` | Silero VAD (local, used for cost gating) |
| `pasimple` | Audio capture (PipeWire/PulseAudio) |
| `numpy` | VAD audio processing |

## Suggested Build Order

### Step 1: Create deepgram_stt.py (independently testable)

Write the `DeepgramSTT` class with:
- Own audio capture thread (pasimple)
- Silero VAD gating
- Deepgram WebSocket connection
- `on_message` callback with `is_final`/`speech_final` accumulation
- `TranscriptSegment` emission to `TranscriptBuffer`
- `_transcript_q` output for pipeline integration
- KeepAlive management
- Reconnection with exponential backoff

Test: Run standalone, pipe mic audio through it, verify transcripts appear.
No changes to live_session.py needed yet.

### Step 2: Rewrite _stt_stage() to consume Deepgram output

Replace the Whisper-based `_stt_stage()` with the thin consumer that reads from `_deepgram_transcript_q`. Wire up `DeepgramSTT` in `__init__()` and `run()`.

Test: Full pipeline with DeepgramSTT feeding LLM stage. Verify end-to-end voice conversation works.

### Step 3: Clean up old code

- Remove `_stt_whisper_fallback()` and `_whisper_transcribe()` from live_session.py
- Delete `continuous_stt.py`
- Remove `faster-whisper` from requirements.txt
- Update imports
- Update tests in `test_live_session.py`

### Step 4: Decision model integration

Wire the local decision model to read from `TranscriptBuffer` (populated by DeepgramSTT). This is a separate concern from STT integration and should be a separate phase.

## Scalability Considerations

| Concern | Current (1 user) | Future (multi-session) |
|---------|-------------------|----------------------|
| Deepgram connections | 1 WebSocket | 1 per session (Deepgram allows concurrent) |
| Cost | ~$0.08/hr with VAD gating | Scales linearly |
| Latency | ~150ms (Deepgram) | Same -- server-side, not affected by client count |
| VRAM | Freed ~1.5GB (no Whisper) | More room for decision model |
| Network dependency | NEW: requires internet | Whisper fallback possible but not planned |

## Sources

- **Codebase analysis** (HIGH confidence): `live_session.py`, `continuous_stt.py`, `transcript_buffer.py`, `pipeline_frames.py`, `event_bus.py`, `openai_realtime.py` -- all read in full
- **Deepgram streaming docs** (HIGH confidence): [Getting Started with Live Streaming](https://developers.deepgram.com/docs/live-streaming-audio), [Audio Keep Alive](https://developers.deepgram.com/docs/audio-keep-alive), [Recovering from Connection Errors](https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio)
- **Deepgram API reference** (HIGH confidence): [Live Audio API](https://developers.deepgram.com/reference/speech-to-text/listen-streaming) -- encoding, sample_rate, endpointing, interim_results, speech_final parameters
- **Deepgram endpointing docs** (HIGH confidence): [Configure Endpointing and Interim Results](https://developers.deepgram.com/docs/understand-endpointing-interim-results) -- is_final vs speech_final semantics, 300ms default recommendation
- **Deepgram Python SDK** (MEDIUM confidence): [GitHub repo](https://github.com/deepgram/deepgram-python-sdk), [PyPI deepgram-sdk](https://pypi.org/project/deepgram-sdk/) -- SDK v3+ API surface, EventType enum, sync vs async clients
- **Deepgram SDK API docs** (MEDIUM confidence): [Live client API](https://deepgram.github.io/deepgram-python-sdk/docs/v3/deepgram/clients/live/v1/client.html) -- LiveOptions, response types, event handler signatures
- **Existing test patterns** (HIGH confidence): `test_live_session.py` lines 132-185 already test Deepgram `on_message` callback patterns with `is_final`/`speech_final` accumulation
