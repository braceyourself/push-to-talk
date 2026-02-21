# Architecture: v2.0 Decoupled Input Stream + LLM Observer

**Domain:** Always-on voice assistant with independent input capture, monitoring layer, and configurable response backend
**Researched:** 2026-02-21
**Overall Confidence:** HIGH (codebase fully read, existing pipeline thoroughly understood, patterns verified)

## Executive Summary

The v2.0 architecture transforms the current 5-stage sequential pipeline into a decoupled system with three independent loops: (1) an always-on input stream that continuously captures audio and produces transcripts, (2) a monitoring loop that watches the transcript stream via a local LLM and decides when and how to respond, and (3) a response generation layer that produces the actual reply through either Claude CLI or Ollama.

The critical insight from reading the existing codebase: the current pipeline is already partially decoupled. Audio capture runs in a daemon thread, STT runs in a thread executor, and the LLM stage consumes from a queue independently. The main coupling point is the `_stt_gated` flag -- STT is suppressed during AI playback because the system assumes a strict turn-taking model (user speaks, AI responds, user waits). Removing this assumption is the single biggest architectural change. Everything else -- composer, playback, barge-in, filler system, event bus -- can remain largely unchanged.

The second insight: GPU memory is the binding constraint. The RTX 3070 has 8GB VRAM. Whisper large-v3 uses ~3-4GB. Ollama with Llama 3.2 3B at int4 uses ~2GB. Together they fit, but switching Whisper to the `small` or `medium` model (0.5-1.5GB) gives comfortable headroom and makes continuous operation sustainable. Continuous Whisper transcription fundamentally changes the GPU utilization profile from burst (transcribe on silence) to sustained (transcribe every few seconds).

## Current Architecture (v1.x)

```
┌─────────────────────────────────────────────────────────────────┐
│                     SEQUENTIAL PIPELINE                         │
│                                                                 │
│  ┌──────────┐  audio_in_q  ┌──────────┐  stt_out_q  ┌───────┐ │
│  │  Audio   ├─────────────>│   STT    ├────────────>│  LLM  │ │
│  │ Capture  │              │ (Whisper) │             │(Claude│ │
│  │(PulseAudio)             │ +VAD     │             │  CLI) │ │
│  └──────────┘              └──────────┘             └───┬───┘ │
│       ^                        ^                        │     │
│       │                        │ _stt_gated             │     │
│       │                        │ (suppressed             │     │
│       │                        │  during playback)       v     │
│  ┌──────────┐              ┌──────────┐           ┌─────────┐ │
│  │ Playback │<─────────────│ Composer │<──────────│ Filler  │ │
│  │(PyAudio) │  audio_out_q │(TTS+queue)│          │ Manager │ │
│  └──────────┘              └──────────┘           └─────────┘ │
│                                                                 │
│  PTT key held = mic unmuted, STT active                        │
│  PTT key released = STT flush, generation cycle                │
│  AI speaking = _stt_gated=True, VAD active for barge-in       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Coupling Points in Current Architecture

1. **`_stt_gated` flag** (live_session.py line 211): When AI is speaking, STT discards all audio. The mic stays live (for VAD barge-in), but no transcription happens. This prevents the system from hearing anything while responding.

2. **`set_muted()` / `_stt_flush_event`** (lines 2656-2673): The PTT key directly controls whether STT accumulates audio. Key release triggers a flush-and-transcribe cycle. This is the fundamental PTT coupling.

3. **`_llm_stage()` blocking on `_stt_out_q`** (line 2395): The LLM stage sits idle waiting for a transcript. It processes exactly one transcript per cycle, then blocks again. There is no concept of "monitoring" an ongoing stream.

4. **`generation_id` for coherence** (line 219): All frames carry a generation ID. When the user interrupts (barge-in), the ID increments and stale frames are discarded. This system works well and should be preserved.

5. **Filler manager races with LLM** (line 2421): The filler system and LLM response run concurrently, with the filler canceled when LLM text arrives. This pattern translates directly to the new architecture.

## Target Architecture (v2.0)

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ╔══════════════════════════════════╗                                 │
│  ║  INPUT STREAM (always running)  ║                                 │
│  ║                                  ║                                 │
│  ║  ┌──────────┐    ┌───────────┐  ║                                 │
│  ║  │  Audio   ├───>│Continuous │  ║                                 │
│  ║  │ Capture  │    │ Whisper   │  ║                                 │
│  ║  │          │    │ STT       │  ║                                 │
│  ║  └──────────┘    └─────┬─────┘  ║                                 │
│  ║                        │        ║                                 │
│  ║               TranscriptSegment ║                                 │
│  ║                        │        ║                                 │
│  ║                        v        ║                                 │
│  ║              ┌──────────────┐   ║                                 │
│  ║              │ Transcript   │   ║                                 │
│  ║              │ Buffer       │   ║  ──> EventBus (all events)      │
│  ║              │ (ring buffer │   ║                                 │
│  ║              │  + context)  │   ║                                 │
│  ║              └──────┬───────┘   ║                                 │
│  ╚═════════════════════╪═══════════╝                                 │
│                        │ (read-only access)                          │
│                        v                                             │
│  ╔═════════════════════════════════════╗                              │
│  ║  MONITOR (Ollama, polling loop)    ║                              │
│  ║                                     ║                              │
│  ║  ┌────────────────────────────┐    ║                              │
│  ║  │ Monitor Loop               │    ║                              │
│  ║  │ - Reads transcript buffer  │    ║                              │
│  ║  │ - Builds context window    │    ║                              │
│  ║  │ - Calls Ollama (3B)       │    ║                              │
│  ║  │ - Decides: RESPOND / WAIT │    ║                              │
│  ║  │ - Routes to backend       │    ║                              │
│  ║  └─────────────┬──────────────┘    ║                              │
│  ╚════════════════╪═══════════════════╝                              │
│                   │                                                   │
│         ResponseDecision                                             │
│         {action, backend,                                            │
│          prompt, context}                                            │
│                   │                                                   │
│                   v                                                   │
│  ╔═════════════════════════════════════╗                              │
│  ║  RESPONSE (Claude CLI or Ollama)   ║                              │
│  ║                                     ║                              │
│  ║  ┌────────────┐  ┌─────────────┐  ║                              │
│  ║  │ Claude CLI │  │ Ollama      │  ║                              │
│  ║  │ (deep/     │  │ (quick/     │  ║                              │
│  ║  │  tools)    │  │  local)     │  ║                              │
│  ║  └─────┬──────┘  └──────┬──────┘  ║                              │
│  ║        └────────┬───────┘         ║                              │
│  ╚═════════════════╪═════════════════╝                              │
│                    │                                                  │
│           text deltas / sentences                                    │
│                    │                                                  │
│                    v                                                  │
│  ╔═════════════════════════════════════╗                              │
│  ║  OUTPUT (unchanged from v1.x)      ║                              │
│  ║                                     ║                              │
│  ║  ┌──────────┐    ┌──────────┐     ║                              │
│  ║  │ Stream   ├───>│ Playback │     ║                              │
│  ║  │ Composer │    │ (PyAudio)│     ║                              │
│  ║  │ (+TTS)   │    │          │     ║                              │
│  ║  └──────────┘    └──────────┘     ║                              │
│  ╚═════════════════════════════════════╝                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────┐              │
│  │ SUPPORT (unchanged from v1.x)                      │              │
│  │ - Filler/Response library (quick clips)            │              │
│  │ - Input classifier (heuristic + semantic)          │              │
│  │ - Learner daemon                                   │              │
│  │ - Event bus (JSONL)                                │              │
│  │ - SSE dashboard                                    │              │
│  │ - Task manager                                     │              │
│  └────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

### New Components

#### 1. Continuous STT (`ContinuousSTT`)

**What it replaces:** The `_stt_stage()` method in `live_session.py` (lines 2179-2383).

**Key change:** Instead of accumulating audio until silence-after-speech and producing one transcript per turn, the continuous STT produces a stream of `TranscriptSegment` objects. Each segment represents a chunk of recognized speech, tagged with timestamps, speaker confidence, and whether it is final or interim.

**Implementation approach:** Use the existing faster-whisper library in a rolling-buffer pattern. Process audio in overlapping windows (e.g., every 2-3 seconds, with 1 second overlap). Compare consecutive transcriptions to extract stable (confirmed) text versus speculative (in-progress) text.

```python
@dataclass
class TranscriptSegment:
    text: str                    # Transcribed text
    timestamp: float             # Wall clock time
    is_final: bool               # True = confirmed, False = interim
    audio_start: float           # Start time in audio stream
    audio_end: float             # End time in audio stream
    confidence: float            # Whisper confidence
    has_speech: bool             # VAD confirmed speech
    metadata: dict               # no_speech_prob, avg_logprob, etc.
```

**Why rolling windows, not streaming Whisper:** Whisper is not a streaming model. Libraries like WhisperLive and whisper_streaming simulate streaming by re-transcribing overlapping audio windows and diffing the results. This is the proven approach. The key optimization: use a smaller Whisper model (`small` or `medium` instead of `large-v3`) to make repeated transcription fast enough for continuous operation.

**Whisper model size trade-off:**

| Model | VRAM | Transcription Speed (3s audio) | WER | Continuous Feasible? |
|-------|------|-------------------------------|-----|---------------------|
| large-v3 | ~3.9GB | 400-800ms | Best | Marginal (GPU contention with Ollama) |
| medium | ~1.5GB | 150-300ms | Good | Yes |
| small | ~0.5GB | 50-150ms | Acceptable | Yes, comfortable |
| distil-large-v3 | ~1.5GB | 100-200ms | Near large-v3 | Yes (best trade-off) |

**Recommendation:** Use `distil-large-v3` for the continuous STT. It has near-large-v3 accuracy at medium-model speed and VRAM. This leaves ~4-5GB free for Ollama. Falls back to `small` if GPU memory is tight.

**Confidence: MEDIUM** -- distil-large-v3 performance claims are from the faster-whisper docs and community benchmarks, not independently verified. The rolling-window approach is proven by WhisperLive and whisper_streaming projects.

**STT gating removal:** The `_stt_gated` flag is removed entirely. The input stream runs continuously regardless of whether the AI is speaking. The monitor decides whether incoming speech is relevant (user talking to the AI vs. background conversation). This is a fundamental shift.

**VAD integration:** Silero VAD (already loaded, line 1306) continues to run on the audio stream. Its role changes from "detect barge-in during playback" to "detect any speech activity at any time." VAD-positive chunks get transcribed; VAD-negative chunks are silently discarded. This prevents wasting GPU on silence/noise.

#### 2. Transcript Buffer (`TranscriptBuffer`)

**What it is:** A shared, thread-safe data structure that accumulates transcript segments and provides a sliding-window view for the monitor.

**Why a separate component:** The monitor needs to read the full recent context (last N seconds or N tokens of conversation). The STT produces segments one at a time. The buffer bridges these two rates, handling deduplication of interim-then-final segments and managing the context window.

```python
class TranscriptBuffer:
    """Thread-safe, append-only transcript accumulator with sliding window access."""

    def __init__(self, max_age_seconds: float = 300, max_tokens: int = 2048):
        self._segments: deque[TranscriptSegment] = deque()
        self._lock = asyncio.Lock()
        self._max_age = max_age_seconds
        self._max_tokens = max_tokens
        self._new_segment_event = asyncio.Event()
        self._token_count = 0

    async def append(self, segment: TranscriptSegment):
        """Add a new segment. Replaces interim segment with same audio range if final."""
        async with self._lock:
            # Replace interim with final if covering same time range
            if segment.is_final:
                self._segments = deque(
                    s for s in self._segments
                    if not (not s.is_final
                            and s.audio_start >= segment.audio_start - 0.5
                            and s.audio_end <= segment.audio_end + 0.5)
                )
            self._segments.append(segment)
            self._evict_old()
            self._new_segment_event.set()

    async def get_context(self, max_tokens: int = 0) -> list[TranscriptSegment]:
        """Get recent segments within token budget."""
        budget = max_tokens or self._max_tokens
        async with self._lock:
            result = []
            tokens_used = 0
            for seg in reversed(self._segments):
                seg_tokens = len(seg.text.split()) * 1.3  # rough token estimate
                if tokens_used + seg_tokens > budget:
                    break
                result.append(seg)
                tokens_used += seg_tokens
            return list(reversed(result))

    async def get_since(self, timestamp: float) -> list[TranscriptSegment]:
        """Get all segments since a timestamp. Used by monitor to get new input."""
        async with self._lock:
            return [s for s in self._segments if s.timestamp > timestamp]

    async def wait_for_new(self, timeout: float = 1.0) -> bool:
        """Block until a new segment arrives or timeout. Returns True if new data."""
        self._new_segment_event.clear()
        try:
            await asyncio.wait_for(self._new_segment_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _evict_old(self):
        """Remove segments older than max_age or exceeding token budget."""
        cutoff = time.time() - self._max_age
        while self._segments and self._segments[0].timestamp < cutoff:
            self._segments.popleft()
```

**Context window management strategy:**

The buffer maintains a 5-minute rolling window of transcript segments. The monitor reads from this buffer with a configurable token budget (default 2048 tokens, which is generous for Llama 3.2 3B's context window of 131K tokens but keeps inference fast).

Why NOT summarization: For a voice assistant monitoring ambient audio, the recent raw text is more valuable than a summary. The monitor needs to see exact phrasing, pauses, and turn boundaries to decide whether to respond. Summarization loses these signals. The 5-minute / 2048-token window is sufficient because conversational context rarely extends beyond a few minutes.

Why NOT vector retrieval: The monitor needs temporal context (what was said recently, in order), not semantic retrieval. Vector stores are for "find relevant past information," not "what just happened in the last 30 seconds."

#### 3. Monitor Loop (`MonitorLoop`)

**What it is:** An asyncio coroutine that polls the transcript buffer, builds a prompt with recent context, calls Ollama (Llama 3.2 3B) to decide whether to respond, and if yes, routes the response to the appropriate backend.

**This is the brain of v2.0.** It replaces the simple "transcript arrives -> send to LLM" flow with a deliberate decision cycle.

```python
@dataclass
class ResponseDecision:
    action: str                  # "respond", "wait", "acknowledge", "ignore"
    backend: str                 # "claude", "ollama", "filler"
    confidence: float            # 0.0 - 1.0
    prompt: str                  # What to send to the response backend
    reasoning: str               # Why (for logging/debugging)
    trigger_segment_id: int      # Which segment triggered the response

class MonitorLoop:
    """Watches transcript buffer, decides when and how to respond."""

    POLL_INTERVAL = 0.5          # Check for new transcripts every 500ms
    SILENCE_THRESHOLD = 2.0      # Seconds of silence before evaluating
    MIN_NEW_TOKENS = 5           # Minimum new words before evaluating
    COOLDOWN_AFTER_RESPONSE = 3.0  # Don't respond again for N seconds

    def __init__(self, transcript_buffer, ollama_client, response_router):
        self._buffer = transcript_buffer
        self._ollama = ollama_client
        self._router = response_router
        self._last_eval_time = 0
        self._last_response_time = 0
        self._last_seen_timestamp = 0
        self._responding = False  # True while AI is generating/speaking

    async def run(self):
        """Main monitor loop. Runs forever."""
        while True:
            # Wait for new transcript data
            has_new = await self._buffer.wait_for_new(timeout=self.POLL_INTERVAL)

            if self._responding:
                continue  # Don't evaluate while AI is speaking

            # Cooldown: don't re-evaluate immediately after responding
            if time.time() - self._last_response_time < self.COOLDOWN_AFTER_RESPONSE:
                continue

            # Get segments since last evaluation
            new_segments = await self._buffer.get_since(self._last_seen_timestamp)
            if not new_segments:
                continue

            # Update timestamp
            self._last_seen_timestamp = new_segments[-1].timestamp

            # Check if there is enough new content to evaluate
            new_text = " ".join(s.text for s in new_segments if s.is_final)
            if len(new_text.split()) < self.MIN_NEW_TOKENS:
                continue

            # Check for silence (user stopped speaking)
            last_segment_age = time.time() - new_segments[-1].timestamp
            if last_segment_age < self.SILENCE_THRESHOLD:
                continue  # User might still be talking

            # Build full context and evaluate
            context = await self._buffer.get_context(max_tokens=1500)
            decision = await self._evaluate(context, new_segments)

            if decision.action == "respond":
                self._responding = True
                self._last_response_time = time.time()
                await self._router.route(decision)
                self._responding = False
            elif decision.action == "acknowledge":
                # Play a quick filler clip without full LLM response
                await self._router.acknowledge(decision)
```

**Monitor decision prompt (sent to Ollama):**

```
You are monitoring a live conversation. Decide whether the AI assistant
should respond to what was just said.

Recent conversation context:
{context_text}

New input (since last check):
{new_text}

The AI assistant's name is "Russel". Respond in JSON:
{
  "action": "respond" | "wait" | "acknowledge" | "ignore",
  "backend": "claude" | "ollama",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Guidelines:
- "respond" when the user is talking to the AI or asking a question
- "wait" when the user is mid-thought and hasn't finished
- "acknowledge" when the user said something that deserves a brief reaction
  but not a full response (e.g., "interesting", sigh, laugh)
- "ignore" when the user is talking to someone else or it is background noise
- Use "claude" backend for complex questions, tasks, code, or anything
  requiring tools
- Use "ollama" backend for simple factual answers, casual chat, or when
  speed matters more than depth
- If the user says the AI's name ("Russel", "hey Russel"), always "respond"
```

**Ollama inference characteristics (Llama 3.2 3B):**

- Model size: ~2GB VRAM at int4 quantization
- Inference latency: 100-300ms for short structured outputs
- Context: 131K tokens (more than enough for our 1500-token window)
- Structured output: Supports JSON mode via `format: "json"` parameter

**Confidence: MEDIUM** -- Llama 3.2 3B's ability to reliably make respond/wait/ignore decisions has not been tested. The 3B model may struggle with nuanced turn-taking detection. If it proves unreliable, a hybrid approach (heuristic pre-filter + Ollama for ambiguous cases) would be the fallback. The existing input classifier's heuristic patterns could serve as the fast path.

**Monitor evaluation frequency:**

The monitor does NOT call Ollama on every transcript segment. It batches: accumulate segments, wait for a silence gap (2 seconds), check minimum word count (5 words), then evaluate. This means:
- ~1-2 Ollama calls per user utterance (not per chunk)
- Each call takes 100-300ms
- Total latency from user-stops-speaking to decision: 2.0-2.3 seconds

This is slower than the current PTT system (which has ~0s decision latency because the key release IS the decision). The 2-second silence threshold is the primary UX parameter to tune. LiveKit's turn detection research suggests semantic endpointing can reduce this, but for v2.0, silence-based endpointing with a 2-second threshold is a reasonable starting point.

#### 4. Response Router (`ResponseRouter`)

**What it is:** Takes a `ResponseDecision` from the monitor and dispatches it to the appropriate backend (Claude CLI or Ollama), manages the response lifecycle, and feeds results to the existing StreamComposer.

```python
class ResponseRouter:
    """Routes response decisions to the appropriate backend."""

    def __init__(self, claude_cli, ollama_client, composer, filler_manager):
        self._claude = claude_cli       # Existing Claude CLI management
        self._ollama = ollama_client    # Ollama chat endpoint
        self._composer = composer        # Existing StreamComposer
        self._filler = filler_manager   # Existing filler/response system

    async def route(self, decision: ResponseDecision):
        """Generate and play a response."""
        if decision.backend == "claude":
            await self._respond_claude(decision)
        elif decision.backend == "ollama":
            await self._respond_ollama(decision)

    async def _respond_claude(self, decision: ResponseDecision):
        """Send to Claude CLI, stream response through composer."""
        # Start filler while waiting for Claude
        filler_cancel = asyncio.Event()
        filler_task = asyncio.create_task(
            self._filler.play_contextual(decision.prompt, filler_cancel)
        )

        # Send to Claude CLI (reuse existing _send_to_cli / _read_cli_response)
        await self._claude.send(decision.prompt)
        # Stream text deltas to composer (same as existing _read_cli_response)
        await self._claude.read_response(self._composer, filler_cancel)

        filler_cancel.set()
        if not filler_task.done():
            filler_task.cancel()

    async def _respond_ollama(self, decision: ResponseDecision):
        """Generate response via Ollama, send to composer."""
        # Ollama responses are fast enough that fillers are usually not needed
        response_text = ""
        sentence_buffer = ""

        async for chunk in self._ollama.chat_stream(decision.prompt):
            response_text += chunk
            sentence_buffer += chunk

            # Same sentence-boundary detection as existing _read_cli_response
            while SENTENCE_END_RE.search(sentence_buffer):
                match = SENTENCE_END_RE.search(sentence_buffer)
                end_pos = match.end()
                sentence = sentence_buffer[:end_pos].strip()
                sentence_buffer = sentence_buffer[end_pos:]
                if sentence:
                    await self._composer.enqueue(
                        AudioSegment(SegmentType.TTS_SENTENCE, data=sentence)
                    )

        # Flush remaining
        if sentence_buffer.strip():
            await self._composer.enqueue(
                AudioSegment(SegmentType.TTS_SENTENCE, data=sentence_buffer.strip())
            )

        await self._composer.enqueue_end_of_turn()

    async def acknowledge(self, decision: ResponseDecision):
        """Play a quick acknowledgment clip without full response."""
        await self._filler.play_contextual(decision.prompt, asyncio.Event())
```

**Backend selection criteria:**

| Criterion | Claude CLI | Ollama |
|-----------|-----------|--------|
| Network available | Required | Not required |
| Complex reasoning | Yes | Limited |
| Tool use (run commands, read files) | Yes (MCP tools) | No |
| Code analysis | Yes | Limited |
| Casual chat | Overkill | Ideal |
| Simple factual Q&A | Works but slow (2-5s) | Fast (0.5-1s) |
| Response latency | 2-5 seconds | 0.5-2 seconds |

**Auto-selection logic in the monitor prompt:** The monitor decides the backend as part of its structured output. If the monitor says "claude" but network is unavailable, the router falls back to Ollama with a degraded prompt. This uses the existing `CircuitBreaker` pattern (line 148) already in the codebase.

#### 5. Barge-In / Name Detection (Modified)

**Current barge-in:** VAD detects sustained speech (6 chunks, ~0.5s) during AI playback, triggers `_trigger_barge_in()` which increments `generation_id`, drains queues, builds annotation.

**New barge-in:** Two modes:
1. **VAD barge-in** (unchanged): Sustained speech during AI output triggers interruption. Same generation_id mechanism, same composer pause/drain.
2. **Name-based interruption** (new): The continuous STT can detect "hey Russel" even during AI playback (since STT is no longer gated). When the transcript buffer receives a segment containing the wake phrase, it triggers an interrupt.

**Name detection approach:** Do NOT use a dedicated wake word engine (openWakeWord, Porcupine). The continuous Whisper STT already transcribes everything. Name detection is a simple string match on the transcript text:

```python
WAKE_PHRASES = {"hey russel", "hey russell", "russel", "russell"}

def _check_wake_phrase(self, segment: TranscriptSegment) -> bool:
    """Check if a transcript segment contains the wake phrase."""
    text_lower = segment.text.lower().strip()
    for phrase in WAKE_PHRASES:
        if phrase in text_lower:
            return True
    return False
```

**Why not openWakeWord:** Adding a wake word engine means another audio processing pipeline running in parallel with Whisper, consuming CPU/GPU. Since Whisper is already transcribing continuously, the transcript IS the wake word detection. This is simpler and more reliable for multi-syllable names (wake word engines excel at short phrases like "hey" but Whisper is better at recognizing actual names).

**Confidence: HIGH** -- This approach is architecturally sound. The only risk is Whisper's latency (the name might be detected 2-3 seconds after being spoken due to the rolling-window transcription). For v2.0, this is acceptable. If sub-second name detection is needed later, openWakeWord can be added as a parallel fast path.

### Unchanged Components

These components require NO architectural changes for v2.0:

| Component | File | Why Unchanged |
|-----------|------|---------------|
| StreamComposer | `stream_composer.py` | Receives `AudioSegment` objects from whoever is generating. Does not care if the source is Claude CLI or Ollama. |
| Playback Stage | `live_session.py` `_playback_stage()` | Consumes `PipelineFrame` from `audio_out_q`. Source-agnostic. |
| Filler/Response Library | `response_library.py`, `input_classifier.py` | Called by the response router before LLM response, same role as today. |
| Event Bus | `event_bus.py` | JSONL event log. New event types can be added (e.g., `monitor_decision`, `transcript_segment`) but the bus infrastructure is unchanged. |
| Learner Daemon | `learner.py` | Tails event bus JSONL, extracts memories. Works regardless of input mode. |
| Task Manager | `task_manager.py` | Spawns/tracks Claude CLI background tasks. Orthogonal to input mode. |
| SSE Dashboard | `live_session.py` `_sse_server_stage()` | Broadcasts bus events. New events are automatically included. |
| Audio Capture | `live_session.py` `_audio_capture_stage()` | PulseAudio recording thread. Unchanged -- it already runs continuously. The only change is that it no longer checks the `muted` flag from PTT. |

### Modified Components

| Component | What Changes | Scope of Change |
|-----------|-------------|-----------------|
| `_stt_stage()` | Replaced by `ContinuousSTT`. No longer gates on `_stt_gated`. Produces `TranscriptSegment` instead of `TRANSCRIPT` frames. | Major rewrite of one method. |
| `_llm_stage()` | Replaced by `MonitorLoop` + `ResponseRouter`. No longer blocks on `_stt_out_q`. | Major rewrite of one method. |
| `_trigger_barge_in()` | Add name-detection path alongside VAD path. | Small addition. |
| `__init__()` | Initialize new components (ContinuousSTT, TranscriptBuffer, MonitorLoop, ResponseRouter, OllamaClient). | Medium -- adding initializers. |
| `run()` | Change stage list: replace `_stt_stage` and `_llm_stage` with new loops. | Medium. |
| Config | Add new settings: `monitor_model`, `response_backend`, `wake_phrase`, `silence_threshold`. | Small. |

## Data Flow: Complete Lifecycle

### Normal Conversation Turn

```
1. User speaks: "What time is it in Tokyo?"
2. Audio Capture -> audio chunks -> ContinuousSTT
3. ContinuousSTT: VAD detects speech, starts accumulating
4. ContinuousSTT: After 2-3s window, Whisper transcribes
   -> TranscriptSegment(text="What time is it in Tokyo?", is_final=True)
5. TranscriptBuffer: Appends segment, signals new data
6. MonitorLoop: Wakes up, sees new segment
7. MonitorLoop: Waits for 2s silence (user stopped talking)
8. MonitorLoop: Builds context, calls Ollama:
   "User said: 'What time is it in Tokyo?' -> respond or wait?"
9. Ollama responds (200ms):
   {"action": "respond", "backend": "ollama", "confidence": 0.9}
10. ResponseRouter: Calls Ollama chat for the actual response
11. Ollama streams: "It's currently 3:42 AM in Tokyo."
12. ResponseRouter: Sends sentence to StreamComposer
13. StreamComposer: Piper TTS -> audio_out_q -> Playback
14. User hears response (~3.5s after finishing speaking)
```

### Barge-In During Response

```
1. AI is speaking (playing audio from composer)
2. ContinuousSTT is running (NOT gated -- this is the key change)
3. User says: "Actually, what about London?"
4. VAD detects speech -> barge-in triggers (same as v1.x)
5. generation_id increments, composer pauses, audio drains
6. Meanwhile, ContinuousSTT transcribes the interruption
7. TranscriptSegment("Actually, what about London?") -> buffer
8. MonitorLoop sees new input, evaluates, decides to respond
9. Response generated for the new question
```

### Name-Based Interruption

```
1. AI is speaking a long response
2. User says: "Hey Russel, stop"
3. ContinuousSTT transcribes: "Hey Russel, stop"
4. TranscriptBuffer receives segment, checks for wake phrase
5. Wake phrase detected -> trigger_barge_in()
6. Same barge-in flow as VAD (generation_id increment, drain, etc.)
7. MonitorLoop sees "stop" in context, decides action "acknowledge"
8. Plays brief acknowledgment clip
```

### Background Conversation (Ignored)

```
1. User is on a phone call, talking to someone else
2. ContinuousSTT transcribes fragments of the conversation
3. TranscriptBuffer accumulates segments
4. MonitorLoop evaluates:
   "User is clearly talking to someone else (no AI name,
    topic is unrelated to previous AI interaction)"
5. Ollama: {"action": "ignore", "reasoning": "not addressed to AI"}
6. No response generated
7. Context window naturally ages out the phone call transcript
```

## GPU Memory Budget

```
RTX 3070: 8192 MB VRAM total

Component                    VRAM (estimated)
─────────────────────────────────────────────
Whisper distil-large-v3      ~1,500 MB
Ollama Llama 3.2 3B (int4)  ~2,000 MB
Silero VAD (ONNX)            ~50 MB
PyTorch/CUDA overhead        ~500 MB
─────────────────────────────────────────────
Total                        ~4,050 MB
Free                         ~4,140 MB (comfortable)

Alternative: Whisper small
─────────────────────────────────────────────
Whisper small                ~500 MB
Ollama Llama 3.2 3B (int4)  ~2,000 MB
Silero VAD (ONNX)            ~50 MB
PyTorch/CUDA overhead        ~500 MB
─────────────────────────────────────────────
Total                        ~3,050 MB
Free                         ~5,140 MB (very comfortable)
```

**Confidence: MEDIUM** -- VRAM estimates are approximate. Actual usage depends on batch sizes, CUDA allocator fragmentation, and whether Ollama keeps the model loaded between inferences. Ollama's GPU memory management (keep-alive, offloading) needs testing.

**Key concern:** Ollama runs as a Docker container (per the shell function in the user's environment). Docker GPU passthrough with `--gpus all` shares the same physical VRAM, but there may be additional overhead from the container runtime. This needs empirical testing.

## Suggested Build Order

### Phase A: Continuous STT (independently testable)

Build `ContinuousSTT` class that reads from `audio_in_q` and produces `TranscriptSegment` objects into a `TranscriptBuffer`. This can be tested standalone -- feed it audio, verify transcript output.

**Deliverables:**
- `continuous_stt.py` (new file)
- `transcript_buffer.py` (new file)
- Unit tests for both

**Dependencies:** Existing audio capture, existing faster-whisper setup
**Can test independently:** Yes -- pipe recorded audio through it
**Risk:** Medium -- Whisper rolling-window approach needs tuning for segment boundaries

### Phase B: Monitor Loop (independently testable)

Build `MonitorLoop` with Ollama integration and `ResponseDecision` structured output. Can be tested with a mock transcript buffer pre-populated with test data.

**Deliverables:**
- `monitor_loop.py` (new file)
- `ollama_client.py` (new file -- thin wrapper around Ollama HTTP API)
- Unit tests with mock buffer

**Dependencies:** TranscriptBuffer (from Phase A), Ollama running locally
**Can test independently:** Yes -- mock the buffer, verify decisions
**Risk:** Medium -- Ollama decision quality needs prompt tuning. Llama 3.2 3B may need structured output guidance.

### Phase C: Response Router + Ollama Backend (partially independent)

Build `ResponseRouter` that dispatches to Claude CLI (reuse existing code) or Ollama (new). The Ollama response path is new; the Claude CLI path wraps existing `_send_to_cli` / `_read_cli_response`.

**Deliverables:**
- `response_router.py` (new file)
- Ollama chat response streaming in `ollama_client.py`
- Integration with existing StreamComposer

**Dependencies:** Phase B (ResponseDecision), existing StreamComposer, existing Claude CLI
**Can test independently:** Partially -- Ollama path yes, Claude path needs live CLI
**Risk:** Low-Medium -- Ollama streaming is straightforward. Claude CLI integration is wrapping existing code.

### Phase D: Pipeline Integration

Wire Phase A/B/C into `live_session.py`. Replace `_stt_stage()` and `_llm_stage()` with the new components. Remove `_stt_gated` flag. Add name detection. Update `run()` method to launch new coroutines.

**Deliverables:**
- Modified `live_session.py`
- Modified `pipeline_frames.py` (new frame types if needed)
- End-to-end integration tests

**Dependencies:** Phases A, B, C all complete
**Risk:** High -- this is the big integration. Timing, concurrency, and edge cases (what happens when Ollama is slow, when Whisper produces garbage, when the user and AI talk simultaneously).

### Phase E: Barge-In + Name Detection

Add name-based interruption. Modify `_trigger_barge_in` to accept both VAD and name-based triggers. Test interruption scenarios.

**Deliverables:**
- Name detection in ContinuousSTT or TranscriptBuffer
- Modified barge-in logic
- Integration tests for interruption scenarios

**Dependencies:** Phase D (pipeline must be working)
**Risk:** Low -- name detection is string matching. Barge-in mechanism exists and works.

### Phase F: Tuning + Polish

Tune silence thresholds, Ollama prompt, backend selection heuristics. Add fallback behaviors (Ollama unavailable, GPU OOM, bad decisions). Harden edge cases.

**Dependencies:** Phase D and E
**Risk:** Medium -- tuning is iterative and requires real-world testing

## Anti-Patterns to Avoid

### Anti-Pattern 1: Running STT and Monitor on Every Audio Chunk

**What:** Transcribing every 85ms audio chunk and evaluating every transcript.
**Why bad:** Whisper inference on short audio is inaccurate and wasteful. Ollama at 200ms per call would be called 12x/second.
**Instead:** Use VAD to gate STT (only transcribe when speech detected). Use silence detection to gate monitor (only evaluate when user stops talking).

### Anti-Pattern 2: Sharing Whisper Model Between Continuous STT and Batch Operations

**What:** Using the same `WhisperModel` instance for both continuous transcription and on-demand batch transcription (e.g., for dictation mode).
**Why bad:** faster-whisper is not thread-safe for concurrent inference. Concurrent calls cause crashes or garbage output.
**Instead:** The continuous STT owns its own model instance. If batch transcription is needed (dictation mode), either stop continuous STT or use a separate model instance.

### Anti-Pattern 3: Making the Monitor Synchronous with the Input Stream

**What:** Requiring the monitor to evaluate every transcript before the next one is produced.
**Why bad:** If Ollama is slow (300ms+), transcript segments pile up. The monitor falls behind, decisions are stale.
**Instead:** The monitor runs its own async loop at its own pace. It reads from the buffer, which is always up-to-date. If the monitor is slow, it simply evaluates a larger batch of segments on the next cycle.

### Anti-Pattern 4: Trying to Make Ollama Do Tool Calls

**What:** Adding MCP tool support to Ollama responses.
**Why bad:** Ollama with Llama 3.2 3B has limited tool-calling reliability. The existing Claude CLI tool pipeline is battle-tested. Duplicating it for Ollama doubles complexity for minimal benefit.
**Instead:** If the monitor decides tools are needed, it routes to Claude CLI. Ollama is for quick, tool-free responses only.

### Anti-Pattern 5: Removing PTT Mode Entirely

**What:** Deleting all PTT code and making always-on the only mode.
**Why bad:** Users may want PTT in noisy environments, shared spaces, or when privacy matters. Always-on is a new mode, not a replacement.
**Instead:** Add always-on as a new `ai_mode` option (alongside existing "claude", "interview", "conversation"). PTT mode remains available as-is. Config drives which mode is active.

### Anti-Pattern 6: Blocking on Ollama Before Playing Filler

**What:** Waiting for the monitor's Ollama decision before playing any audio feedback.
**Why bad:** Adds 200-300ms of silence on top of the 2-second silence threshold. Total time from user-stops-speaking to any audio: ~2.5 seconds.
**Instead:** The existing filler/response library should still fire on heuristic classification (fast path, <10ms). The monitor's decision gates the FULL response, not the filler. If the monitor decides "ignore," the filler may have already played -- that is acceptable (a brief "hmm" acknowledging the user is better than 2.5 seconds of silence followed by nothing).

## Configuration Additions

```json
{
  "ai_mode": "always_on",           // New mode alongside "claude", "interview", etc.
  "monitor_model": "llama3.2:3b",   // Ollama model for monitoring
  "response_model": "llama3.2:3b",  // Ollama model for quick responses
  "wake_phrase": "russel",           // Name for interruption detection
  "silence_threshold": 2.0,          // Seconds of silence before evaluating
  "monitor_cooldown": 3.0,           // Seconds between monitor evaluations
  "whisper_model": "distil-large-v3", // STT model (continuous mode)
  "response_backend": "auto",        // "auto", "claude", "ollama"
  "always_on_stt": true              // Enable continuous STT
}
```

## Event Bus Additions

New event types for v2.0 observability:

| Event Type | Payload | Purpose |
|------------|---------|---------|
| `transcript_segment` | `{text, is_final, confidence, timestamp}` | Track continuous STT output |
| `monitor_decision` | `{action, backend, confidence, reasoning}` | Track monitor decisions |
| `response_routed` | `{backend, prompt_preview}` | Track which backend was selected |
| `wake_phrase_detected` | `{phrase, timestamp}` | Track name-based interruptions |
| `ollama_inference` | `{model, latency_ms, tokens}` | Track Ollama performance |

## Open Questions

1. **Whisper model choice needs empirical testing.** distil-large-v3 is recommended but VRAM usage with Ollama simultaneously loaded needs measurement. Might need to fall back to `small`.

2. **Ollama decision quality at 3B.** Llama 3.2 3B may struggle with nuanced "respond vs ignore" decisions, especially distinguishing "user talking to AI" from "user talking to someone else." If this proves unreliable, a hybrid approach (heuristic fast path + Ollama only for ambiguous cases) should be tried.

3. **Latency budget.** The 2-second silence threshold + 200ms Ollama decision = 2.2 seconds minimum from user-stops-speaking to response start. This is slower than PTT (~0.5s). LiveKit's research suggests transformer-based turn detection can reduce silence thresholds to 0.5-1.0 seconds, but that requires a specialized model. For v2.0, the 2-second threshold is the starting point.

4. **GPU memory under sustained load.** Continuous Whisper + Ollama keep-alive both want GPU memory allocated permanently. Need to test whether CUDA allocator handles this gracefully or whether fragmentation causes OOM over long sessions.

5. **Ollama Docker vs native.** Ollama is currently a Docker alias. Docker GPU passthrough may add latency. Worth testing native Ollama installation for comparison.

## Sources

- **Codebase analysis** (HIGH confidence): `live_session.py` (2900+ lines, read in full), `stream_composer.py`, `pipeline_frames.py`, `event_bus.py`, `input_classifier.py`, `learner.py`, `push-to-talk.py` (config), all read in full.
- **WhisperLive / whisper_streaming** (MEDIUM confidence): [GitHub: collabora/WhisperLive](https://github.com/collabora/WhisperLive), [GitHub: ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) -- rolling-window approach for continuous Whisper transcription.
- **LiveKit turn detection** (MEDIUM confidence): [LiveKit blog: using a transformer for end-of-turn detection](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection/) -- semantic endpointing with transformer model, <500MB RAM, runs on CPU. Potential future improvement for v2.1.
- **Ollama structured outputs** (MEDIUM confidence): [Ollama docs: structured outputs](https://docs.ollama.com/capabilities/structured-outputs) -- JSON mode with Pydantic schema, confirmed for Llama 3.2.
- **AssemblyAI voice agent stack** (LOW confidence): [AssemblyAI blog: voice AI stack](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents) -- streaming architecture patterns, immutable transcription concept.
- **openWakeWord** (MEDIUM confidence): [GitHub: dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) -- evaluated and rejected for v2.0 in favor of transcript-based name detection.
- **GPU specs** (HIGH confidence): `nvidia-smi` on the development machine -- RTX 3070, 8192 MB VRAM.
- **Ollama environment** (HIGH confidence): Ollama runs as Docker container via shell alias in user's `.zshrc`.
