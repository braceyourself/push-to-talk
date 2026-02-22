# Feature Landscape: Streaming STT + Local Decision Model for Always-On Voice Assistant

**Domain:** Desktop always-on voice assistant with Deepgram streaming STT and local decision model
**Researched:** 2026-02-22
**Confidence:** HIGH (Deepgram API docs verified, latency numbers from official sources, billing model confirmed by Deepgram staff on GitHub)

**Architectural Context:** This is a REFRESH of the v2.0 features research. The original plan used local Whisper distil-large-v3 for batch transcription. Testing revealed batch-transcribe latency (850ms silence gap + 500ms-2s Whisper inference) was too slow for natural conversational awareness. The pivot to Deepgram streaming STT changes the feature landscape significantly: streaming interim results enable new capabilities (real-time word tracking, progressive transcript building) while introducing new concerns (WebSocket lifecycle, cost management, network dependency).

## Existing Foundation

The system already has:
- 5-stage asyncio pipeline: Audio Capture -> STT -> LLM (Claude CLI) -> TTS (OpenAI/Piper) -> Playback
- Deepgram SDK already in requirements.txt (`deepgram-sdk>=3.0`), API key wired through LiveSession constructor
- Tests already validate the `is_final`/`speech_final` accumulation pattern (test_live_session.py lines 132-178)
- Heuristic + semantic input classification (6 categories, <10ms via model2vec)
- Categorized quick response library with situation-matched audio clips
- StreamComposer for unified audio queue with pre-buffering and barge-in
- Barge-in interruption with Silero VAD detection
- TranscriptBuffer ring buffer (200 segments, 5min TTL)
- Silero VAD ONNX model already loaded for barge-in
- Event bus (JSONL) for cross-component communication
- CircuitBreaker class for service fallback (already exists for STT/Deepgram)

**What this pivot changes vs. the original v2.0 plan:**

| Aspect | Original Plan (Whisper) | New Plan (Deepgram Streaming) |
|--------|------------------------|-------------------------------|
| STT latency | 850ms silence + 500ms-2s inference = 1.3-2.8s | ~150ms streaming interim, <300ms finals |
| GPU VRAM | ~3GB Whisper + ~2.5GB Ollama = tight on 8GB | 0 GPU for STT, full budget for decision model |
| Cost | Free (local) | ~$0.0077/min audio sent (~$0.46/hr continuous) |
| Transcript granularity | Batch: complete utterance after silence | Streaming: word-by-word with timestamps |
| Network dependency | None for STT | Required for STT (fallback to Whisper if down) |
| Hallucination risk | High (Whisper on ambient noise) | Low (Deepgram trained on real-world audio) |

---

## Table Stakes

Features the system MUST have for Deepgram streaming STT + decision model to work. Missing any one of these makes the system broken or unusable.

---

### TS-1: Deepgram WebSocket Streaming Connection

**What:** Establish and maintain a persistent WebSocket connection to Deepgram's streaming API (`wss://api.deepgram.com/v1/listen`) using the official Python SDK (`deepgram-sdk`). Audio flows from the mic through the WebSocket as binary frames; transcripts flow back as JSON events.

**Why Required:** This replaces the entire local Whisper STT pipeline. Without a working Deepgram connection, there is no transcription at all. The SDK is already a dependency (`deepgram-sdk>=3.0` in requirements.txt), the API key is already wired through the LiveSession constructor, and the test suite already validates Deepgram response patterns.

**Key configuration parameters (verified from Deepgram API docs):**
```python
{
    "model": "nova-3",           # Best accuracy, $0.0077/min PAYG
    "language": "en-US",
    "encoding": "linear16",      # Match existing 24kHz 16-bit mono PCM
    "sample_rate": 24000,        # Match SAMPLE_RATE constant
    "channels": 1,               # Mono mic input
    "interim_results": True,     # Get word-by-word updates (~150ms)
    "endpointing": 300,          # 300ms silence = speech_final (tunable)
    "utterance_end_ms": 1000,    # 1s silence = UtteranceEnd event
    "vad_events": True,          # SpeechStarted events for latency tracking
    "smart_format": True,        # Auto-format numbers, currency, etc.
    "punctuate": True,           # Auto-punctuation
}
```

**Complexity:** MEDIUM -- The SDK handles WebSocket lifecycle, but integrating it into the existing asyncio pipeline requires replacing the Whisper-based `_stt_stage` and adapting the audio capture flow. The existing tests already validate the callback pattern.

**Dependencies:** Deepgram API key (already wired), `deepgram-sdk` (already in requirements.txt)

**Confidence:** HIGH (SDK API verified from official docs, tests already exist for the transcript accumulation pattern)

---

### TS-2: Interim + Final Transcript Accumulation

**What:** Manage the three-tier Deepgram transcript lifecycle:
1. **Interim results** (`is_final: false`): Preliminary word-by-word updates, may change. Display for feedback, do NOT use for decisions.
2. **Final results** (`is_final: true`): Accurate transcript for a segment. Accumulate in a buffer.
3. **Speech final** (`speech_final: true`): Silence detected (endpointing). Concatenate all accumulated finals = complete utterance. Feed to decision engine.

Plus the **UtteranceEnd** event: fires after `utterance_end_ms` of silence after the last finalized word. Signals "user is done talking, safe to process."

**Why Required:** This is the core data flow. Getting it wrong means either: (a) acting on interim results that get corrected (wrong decisions), (b) waiting too long for speech_final (laggy responses), or (c) missing multi-segment utterances (truncated context). The existing test suite (test_live_session.py lines 132-178) already validates the correct accumulation pattern:

```python
# From existing test: accumulate is_final segments, flush on speech_final
def on_message_logic(transcript, is_final, speech_final):
    if not transcript:
        return
    if is_final:
        accumulated.append(transcript)
        if speech_final:
            full_text = " ".join(accumulated).strip()
            accumulated.clear()
```

**Implementation detail -- endpointing value tradeoff:**
- `endpointing=10` (default): Near-instant speech_final. Good for chatbots. Too aggressive for natural speech -- 10ms pauses between words would split mid-sentence.
- `endpointing=300`: 300ms silence = speech_final. Good balance for conversational. Natural pauses (commas, thinking) are ~200ms, so 300ms catches sentence boundaries without splitting.
- `endpointing=500-1000`: Very conservative. Waits for clear end of thought. Adds latency but reduces false splits.

**Recommended starting value:** `endpointing=300` with `utterance_end_ms=1000`. The 300ms catches natural sentence boundaries; the 1000ms UtteranceEnd signals "user is truly done, not just pausing."

**Complexity:** LOW-MEDIUM -- The pattern is already tested. Implementation is wiring the Deepgram SDK callbacks to the existing pipeline queues.

**Dependencies:** TS-1 (Deepgram connection)

**Confidence:** HIGH (pattern validated in existing tests, Deepgram docs are clear on the lifecycle)

---

### TS-3: VAD-Gated Audio Streaming (Cost Optimization)

**What:** Use the existing Silero VAD (already loaded for barge-in) to gate what audio reaches Deepgram. Only stream audio when speech is detected. During silence, send KeepAlive messages to maintain the WebSocket connection without billing.

**Why Required:** Deepgram bills per second of audio sent ($0.0077/min). In an always-on scenario with ~8 hours/day of mic-open time, naive continuous streaming would cost:
- All audio streamed: 480 min x $0.0077 = $3.70/day
- VAD-gated (assume 20% speech): 96 min x $0.0077 = $0.74/day

That is an 80% cost reduction. More importantly, not streaming silence prevents Deepgram from producing empty transcripts and avoids wasting bandwidth.

**Billing model (verified by Deepgram staff on GitHub Discussion #1423):**
- Billing is per second of audio SENT to Deepgram
- An open WebSocket NOT transmitting audio does NOT cost anything
- KeepAlive messages do NOT incur charges
- KeepAlive must be sent every 3-5 seconds to prevent 10-second timeout (NET-0001 error)

**Implementation pattern:**
```
Audio chunk (85ms) -> Silero VAD (CPU, <1ms)
  |
  +-- Speech detected (prob > 0.5) -> Stream to Deepgram as binary frame
  |
  +-- Silence detected -> Do NOT stream
                          Send KeepAlive every 3-5s instead
```

**Why Silero VAD, not Deepgram's built-in VAD:**
Deepgram has `vad_events=true` which fires SpeechStarted events, but this runs SERVER-SIDE after audio is already sent and billed. Local Silero VAD runs BEFORE sending audio, preventing unnecessary data from reaching Deepgram at all. The two are complementary: Silero gates locally for cost, Deepgram's VAD provides server-side speech boundary detection for accurate endpointing.

**Complexity:** MEDIUM -- The Silero VAD infrastructure exists but needs adaptation. Currently VAD only runs during AI playback for barge-in detection (the `_stt_gated` path in `_stt_stage`). For always-on, VAD must run on every audio chunk. The key change: VAD output controls whether to send audio to Deepgram, not whether to suppress STT.

**Edge case -- speech onset latency:** When VAD transitions from silence to speech, the first ~85ms chunk that triggered VAD has already been consumed. If this chunk is NOT sent to Deepgram, the first word may be clipped. Solution: buffer the previous 1-2 silence chunks and send them when speech starts (a "lookback buffer"). This is a well-known pattern in VAD-gated streaming.

**Dependencies:** Silero VAD (already loaded), TS-1 (Deepgram connection to send KeepAlive to)

**Confidence:** HIGH (VAD already works, billing model confirmed, pattern well-established)

---

### TS-4: Transcript Buffer Integration

**What:** Feed Deepgram streaming transcripts into the existing TranscriptBuffer ring buffer. On each `speech_final`, create a TranscriptSegment and append it. The buffer provides the context window that the decision model reads.

**Why Required:** The decision model needs conversation history to make respond/don't-respond decisions. "Yeah that's true" in isolation is meaningless; with 3 minutes of preceding context, the model can tell if it's a response to a question it should follow up on. The TranscriptBuffer already exists (transcript_buffer.py) with proper threading, time-based eviction, and LLM-friendly formatting via `get_context(max_tokens)`.

**Key change from the Whisper-based plan:** With Whisper batch, each segment was a complete utterance (1-10 seconds of speech processed at once). With Deepgram streaming, segments arrive faster and in smaller pieces. The buffer will receive more frequent, shorter segments. This is actually better for the decision model -- it gets context updates in near-real-time rather than in delayed batches.

**What goes into the buffer:**
- `speech_final` utterances as `source="user"` segments
- AI responses as `source="ai"` segments (already tracked in existing `_spoken_sentences`)
- Optionally: interim results as temporary entries that get replaced by finals (for real-time monitoring display only, not for decision model input)

**Complexity:** LOW -- TranscriptBuffer.append() already works. The change is in the source of segments (Deepgram callbacks instead of Whisper transcription results).

**Dependencies:** TS-2 (needs finalized transcripts to buffer)

**Confidence:** HIGH (TranscriptBuffer is tested and proven)

---

### TS-5: Decision Engine with Transcript Stream

**What:** After each complete utterance (speech_final or UtteranceEnd), the local decision model (Ollama + Llama 3.2 3B) evaluates the transcript buffer and decides: should Russel respond? If yes, what kind of response?

**Why Required:** This is the intelligence layer. Without it, the system is just a transcription service. The decision engine is what makes the assistant "always-on" rather than "always-recording." It is the single most important feature.

**Decision input:** The `get_context()` output from TranscriptBuffer, plus the current utterance that triggered evaluation.

**Decision output (structured JSON via Ollama `format` parameter):**
```json
{
    "should_respond": true,
    "confidence": 0.85,
    "response_type": "substantive",
    "complexity": "quick",
    "tone": "casual",
    "reason": "User asked a factual question directed at the room"
}
```

**Trigger timing -- when to evaluate:**
- On every `speech_final` event (natural sentence boundary)
- On `UtteranceEnd` event (user done talking for >1s)
- NOT on every interim result (too frequent, wastes Ollama cycles)

**Latency budget for decision:**
- Deepgram streaming final: ~150-300ms from end of speech
- Ollama classification (3B, `num_predict: 100`, `temperature: 0`): ~200ms
- Total decision latency: ~350-500ms from end of speech to "should I respond?" answer
- This is within the 500-800ms natural conversation response window

**Complexity:** HIGH -- This is the hardest feature. The prompt engineering must balance sensitivity (not missing direct questions) with restraint (not responding to everything). The latency budget is tight.

**Dependencies:** TS-4 (transcript buffer as input), Ollama + Llama 3.2 3B (already planned in STACK.md)

**Confidence:** MEDIUM (concept validated in research, but Llama 3.2 3B reliability for this specific task needs benchmarking)

---

### TS-6: Name-Based Activation via Transcript Matching

**What:** Monitor streaming transcripts for the assistant's name ("Russel", "Russell", "hey Russel", "Russ"). Name mentions bypass the decision engine's confidence threshold -- if you say the name, the AI always responds. Also serves as name-based barge-in (say "Russel" to stop playback, replacing PTT-based barge-in).

**Why Required:** Users need a deterministic activation path. The decision model is probabilistic (might or might not respond). Name activation is guaranteed. Every always-on assistant has this: "Alexa", "Hey Siri", "OK Google."

**Why NOT a separate wake word model:**
- Deepgram streaming STT is already running continuously and produces transcripts
- `if "russel" in transcript.lower()` is instant and free
- A separate wake word model (Picovoice, openWakeWord) would require a separate audio pipeline, separate model loading, and adds complexity for no benefit when STT is already transcribing everything
- The only downside: name detection has STT latency (~150-300ms). A dedicated wake word model runs in ~80ms. For this use case, 300ms is acceptable.

**New capability from streaming:** With Deepgram interim results, name detection can happen on INTERIM transcripts, not just finals. This means "Hey Russel" can be detected as soon as those words appear in the stream (~150ms), before endpointing fires. This is faster than waiting for speech_final.

**Implementation:**
```python
# On EVERY interim/final result:
def check_name(transcript: str) -> bool:
    lower = transcript.lower().strip()
    triggers = ["hey russel", "hey russell", "russel", "russell", "hey russ"]
    return any(t in lower for t in triggers)

# During AI playback: name in interim -> immediate barge-in
# During silence: name in final -> force decision engine to respond
```

**Complexity:** LOW -- String matching on existing transcript stream. The barge-in infrastructure already exists.

**Dependencies:** TS-1 (Deepgram producing transcripts)

**Confidence:** HIGH (trivial implementation, well-understood pattern)

---

### TS-7: WebSocket Lifecycle Management

**What:** Handle the full Deepgram WebSocket lifecycle: connection, KeepAlive during silence, reconnection on errors, graceful shutdown, and timestamp management across reconnections.

**Why Required:** An always-on system runs for 8+ hours. WebSocket connections drop. Networks hiccup. Deepgram server restarts happen. Without proper lifecycle management, the system silently stops transcribing and the user doesn't know.

**Key behaviors:**
1. **KeepAlive during VAD silence:** Send `{"type": "KeepAlive"}` as text frame every 3-5 seconds when no audio is being streamed. The SDK may handle this automatically (needs verification), but defensive sending is recommended.
2. **Auto-reconnection:** On connection drop (NET-0001 timeout, network error), immediately reconnect. Buffer audio during reconnection gap.
3. **Timestamp offset management:** Each new connection resets timestamps to 00:00:00. Maintain a running offset: `real_timestamp = deepgram_timestamp + offset`. Update offset on reconnection.
4. **Audio must flow within 10 seconds of connection:** After opening the WebSocket, audio (or KeepAlive) must arrive within 10 seconds or Deepgram closes the connection.
5. **Graceful shutdown:** Send `{"type": "CloseStream"}` before closing the WebSocket to flush any pending transcripts.
6. **Finalize message:** Send `{"type": "Finalize"}` to flush the current audio buffer without closing the connection. Useful before transitioning to KeepAlive mode.

**Error recovery strategy:**
- Connection error -> Log, increment CircuitBreaker (already exists: `_stt_breaker`), attempt reconnection
- Circuit breaker trips (3 consecutive failures) -> Fall back to local Whisper STT (`_stt_whisper_fallback` already exists)
- Recover after 60s -> Attempt Deepgram reconnection

**Complexity:** MEDIUM -- The patterns are well-documented, but edge cases (audio buffering during reconnection, timestamp alignment) need careful handling.

**Dependencies:** TS-1 (connection to manage)

**Confidence:** HIGH (Deepgram docs on recovery are thorough, CircuitBreaker already exists)

---

### TS-8: Echo Suppression During AI Playback

**What:** When the AI is speaking via TTS, prevent the AI's own voice from being transcribed. This is the "audio feedback loop" problem -- the single most reported problem in always-on voice assistants.

**Why Required:** Without echo suppression, the AI speaks -> Deepgram transcribes the AI's speech -> decision engine sees it as user input -> AI responds to itself -> infinite loop. This WILL happen if not addressed.

**Implementation layers (defense in depth):**

1. **PipeWire AEC (primary):** Already planned from the original v2.0 research. Route capture through `echo-cancel-capture` virtual source. Subtracts playback signal from mic input. The ContinuousSTT module already supports an `aec_device_name` parameter.

2. **Transcript gating during playback (secondary):** During AI playback, tag Deepgram transcripts as `during_ai_speech=True`. The decision engine ignores these UNLESS they contain the AI's name (barge-in). This is a streaming-aware evolution of the existing `_stt_gated` binary gate.

3. **Transcript fingerprinting (tertiary):** Compare incoming transcripts against the last N sentences the AI spoke (already tracked in `_spoken_sentences`). Fuzzy match (Levenshtein ratio > 0.7) = echo. Reject before reaching the decision engine.

**Key difference from Whisper-based approach:** With Whisper, `_stt_gated` simply discarded all audio during playback. With Deepgram streaming, audio must CONTINUE flowing (for KeepAlive and to avoid reconnection issues), but transcripts from that audio should be filtered. The WebSocket stays open; the transcript consumer applies the gating logic.

**Complexity:** MEDIUM-HIGH -- PipeWire AEC setup is straightforward but system-level. Transcript gating during streaming requires careful state management (what if speech_final crosses the playback boundary?).

**Dependencies:** TS-1 (audio flowing to Deepgram), TS-2 (transcript accumulation), existing barge-in infrastructure

**Confidence:** MEDIUM (PipeWire AEC is well-documented, but effectiveness with desktop speakers in a real room needs testing)

---

### TS-9: Configurable Response Backend (Claude CLI / Ollama)

**What:** After the decision engine says "respond," route to the appropriate LLM backend. Claude CLI for complex/tool-using queries. Ollama for quick conversational responses. Selection is automatic based on the decision engine's `complexity` field and network state.

**Why Required:** The decision engine produces a decision, not a response. Something needs to generate the actual words. Claude CLI is powerful but slow (2-5s TTFT). Ollama is fast (~200ms TTFT) but less capable. The system needs both.

**Routing logic:**
```
decision.complexity == "quick" AND Ollama available -> Ollama
decision.complexity == "deep" OR tool request -> Claude CLI
Network down -> Ollama (forced, regardless of complexity)
Ollama down -> Claude CLI with heuristic classifier
```

**Latency impact on streaming vs batch:**
With Deepgram streaming, the decision happens ~300-500ms after speech ends. If routed to Ollama, first audio can arrive ~700ms after speech ends (300ms decision + 200ms Ollama TTFT + 200ms TTS). This is within the natural 500-800ms response window. With Claude CLI, it's 2-5s -- still acceptable for complex requests but not for quick quips.

**Complexity:** MEDIUM -- Two separate response pipelines, but the existing StreamComposer handles TTS audio from either source identically.

**Dependencies:** TS-5 (decision engine provides routing signal), Ollama (STACK.md), existing Claude CLI pipeline

**Confidence:** HIGH (Ollama API verified, Claude CLI pipeline battle-tested)

---

### TS-10: Resource Management for Continuous Operation

**What:** The system must run 8+ hours without memory leaks, connection failures, or degraded performance. Specific concerns for the streaming architecture:

1. **WebSocket connection health monitoring:** Detect silent failures where the connection appears open but stops delivering transcripts
2. **Transcript buffer bounding:** Already implemented (ring buffer, 200 segments, 5min TTL)
3. **Deepgram SDK memory:** The SDK maintains internal buffers for audio and response data. Monitor for growth over time.
4. **Ollama model keep-alive:** Ensure the decision model stays loaded in GPU VRAM via `keep_alive=-1`
5. **Audio buffer overflow:** During network hiccups, audio buffers in the capture stage can grow unbounded. Cap at 10s of audio and drop oldest frames.

**GPU VRAM budget (post-pivot):**

| Model | VRAM | Notes |
|-------|------|-------|
| Whisper (removed) | 0 GB | No longer running -- freed by Deepgram pivot |
| Ollama Llama 3.2 3B | ~2-2.5 GB | Decision model + quick responses |
| Silero VAD | ~50 MB | ONNX on CPU, negligible |
| **Total GPU** | **~2-2.5 GB** | Massively better than the original ~5-6GB |

The freed VRAM opens options: run a larger decision model (Llama 3.2 8B at ~4.5GB), or keep headroom for desktop GPU usage.

**Complexity:** MEDIUM -- Mostly engineering discipline. The streaming architecture adds network-related failure modes not present in the local-only plan.

**Dependencies:** All other features (cross-cutting concern)

**Confidence:** HIGH (known patterns, much better VRAM situation than original plan)

---

## Differentiators

Features that make this system better than existing voice assistants. Not required for basic operation, but these are what make the experience feel natural and useful.

---

### D-1: Real-Time Transcript Display from Interim Results

**What:** Display interim results (word-by-word) in the overlay/dashboard as the user speaks. Provides immediate visual feedback that the system is hearing and transcribing. Replace interim text with final text when `is_final` arrives.

**Value Proposition:** No consumer voice assistant shows you what it's hearing in real-time. This creates a "glass box" experience where the user can see the system is working, correct if it mishears, and feel confident the assistant is paying attention. It's the difference between talking to a person who nods vs. one who stares blankly.

**Implementation approach:**
- On each interim result: Update the overlay with current text (with a "..." indicator)
- On `is_final`: Replace interim with final text
- On `speech_final`: Move accumulated text to the conversation log, clear the interim display

**Complexity:** LOW -- The overlay already exists. This is a new data source for it.

**Dependencies:** TS-2 (interim results from Deepgram)

**Confidence:** HIGH (trivial UI update)

---

### D-2: Word-Level Timestamp Analysis for Response Timing

**What:** Use Deepgram's word-level timestamps to make smarter decisions about when to respond. Analyze speaking pace, pause patterns, and sentence rhythm to detect natural turn-taking points.

**Value Proposition:** Research shows the average gap between speakers in natural conversation is ~200ms. Responses after 300-400ms feel "awkward." With word-level timestamps, the system can calculate:
- Speaking rate (words/second) -- is the user speaking slowly (thinking) or fast (excited)?
- Pause duration between words -- is this a within-sentence pause or a between-sentence pause?
- Utterance duration -- short utterance after a question = likely an answer; long utterance = ongoing thought

This enables the decision engine to time responses more naturally than relying solely on a fixed endpointing threshold.

**Complexity:** MEDIUM -- Requires analyzing the timestamp data from Deepgram word arrays. The analytics are straightforward but tuning the thresholds needs experimentation.

**Dependencies:** TS-2 (word-level timestamps from Deepgram finals)

**Confidence:** MEDIUM (data is available, but whether a 3B local model can meaningfully use this information is unverified)

---

### D-3: Speaker Diarization via Deepgram

**What:** Enable Deepgram's `diarize=true` parameter to get per-word speaker IDs in transcripts. Use speaker labels to distinguish "user talking to Russel" from "user talking to someone else" or "TV audio."

**Value Proposition:** The biggest complaint about always-on assistants is responding to TV/radio/other people. Deepgram's diarization assigns speaker IDs per word, trained on 100,000+ voices across real-world conversational data. This is dramatically better than the energy-based heuristics planned in the original v2.0 research.

**Tradeoff:** Diarization adds $0.0020/min to the streaming cost (from $0.0077 to $0.0097/min). For 8 hours of VAD-gated audio (~96 min actual speech), this is ~$0.19/day additional cost. Worth it for significantly reduced false activations.

**Complexity:** LOW -- It's a single parameter in the Deepgram connection config. The transcript response includes `speaker` field per word. The harder part is teaching the decision model to use speaker labels effectively.

**Dependencies:** TS-1 (Deepgram connection), TS-5 (decision engine needs to understand speaker context)

**Confidence:** MEDIUM (Deepgram diarization works well in pre-recorded audio; real-time streaming diarization quality needs testing. Historical GitHub issue #108 reported diarize always returning speaker 0 in streaming -- may be resolved in Nova-3 but needs verification.)

---

### D-4: Proactive Conversation Participation

**What:** Russel joins conversations even when not addressed. If someone discusses a code problem and Russel knows the answer, he speaks up. If someone makes a factual error, Russel corrects it.

**Value Proposition:** This is the core differentiator from every existing voice assistant. Alexa/Siri/Google only respond when addressed. Russel is a participant.

**How streaming STT improves this:** With batch Whisper, the system only knew what was said after silence detection + transcription (1.5-3s delay). With Deepgram streaming, the system sees words as they are spoken (~150ms). The decision model can begin thinking about whether to respond while the user is still talking, enabling faster proactive contributions that feel natural rather than delayed.

**Complexity:** HIGH -- The prompt engineering for proactive participation must encode the "eight scoring heuristics" from the Inner Thoughts framework (CHI 2025). Too aggressive = annoying. Too conservative = useless.

**Dependencies:** TS-5 (decision engine), TS-9 (fast Ollama backend for quick contributions), D-6 (attention signals to avoid startling)

**Confidence:** MEDIUM (concept validated in research, implementation quality depends on prompt engineering)

---

### D-5: Attention Signals Before Proactive Responses

**What:** Before Russel proactively interjects, play a brief attention signal: a subtle sound, or a short verbal cue like "Hey," "Actually," or "Oh." This gives the user a moment to register that Russel is about to speak.

**Value Proposition:** Research (CHI 2024 "Better to Ask Than Assume") shows users strongly prefer an explicit signal before unsolicited VA contributions. Without it, the AI suddenly talking is startling and annoying.

**Implementation:** Add an "attention_signal" category to the response library. Before proactive responses, enqueue an attention clip into the StreamComposer before the main response. The StreamComposer's cadence system (post_clip_pause) handles the natural gap between signal and response.

**Complexity:** LOW -- Uses existing StreamComposer and response library infrastructure.

**Dependencies:** D-4 (proactive participation triggers the signal)

**Confidence:** HIGH (simple implementation, well-researched pattern)

---

### D-6: Interruptibility Detection

**What:** Detect when the user is busy/focused and suppress proactive responses. Signals: long silence (deep work), explicit "quiet mode" command, sustained ambient non-speech (typing).

**Value Proposition:** The biggest risk with proactive AI is annoying the user during focus time.

**How streaming STT helps:** With continuous transcript monitoring, the system can detect silence duration in real-time. "No speech_final events for 10 minutes" = user is in deep work mode. No need for a separate timer or OS integration -- the transcript stream (or absence of it) IS the signal.

**Implementation approach:**
- **Time-based:** Track time since last `speech_final`. If >10min, raise decision confidence threshold.
- **Explicit:** Detect "quiet mode" / "shut up" in transcripts. Suppress proactive responses for configurable duration.
- **Ambient:** If VAD detects speech but Deepgram returns empty transcripts (non-speech sound), tag as ambient noise.

**Complexity:** LOW-MEDIUM -- Time-based and explicit are trivial. Ambient classification is harder.

**Dependencies:** TS-5 (decision engine respects interruptibility signals)

**Confidence:** HIGH for time-based/explicit, MEDIUM for ambient

---

### D-7: Non-Speech Event Awareness

**What:** Detect coughs, sighs, laughter, and respond appropriately. A cough gets "bless you." Laughter might get the AI joining in.

**Value Proposition:** This makes Russel feel like a real presence in the room. No consumer assistant does this.

**How Deepgram changes this vs Whisper:** With Whisper, non-speech events appeared as rejected segments with high `no_speech_prob`. With Deepgram, the situation is different:
- Deepgram may transcribe some non-speech events as words (e.g., laughter as empty transcript with specific patterns)
- The SpeechStarted event fires for any vocal activity, including coughs -- but the resulting transcript may be empty
- The VAD gate might not pass non-speech events to Deepgram at all (Silero VAD is trained on speech, not coughs)

**This needs investigation** during implementation. The approach may need to differ from the original Whisper-based plan.

**Complexity:** MEDIUM -- Uncertain how non-speech events flow through the Deepgram pipeline.

**Dependencies:** TS-1 (Deepgram producing events), TS-3 (VAD gating may need adjustment for non-speech)

**Confidence:** LOW -- The interaction between Silero VAD, Deepgram, and non-speech events is unverified.

---

### D-8: Post-Session Library Growth (Curator Daemon)

**What:** After a session, a curator process reviews conversations, identifies response gaps, generates new quick response clips, and adds them to the response library.

**Value Proposition:** The response library grows organically based on usage. Over time, Russel's quick responses become more varied and appropriate.

**Unchanged from original plan.** This feature is independent of the STT backend. It operates on conversation logs (JSONL events), not on raw audio. The Deepgram pivot does not affect it.

**Complexity:** MEDIUM -- Carried forward from v1.2 scope.

**Dependencies:** Existing clip_factory.py, response_library.py, event bus logs

**Confidence:** HIGH (infrastructure exists)

---

## Anti-Features

Features to explicitly NOT build. These seem valuable but create problems.

---

### AF-1: Continuous Audio Streaming Without VAD Gate

**Why It Seems Good:** Simpler architecture -- just pipe all audio to Deepgram. Let Deepgram handle silence detection. No local VAD complexity.

**Why Problematic:**
1. **Cost:** 8 hours continuous = $3.70/day. With VAD gating, ~$0.74/day. Over a month: $111 vs $22.
2. **Bandwidth:** 24kHz 16-bit mono = ~48KB/s = ~173MB/hour. Over 8 hours = ~1.4GB upstream.
3. **Unnecessary load on Deepgram servers:** Empty transcripts from silence waste resources on both sides.
4. **Privacy:** Sending all ambient audio to cloud is a bigger privacy surface than sending only speech segments.

**Do This Instead:** TS-3 (VAD-gated streaming). Only send speech.

---

### AF-2: Using Interim Results for Decision Making

**Why It Seems Good:** Interim results arrive ~150ms after speech. Acting on interims would make responses faster.

**Why Problematic:** Interim results are preliminary guesses that get corrected. "What's the wetter like" becomes "What's the weather like" in the final. If the decision engine acts on "wetter," it may make wrong choices. Worse: if the system starts generating a response based on interim text that then changes, the response is based on stale/wrong input.

**Do This Instead:** Use interims ONLY for name detection (TS-6) and real-time display (D-1). Use finals/speech_final for decision engine input (TS-5). This is the Deepgram-recommended pattern.

**Exception:** Name detection on interims IS safe because "Russel" doesn't get corrected to something else -- proper nouns are stable across interim/final.

---

### AF-3: Hardware Wake Word Detection (Picovoice / openWakeWord)

**Why It Seems Good:** Dedicated wake word models run in ~80ms, faster than Deepgram's ~150-300ms for name detection.

**Why Problematic:** With Deepgram streaming already producing continuous transcripts, name detection via string matching is free, instant on each interim result, and requires zero additional infrastructure. Adding a separate audio pipeline for wake word detection adds complexity without meaningful benefit.

**Do This Instead:** TS-6 (name detection on Deepgram transcripts). The 150ms difference is not perceptible.

**When to revisit:** Only if Deepgram goes down frequently and name activation must work during STT outage.

---

### AF-4: Cloud-Based Decision Model

**Why It Seems Good:** Claude Haiku or GPT-4o-mini would be more capable than Llama 3.2 3B for the "should I respond?" decision.

**Why Problematic:** The decision model evaluates EVERY speech_final event -- potentially dozens per minute. Cloud latency (200-500ms) stacks ON TOP of Deepgram latency. The total decision chain becomes: speech -> Deepgram (300ms) -> cloud LLM (300ms) -> response (200ms+) = 800ms minimum, pushing past the natural response window. Plus: sends all conversation transcripts to a second cloud service = doubled privacy exposure.

**Do This Instead:** TS-5 (local Ollama). The Deepgram pivot frees ~3GB of GPU VRAM that was used by Whisper, making room for a larger local model if 3B proves inadequate.

---

### AF-5: Full Conversation Transcription Storage

**Why It Seems Good:** With streaming transcripts, it's trivial to log everything to disk.

**Why Problematic:** This is surveillance. An always-on mic that records and stores everything crosses a privacy line. Guests, family members, and the user themselves don't need a permanent transcript of every word spoken near the computer.

**Do This Instead:** Rolling transcript buffer (TS-4) with 5-minute TTL. Ephemeral by design. Conversation logs only capture deliberate AI interactions, not ambient speech.

---

### AF-6: Streaming to Multiple STT Providers Simultaneously

**Why It Seems Good:** Redundancy -- if Deepgram goes down, AssemblyAI or Google take over instantly.

**Why Problematic:** Double the cost, double the bandwidth, double the API keys, double the billing complexity. The CircuitBreaker + Whisper fallback (TS-7) provides sufficient redundancy: Deepgram streaming as primary, local Whisper as fallback. No need for a second cloud STT.

**Do This Instead:** TS-7 (connection lifecycle with Whisper fallback).

---

## Feature Dependencies

```
TS-1: Deepgram WebSocket Connection
  |
  +---> TS-2: Interim + Final Transcript Accumulation
  |       |
  |       +---> TS-4: Transcript Buffer Integration
  |       |       |
  |       |       +---> TS-5: Decision Engine
  |       |       |       |
  |       |       |       +---> TS-9: Response Backend (Claude/Ollama)
  |       |       |       |       |
  |       |       |       |       +---> D-4: Proactive Participation
  |       |       |       |       |       |
  |       |       |       |       |       +---> D-5: Attention Signals
  |       |       |       |       |
  |       |       |       |       +---> D-2: Word-Level Timing Analysis
  |       |       |       |
  |       |       |       +---> D-6: Interruptibility Detection
  |       |       |
  |       |       +---> D-3: Speaker Diarization
  |       |
  |       +---> D-1: Real-Time Transcript Display (interims)
  |
  +---> TS-3: VAD-Gated Audio Streaming
  |
  +---> TS-6: Name-Based Activation (on interims)
  |
  +---> TS-7: WebSocket Lifecycle Management
  |
  +---> TS-8: Echo Suppression
  |
  +---> D-7: Non-Speech Event Awareness
  |
  +---> TS-10: Resource Management (cross-cutting)

D-8: Library Growth (independent, post-session)
```

## Latency Budget

End-to-end from user stops speaking to AI starts responding:

### Fast Path (Ollama quick response)

| Stage | Time | Cumulative | Source |
|-------|------|------------|--------|
| User stops speaking | 0ms | 0ms | - |
| Endpointing silence (configurable) | 300ms | 300ms | Deepgram `endpointing=300` |
| Deepgram processes + returns speech_final | ~100-150ms | 400-450ms | Deepgram streaming latency |
| Decision engine (Ollama 3B, classify) | ~200ms | 600-650ms | Ollama with `num_predict: 100` |
| Ollama response TTFT | ~200ms | 800-850ms | Streaming response |
| TTS first sentence | ~200ms | 1000-1050ms | OpenAI TTS or Piper |
| **User hears first audio** | | **~1.0s** | |

### Deep Path (Claude CLI complex response)

| Stage | Time | Cumulative | Source |
|-------|------|------------|--------|
| User stops speaking | 0ms | 0ms | - |
| Endpointing + Deepgram final | ~400-450ms | 400-450ms | Same as fast path |
| Decision engine (Ollama 3B) | ~200ms | 600-650ms | Same as fast path |
| Claude CLI TTFT | ~2-5s | 2.6-5.6s | Network + model loading |
| TTS first sentence | ~200ms | 2.8-5.8s | OpenAI TTS |
| **User hears first audio** | | **~3-6s** | Acceptable for complex queries |

### Name-Based Activation (fastest)

| Stage | Time | Cumulative | Source |
|-------|------|------------|--------|
| User says "Hey Russel, what time is it?" | 0ms | 0ms | - |
| "Hey Russel" detected in interim | ~150ms | 150ms | Deepgram interim latency |
| Wait for speech_final (rest of utterance) | Variable | Variable | Depends on utterance length |
| Route to Ollama (name = force respond) | ~200ms | - | Skip decision engine threshold |
| **Total from end of utterance to audio** | | **~600-800ms** | |

### Comparison: Old Whisper vs New Deepgram

| Metric | Whisper Batch | Deepgram Streaming | Improvement |
|--------|--------------|-------------------|-------------|
| Silence detection | 850ms | 300ms (configurable) | 2.8x faster |
| STT processing | 500ms-2s | ~150ms (streaming) | 3-13x faster |
| Decision latency | Same | Same | - |
| Total to decision | 1.5-2.8s | 400-650ms | 2-4x faster |
| First audio to user | 2.5-4.5s | 1.0-1.5s | 2-3x faster |

---

## Cost Analysis

### Deepgram Streaming Cost (PAYG, Nova-3 monolingual)

**Rate:** $0.0077/min of audio sent

**Scenario: 8-hour workday, VAD-gated**

| Metric | Value |
|--------|-------|
| Total mic-open time | 480 min |
| Estimated speech fraction (single user, home office) | 15-25% |
| Audio sent to Deepgram | 72-120 min |
| Daily cost | $0.55-$0.92 |
| Monthly cost (22 workdays) | $12-$20 |

**With diarization (+$0.0020/min):** Add ~$0.15-0.25/day, ~$3-5/month.

**Free tier:** $200 of credits, no expiry. At $0.75/day, that is ~267 days of free usage.

**Comparison to Whisper (local):** $0/day, but 2-4x worse latency and requires 3GB GPU VRAM.

### Ollama Decision Model Cost

**Rate:** Free (local inference)

**GPU VRAM:** ~2-2.5GB (Llama 3.2 3B Q4_K_M)

**Compute per decision:** ~200ms on RTX 3070

**Decisions per day (8 hours, moderate speech):** ~200-500 (one per speech_final)

**Total daily GPU time for decisions:** ~40-100 seconds. Negligible.

---

## MVP Recommendation

For MVP (first shippable version of Deepgram streaming + decision model):

1. **TS-1: Deepgram WebSocket** -- Everything depends on this
2. **TS-2: Transcript Accumulation** -- Core data flow
3. **TS-3: VAD-Gated Streaming** -- Cost control from day one
4. **TS-7: WebSocket Lifecycle** -- Must-have for reliability
5. **TS-8: Echo Suppression** -- Without this, infinite loop
6. **TS-4: Transcript Buffer** -- Feed for decision engine
7. **TS-6: Name-Based Activation** -- Deterministic activation path
8. **TS-5: Decision Engine** -- The intelligence layer
9. **TS-9: Response Backend** -- Start with Ollama-only, add Claude routing later
10. **TS-10: Resource Management** -- Bounded buffers from day one

Defer to post-MVP:
- **D-1 (Real-Time Display):** Nice visual feedback, not blocking
- **D-2 (Word-Level Timing):** Optimization, not core functionality
- **D-3 (Speaker Diarization):** Enable if false activations from TV/guests are a problem
- **D-4 (Proactive Participation):** Start with respond-when-addressed only, lower thresholds gradually
- **D-5 (Attention Signals):** Required only when D-4 is enabled
- **D-6 (Interruptibility):** Add when proactive mode is active
- **D-7 (Non-Speech Events):** Needs investigation, defer
- **D-8 (Library Growth):** Independent track

**The critical path is: TS-1 -> TS-2 -> TS-3 -> TS-8 -> TS-4 -> TS-5 -> TS-9.**

## Phasing Recommendation

**Phase 1: Deepgram Streaming Infrastructure** (TS-1, TS-2, TS-3, TS-7, TS-8, TS-10)
- Replace Whisper STT with Deepgram streaming WebSocket
- VAD-gated audio streaming (cost optimization)
- Transcript accumulation (interim -> final -> speech_final)
- Echo suppression via PipeWire AEC + transcript gating
- WebSocket lifecycle (KeepAlive, reconnection, timestamp management)
- Resource monitoring and bounded buffers
- System transcribes continuously but makes no autonomous decisions yet
- Existing PTT-triggered pipeline still works as a fallback

**Phase 2: Decision Engine + Name Activation** (TS-5, TS-6, TS-4)
- Ollama integration for transcript monitoring
- Decision engine evaluates each speech_final
- Name-based activation ("Hey Russel") via interim transcript matching
- Transcript buffer provides context window
- Initially conservative (high confidence threshold, name-only activation)

**Phase 3: Response Backend + Integration** (TS-9)
- Ollama as quick response generator
- Claude CLI for complex/tool-using requests
- Backend selection logic
- Full pipeline integration (STT -> Decision -> Response -> TTS -> Playback)

**Phase 4: Proactive Participation** (D-4, D-5, D-6)
- Lower decision thresholds for proactive contributions
- Attention signals before unsolicited responses
- Interruptibility detection
- Conversation balance tracking

**Phase 5: Polish + Enrichment** (D-1, D-2, D-3, D-7, D-8)
- Real-time transcript display
- Word-level timing analysis
- Speaker diarization
- Non-speech event awareness
- Library growth curator

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deepgram network dependency | No STT = no transcription = deaf assistant | CircuitBreaker + Whisper fallback (already exists) |
| VAD onset clipping | First word of utterance cut off by VAD gate delay | Lookback buffer: keep last 2 silence chunks, send when speech starts |
| Endpointing too aggressive (300ms) | Splits sentences mid-thought | Start at 300ms, expose as config. User can raise to 500ms+ if needed |
| Deepgram diarization unreliable in streaming | Speaker labels wrong/missing, decision engine confused | Start WITHOUT diarization. Add as an optional enhancement once base system works |
| Decision engine latency exceeds budget | Ollama 3B takes >500ms, pushing total past natural response window | Profile on actual hardware. Fallback: reduce `num_predict`, use smaller model, or use heuristic fast-path for obvious cases (name mention = always respond) |
| KeepAlive not handled by SDK automatically | Connection drops during long silence periods | Defensive KeepAlive timer in application code, every 5 seconds during VAD silence |
| Cost higher than expected | Speech fraction higher than 20% assumption | Monitor billing via Deepgram dashboard. VAD threshold tuning (raise to reduce speech detection) |

## Sources

**Deepgram Official Documentation (HIGH confidence):**
- [Configure Endpointing and Interim Results](https://developers.deepgram.com/docs/understand-endpointing-interim-results) -- is_final, speech_final, accumulation pattern
- [Interim Results](https://developers.deepgram.com/docs/interim-results) -- Interim result behavior and timing
- [Endpointing](https://developers.deepgram.com/docs/endpointing) -- VAD-based silence detection, configurable values
- [UtteranceEnd](https://developers.deepgram.com/docs/utterance-end) -- Post-speech silence detection, 1000ms default
- [Audio Keep Alive](https://developers.deepgram.com/docs/audio-keep-alive) -- KeepAlive message format, 10s timeout
- [Speech Started](https://developers.deepgram.com/docs/speech-started) -- vad_events=true, SpeechStarted event
- [Live Audio API Reference](https://developers.deepgram.com/reference/speech-to-text/listen-streaming) -- All query parameters, response format
- [Measuring Streaming Latency](https://developers.deepgram.com/docs/measuring-streaming-latency) -- 80ms min, 674ms avg latency measurement
- [Recovering From Connection Errors](https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio) -- Reconnection, buffering, timestamp offsets
- [Speaker Diarization](https://developers.deepgram.com/docs/diarization) -- Per-word speaker IDs
- [Deepgram Pricing](https://deepgram.com/pricing) -- $0.0077/min Nova-3, $200 free credits

**Deepgram Billing (HIGH confidence):**
- [GitHub Discussion #1423](https://github.com/orgs/deepgram/discussions/1423) -- Official: "bill based on duration of audio sent," "open websocket not transmitting audio does not have a cost," "keep-alive messages do not incur a charge"

**Deepgram SDK (HIGH confidence):**
- [deepgram-sdk on PyPI](https://pypi.org/project/deepgram-sdk/) -- v5.3.2 stable, v6.0.0rc2 pre-release
- [deepgram-python-sdk GitHub](https://github.com/deepgram/deepgram-python-sdk) -- AsyncClient, event handlers, streaming

**Latency Research (MEDIUM-HIGH confidence):**
- [Twilio Voice Agent Latency Guide](https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents) -- STT 100-300ms, LLM 375ms, TTS 100ms typical breakdown
- [Tavus Turn-Taking Guide](https://www.tavus.io/post/ai-turn-taking) -- 200ms natural response gap, 300-400ms feels awkward
- [Krisp Turn-Taking Model](https://krisp.ai/blog/turn-taking-for-voice-ai/) -- Semantic endpointing, TRP detection

**Existing Codebase (HIGH confidence):**
- test_live_session.py lines 132-178: Deepgram `is_final`/`speech_final` accumulation tests
- live_session.py line 294: `CircuitBreaker("STT/Deepgram")` already exists
- requirements.txt line 24: `deepgram-sdk>=3.0` already a dependency
- continuous_stt.py: ContinuousSTT with VAD gating pattern (to be replaced by Deepgram streaming)
- transcript_buffer.py: TranscriptBuffer ring buffer (to be reused)

---
*Feature landscape research for: v2.0 Always-On Observer (Deepgram streaming pivot)*
*Researched: 2026-02-22*
*Supersedes: 2026-02-21 features research (Whisper batch-based)*
