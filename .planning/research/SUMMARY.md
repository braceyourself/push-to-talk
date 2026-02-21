# Project Research Summary

**Project:** Push-to-Talk v2.0 — Always-On Observer
**Domain:** Always-on desktop voice assistant with local LLM monitoring layer
**Researched:** 2026-02-21
**Confidence:** MEDIUM-HIGH overall (stack HIGH, architecture HIGH, pitfalls HIGH, features MEDIUM on LLM decision quality)

## Executive Summary

The v2.0 milestone transforms Russel from a push-to-talk assistant into an ambient participant: the microphone stays open permanently, a local Llama 3.2 3B model (via Ollama) continuously monitors transcripts and decides when to respond, and responses route to either Claude CLI (deep/tool-using) or Ollama (fast/conversational) based on query complexity. The codebase is well-positioned for this transformation — audio capture already runs in a daemon thread, Silero VAD is already loaded for barge-in, and the StreamComposer/TTS/playback stack requires zero changes. The primary structural surgery is replacing the gated `_stt_stage` + `_llm_stage` pair with three new independently-testable components: a continuous STT layer, a transcript buffer, and a monitor loop.

The recommended approach is to build and validate each new component in isolation before integration. The continuous STT (decoupled from the monitor) and the monitor loop (decoupled from the response router) can each be tested against recorded audio and mock buffers respectively, before wiring them into the live pipeline. This isolates the two highest-risk areas — Whisper rolling-window transcription accuracy and Llama 3.2 3B decision quality — from each other and from the existing proven pipeline. The existing PTT mode must remain intact throughout all phases; always-on should be added as a new `ai_mode` value, not a replacement.

The dominant risks cluster around three areas that must all be resolved in Phase 1 before any monitoring intelligence is built: the audio feedback loop (the AI hearing its own TTS output and responding to it), GPU VRAM pressure from running Whisper and Ollama simultaneously on an RTX 3070 (8GB), and Whisper hallucinations exploding when exposed to continuous ambient audio rather than clean PTT segments. Starting with name-based activation only — respond only when "Russel" is explicitly said — and gradually unlocking proactive participation after the system is validated is the right UX strategy. One false positive that interrupts a phone call will cause users to disable the feature permanently.

## Key Findings

### Recommended Stack

The stack additions for v2.0 are minimal and deliberately lean. Ollama server (system-level install) with Llama 3.2 3B (2.0GB model, Q4_K_M quantization, ~2-2.5GB VRAM) provides the local monitoring and quick-response LLM. The `ollama` Python package (0.6.1) adds zero new transitive dependencies — it requires only `httpx` and `pydantic`, both already installed. No other new dependencies are needed. The wake word question is resolved: Whisper transcript string-matching replaces a dedicated wake word engine because Whisper is already running continuously and checking `if "russel" in transcript.lower()` is trivial.

The critical hardware constraint is VRAM. The recommended configuration uses `distil-large-v3` (not `large-v3`) for continuous STT — it delivers near-large-v3 accuracy at ~1.5GB VRAM versus ~3.5GB, leaving comfortable headroom for Ollama. The fallback chain is: distil-large-v3 → small (0.5GB) → Whisper on CPU. Ollama must be configured with `OLLAMA_KEEP_ALIVE=-1` to prevent model eviction after idle periods, plus a 2-minute heartbeat keepalive to guard against the documented `keep_alive` GPU loading bug (Ollama issue #9410).

**Core technologies:**
- Ollama server + Llama 3.2 3B: local monitoring LLM and quick-response backend — zero network dependency, ~200ms inference, fits 8GB VRAM alongside Whisper
- `ollama` 0.6.1 Python client: async Ollama integration with structured JSON output via `format` parameter — zero new transitive deps
- Whisper distil-large-v3: continuous STT replacement for large-v3 — 51% smaller, 6x faster, within 1% WER
- Silero VAD (already loaded): repurposed from barge-in-only to always-on speech gate — prevents Whisper running on silence/noise
- PipeWire echo cancel module: WebRTC AEC via `libpipewire-module-echo-cancel` — primary defense against AI-hears-itself feedback loop

See STACK.md for full installation commands, VRAM budget table, Ollama API quick reference, and performance budget.

### Expected Features

The feature set splits cleanly into table stakes (required for always-on to work at all) and differentiators (what makes this better than Alexa/Siri). The critical path through table stakes is: TS-1 → TS-2 → TS-3 → TS-5. Everything else builds on top of this chain.

**Must have (table stakes):**
- TS-1: Continuous audio capture with VAD gating — prerequisite for everything; Silero VAD already present in codebase
- TS-2: Rolling transcript buffer (ring buffer, 5-min window, ~2000 tokens max) — the monitoring LLM's working memory
- TS-3: Response decision engine (Ollama + Llama 3.2 3B, structured JSON output: action/backend/confidence/reasoning) — "should I respond?" classifier; this is the hardest and riskiest feature
- TS-4: Name-based activation ("hey Russel" via transcript string matching with fuzzy variants) — deterministic activation path, required for user trust
- TS-5: Configurable response backend (Claude CLI for deep/tool queries, Ollama for fast conversational) — routes decisions to appropriate LLM
- TS-6: Resource management (bounded buffers, VRAM monitoring, per-inference memory checks) — cross-cutting concern, must be built in from day one
- TS-7: Graceful degradation (Ollama down → Claude only; network down → Ollama only; GPU pressure → smaller Whisper model)

**Should have (differentiators, Phase 4+):**
- D-1: Proactive conversation participation (Inner Thoughts CHI 2025 framework — 8 heuristics: relevance, information gap, urgency, balance, dynamics, etc.) — the core differentiator vs every existing assistant
- D-2: Conversation-aware response calibration (Ollama decision includes `tone` field: casual/focused/supportive/excited, passed to response backend as system prompt overlay)
- D-3: Non-speech event awareness (cough/laughter/sigh response categories using existing Whisper rejection metadata — already partially implemented, just currently discarded)
- D-6: Attention signals before proactive responses (brief verbal cue or audio chime before interjecting, using existing StreamComposer + response library)
- D-7: Interruptibility detection (time-based: raise threshold if silent >10min; explicit: "quiet mode" command; OS: fullscreen app check)

**Defer to post-v2.0:**
- D-4: Post-session library growth curator — clip_factory.py infrastructure exists; needs usage analysis logic
- D-5: Multi-speaker diarization — add only if false activations from TV/others become a confirmed real problem
- Hardware wake word engine — revisit only if Whisper-based name detection proves too slow
- Schedule-based proactivity (Alexa Hunches pattern) — different product, out of scope
- Cloud-based monitoring LLM — privacy violation; local-first is a core value

See FEATURES.md for the full dependency graph, feature complexity ratings, and key risks table.

### Architecture Approach

The v2.0 architecture restructures the sequential 5-stage pipeline (capture → STT → LLM → TTS → playback) into three independent loops connected by the transcript buffer: an always-on input stream (audio capture + continuous Whisper + VAD gating), a monitor loop (Ollama polling the transcript buffer, emitting ResponseDecision objects), and a response layer (ResponseRouter dispatching to Claude CLI or Ollama, feeding the existing StreamComposer). The output stack (StreamComposer, TTS, PyAudio playback) is entirely unchanged. The support layer (event bus, filler manager, learner daemon, task manager, SSE dashboard) is also unchanged — new event types are additive, not breaking.

The single biggest coupling point to break is `_stt_gated` (live_session.py line 211): the flag that suppresses STT during AI playback. Removing this flag — and replacing it with PipeWire AEC + transcript fingerprinting — is what makes continuous STT possible. The monitor loop batches transcripts rather than calling Ollama on every chunk: accumulate segments, wait for 2-second silence gap, check minimum 5-word count, then evaluate. This means ~1-2 Ollama calls per user utterance, not per 85ms audio chunk.

**Major components:**
1. `ContinuousSTT` (new file) — rolling-window Whisper transcription producing `TranscriptSegment` objects; owns its own thread executor; VAD-gated; handles interim-to-final segment replacement
2. `TranscriptBuffer` (new file) — thread-safe asyncio ring buffer with `wait_for_new()`, `get_context()`, `get_since()` API; 5-min max age, 2048-token budget
3. `MonitorLoop` (new file) — asyncio coroutine polling TranscriptBuffer; calls Ollama with structured JSON output; silence threshold + cooldown + minimum-word-count gates before invoking Ollama; emits `ResponseDecision(action, backend, confidence, prompt, reasoning)`
4. `ResponseRouter` (new file) — dispatches ResponseDecision to Claude CLI (reusing existing `_send_to_cli`/`_read_cli_response`) or Ollama streaming; sentence-boundary detection feeds existing StreamComposer
5. `OllamaClient` (new file) — thin async wrapper around `ollama` library; handles keepalive heartbeat, error handling, structured output, and streaming

Build order (each independently testable before pipeline integration): Phase A (ContinuousSTT + TranscriptBuffer) → Phase B (MonitorLoop with mock buffer) → Phase C (ResponseRouter + Ollama backend) → Phase D (pipeline integration in live_session.py) → Phase E (barge-in + name detection) → Phase F (tuning).

See ARCHITECTURE.md for complete ASCII diagrams, full data flow lifecycle examples (normal turn, barge-in, name interrupt, ignored background conversation), anti-patterns to avoid, and configuration additions.

### Critical Pitfalls

All five critical pitfalls must be solved in Phase 1 before the monitoring intelligence is built. There is no point building a sophisticated decision engine on top of a system that feeds itself hallucinated transcripts of its own voice or crashes after 30 minutes due to GPU OOM.

1. **Audio feedback loop** (CRITICAL, Phase 1) — The AI hears its own TTS output through the always-on mic and responds to itself; well-documented in OpenAI Realtime API community reports. Prevention: PipeWire `libpipewire-module-echo-cancel` (WebRTC AEC) as primary; transcript fingerprinting against `_spoken_sentences` ring buffer as secondary; `during_ai_speech=True` tagging as tertiary. Must be set up before any continuous listening code runs.

2. **GPU VRAM exhaustion** (CRITICAL, Phase 1) — Whisper large-v3 (~3.5GB) + Ollama 3B (~2-2.5GB) + overhead leaves <2GB headroom on 8GB RTX 3070; temporary activation spikes during simultaneous inference can OOM. Prevention: switch continuous STT to distil-large-v3 (~1.5GB); cap Ollama `num_ctx` at 2048 (`options={"num_ctx": 2048}`); real-time VRAM monitoring; validate empirically before Phase 2.

3. **Whisper hallucinations explode in continuous mode** (CRITICAL, Phase 1) — Whisper has a 40.3% hallucination rate on non-speech audio (keyboard, HVAC, TV); the existing 18-phrase filter was tuned for PTT where audio is almost always genuine speech. Prevention: more aggressive VAD parameters (`min_speech_duration_ms: 250`, `threshold: 0.7`); expanded hallucination phrase list (~30 well-documented phrases + single-word fillers); confidence gating (>3 words, logprob threshold, energy check); rate-limit STT during long quiet periods.

4. **Monitoring LLM context grows unboundedly** (CRITICAL, Phase 2) — Naive implementation appends all transcripts; 30 minutes of ambient audio fills the Ollama context window, causing truncation or VRAM growth. Prevention: two-tier context (rolling summary ~100-200 tokens + recent raw buffer ~500 tokens); event-driven triggering rather than continuous polling; explicit context reset on topic change or long silence (>5 min).

5. **False positive responses** (CRITICAL, Phase 2 and ongoing) — The AI speaks when it shouldn't (phone call, TV dialogue, self-talk); one bad interjection during a phone call causes users to disable the feature permanently. Prevention: default to NOT responding; name-based activation only for v2.0 initial launch; confidence threshold >0.8; 60-second cooldown after proactive responses; track suppression rate (user interrupts within 2s = likely false positive).

See PITFALLS.md for 13 additional major/moderate pitfalls with detailed prevention strategies, phase-to-pitfall mapping table, and warning sign indicators.

## Implications for Roadmap

All four research files independently converged on the same phase breakdown, giving high confidence in this structure. The FEATURES.md phasing recommendation, ARCHITECTURE.md build order, and PITFALLS.md phase warnings all align.

### Phase 1: Infrastructure and Safety Net

**Rationale:** The five critical pitfalls all belong here. No monitoring intelligence is useful without a clean audio stream, a stable VRAM budget, and robust hallucination filtering. This phase produces no visible AI behavior change — always-on audio capture runs but makes no autonomous decisions. PTT mode continues working unchanged. The foundation must be proven solid before building intelligence on it.

**Delivers:** Continuous VAD-gated audio capture decoupled from LLM; `ContinuousSTT` producing `TranscriptSegment` objects with rolling-window Whisper; `TranscriptBuffer` with bounded ring buffer; PipeWire AEC configured and validated; VRAM measured empirically with both models loaded simultaneously; expanded hallucination filter (30+ phrases, VAD parameter tuning, confidence gating); profile-based config schema (`"mode": "always_on"` vs `"mode": "ptt"` with sensible defaults); startup validation (AEC configured? Ollama running? VRAM sufficient?); persistent system tray indicator showing always-on state; JSONL rotation policy (24-hour auto-purge).

**Addresses:** TS-1 (partial), TS-6, TS-7 (partial), Pitfalls 1, 2, 3, 9, 10, 16, 17

**Avoids:** Building monitoring intelligence before the audio stream is clean, VRAM budget is validated, and hallucination rate is under control.

**Research flag:** PipeWire echo cancellation device selection — the module is documented, but the exact `pasimple` device name (`echo-cancel-capture`) and routing in this specific PipeWire version needs verification on this machine. Targeted 30-minute spike recommended before implementation.

### Phase 2: Monitoring Decision Engine

**Rationale:** With a clean, bounded, low-hallucination transcript stream in place, the monitoring LLM can be built and tuned in isolation using mock buffers. This phase adds Ollama integration, name-based activation (the deterministic path), and the monitor loop — but keeps the response threshold maximally conservative (respond only on explicit name mention). This validates the Ollama pipeline and decision quality before any proactive logic is added.

**Delivers:** `OllamaClient` with structured JSON output, keepalive configuration (`OLLAMA_KEEP_ALIVE=-1` in systemd + 2-min heartbeat ping), pre-load on session start, error handling (`ResponseError` catch, 404 = pull model); `MonitorLoop` with silence threshold (2.0s), minimum-word-count gate (5 words), cooldown (3.0s), and `_responding` flag; name-based activation ("russel", "russell", "rusel", "russ" fuzzy variants + `initial_prompt` Whisper bias for better name transcription); two-tier context management (rolling summary + recent buffer); event bus additions (`transcript_segment`, `monitor_decision`, `wake_phrase_detected`, `ollama_inference`).

**Addresses:** TS-2 (full), TS-3, TS-4, Pitfalls 4, 5, 6, 7, 15, 18

**Uses:** `ollama` 0.6.1 Python client, Llama 3.2 3B structured output with JSON schema `format` parameter, `options={"temperature": 0, "num_predict": 100, "num_ctx": 2048}`

**Research flag:** Monitoring LLM decision quality is LOW confidence — Llama 3.2 3B's accuracy on respond/wait/ignore/acknowledge classification against realistic ambient conversation transcripts is unknown. Create a benchmark dataset (20-30 examples: phone call, TV dialogue, direct question, self-talk, multi-person chat) and evaluate before building Phase 2 fully. If accuracy is inadequate (<80% correct classification), fall back to heuristic pre-filter (existing input classifier fast path) + Ollama only for ambiguous cases.

### Phase 3: Response Backend and Pipeline Integration

**Rationale:** With the monitor producing validated decisions, the response layer can be wired in. This is the highest-risk integration phase (replacing `_stt_stage` and `_llm_stage` in `live_session.py`) but each new component has been independently tested by this point. Two backends are integrated behind a `ResponseRouter` with circuit-breaker fallback. Use the simple initial routing rule: name-triggered → Claude CLI, ambient/proactive → Ollama.

**Delivers:** `ResponseRouter` dispatching to Claude CLI (reusing `_send_to_cli`/`_read_cli_response`) or Ollama streaming (sentence-boundary detection matching existing pattern); pipeline integration in `live_session.py` (replace `_stt_stage`/`_llm_stage` with new coroutines, update `run()`, remove `_stt_gated` flag); barge-in extended with name-detection path alongside VAD path; quick-response filler bridge (filler plays immediately on "should respond" decision while response LLM generates, using existing `_filler_manager` + StreamComposer); backend circuit breaker extending existing `CircuitBreaker` class (line 148); regression test run against full test suite after every change.

**Addresses:** TS-1 (full, pipeline wired), TS-5, TS-7 (full), Pitfalls 8, 13, 14, 17 (ongoing)

**Research flag:** Standard patterns apply. ResponseRouter Ollama streaming → StreamComposer integration is fully specified in ARCHITECTURE.md. The pipeline integration (Phase D in architecture) is high implementation risk but the approach is clear — no deeper research needed, just careful execution and regression testing.

### Phase 4: Proactive Participation

**Rationale:** Phases 1-3 deliver a working always-on system that responds when explicitly addressed by name. Phase 4 adds the core differentiator: proactive participation in conversations without being addressed. This must come after the name-only system is validated in production — the worst outcome is loosening thresholds on an untested monitoring pipeline. The Inner Thoughts CHI 2025 framework provides the theoretical model for this phase.

**Delivers:** Proactive response threshold configuration (Ollama prompt encoding of Inner Thoughts 8 heuristics: relevance, information gap, urgency, balance, dynamics; configurable `proactivity_level: 0-5` mapping to `imThreshold` parameter); attention signals (brief verbal cue "hey" / "actually" or audio chime played before proactive interjection via StreamComposer + response library new category); conversation-aware tone calibration (Ollama decision includes `tone` field passed to response backend system prompt modifier); interruptibility detection (time-based: raise threshold if no speech >10min; explicit: "quiet mode" voice command or key shortcut; OS: check for fullscreen app); non-speech event awareness (cough → "bless you", laughter → reaction, sigh → "rough day?" — using existing Whisper rejection metadata, requires high confidence ≥0.85).

**Addresses:** D-1, D-2, D-3, D-6, D-7

**Research flag:** Proactive participation prompt engineering is inherently experimental. Plan for 2-3 iteration cycles on the Ollama monitoring prompt. The research (Inner Thoughts framework) provides the conceptual model, but the encoding for a 3B model and the right default aggressiveness level are unknown. Budget explicit tuning time here — do not treat this as a one-shot implementation.

### Phase 5: Reliability and Library Growth

**Rationale:** D-4 (library growth curator) and multi-speaker diarization (D-5) are independent tracks that don't block the core always-on experience. Library growth reuses existing clip_factory.py infrastructure. Multi-speaker diarization should only be added if false activations from other people/TV become a confirmed problem in Phase 4 production use — do not add preemptively.

**Delivers:** Curator daemon (post-session, reviews JSONL logs, identifies missing response phrases, generates via Piper TTS, quality-evaluates, adds to response library); extended reliability (faster-whisper memory leak monitoring — RSS every 5 minutes, alert on >20% growth; periodic model reload every 2 hours to release leaked tensors; 4-hour soak test before shipping); event bus rate limiting for continuous events (only log state changes, not every evaluation); always-on observability panel in SSE dashboard (VRAM usage, hallucination rate, monitoring decision per minute, response latency histogram).

**Addresses:** D-4, D-5 (conditional), Pitfalls 11, 12, 18

**Research flag:** Standard patterns. Library growth curator follows the existing `learner.py` daemon pattern. Reliability work is standard engineering discipline. No deeper research needed.

### Phase Ordering Rationale

- Phases 1-3 follow the hard dependency chain: clean audio → monitoring intelligence → response routing. These cannot be reordered.
- Phase 3 (integration) comes before Phase 4 (proactive) because tuning thresholds on an unintegrated system is wasted work. The end-to-end pipeline must be operational before restrictions are loosened.
- Phase 4 (proactive) is explicitly gated behind Phase 3 production validation. The research is unambiguous: false positives are more damaging than missed responses. Conservative first, looser after evidence of reliability.
- Phase 5 is decoupled and can be started in parallel with Phase 4 tuning.
- PTT mode preservation is a continuous requirement across all phases — run the full test suite after every always-on change.

### Research Flags

Phases needing deeper research during planning:
- **Phase 1:** PipeWire echo cancellation device selection — `libpipewire-module-echo-cancel` documentation covers configuration, but the `pasimple` virtual device name and routing specific to this machine's PipeWire version needs a targeted spike before implementation begins.
- **Phase 2:** Llama 3.2 3B monitoring decision quality — benchmark against 20-30 realistic transcript scenarios before committing to the architecture. If accuracy is below 80%, the fallback (heuristic pre-filter + Ollama for ambiguous only) must be built instead of the pure-LLM approach.

Phases with standard patterns (skip research-phase):
- **Phase 3:** Architecture fully specified in ARCHITECTURE.md with working code examples. Careful implementation work, not novel research.
- **Phase 4:** Inner Thoughts framework is the research. Encoding it in prompts requires iteration, not more literature review.
- **Phase 5:** Curator follows `learner.py` pattern. Reliability is standard engineering practice.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Ollama + Llama 3.2 3B verified against official library docs, PyPI, model registry. Python client dependencies confirmed present in venv. VRAM estimates are MEDIUM — community benchmarks, not empirically verified on this specific hardware with both models loaded simultaneously. |
| Features | MEDIUM-HIGH | Table stakes features are well-understood with clear implementation paths. Differentiator D-1 (proactive participation) is MEDIUM — Inner Thoughts CHI 2025 framework validates the concept, Gemini Proactive Audio validates the commercial viability, but Llama 3.2 3B's ability to implement the 8-heuristic decision reliably is LOW confidence until benchmarked. |
| Architecture | HIGH | Codebase fully read (live_session.py 2900+ lines, all other modules). Existing coupling points identified with exact line references. New component APIs designed with working code examples. Build order validated against component dependencies. Anti-patterns explicitly identified. |
| Pitfalls | HIGH | 18 pitfalls documented, verified against codebase line references, hardware specs, official documentation, and academic research. Critical pitfalls have 2-3 independent sources. Phase-to-pitfall mapping is explicit and actionable. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **VRAM empirical validation (Phase 1 blocker):** Distil-large-v3 + Ollama Llama 3.2 3B VRAM under concurrent inference load has not been measured on the actual RTX 3070. This must be the first thing validated in Phase 1 — install both, run simultaneously, stress with `nvidia-smi dmon`. If peak VRAM exceeds 7GB, switch to Whisper `small` model before building anything else.

- **Llama 3.2 3B monitoring accuracy (Phase 2 blocker):** The model's respond/wait/ignore/acknowledge classification accuracy on real ambient conversation transcripts is unknown. Create a benchmark dataset before building Phase 2 fully. If results are inadequate, the monitoring architecture changes significantly (heuristic primary path, LLM secondary).

- **PipeWire AEC device selection (Phase 1):** The exact virtual source device name exposed by `libpipewire-module-echo-cancel` and how to select it in `pasimple.PaSimple()` needs verification on this specific PipeWire setup. 30-minute investigation spike recommended.

- **Ollama Docker vs native latency (Phase 1):** Ollama runs as a Docker container alias on this machine. Docker GPU passthrough adds overhead. The architecture research recommends evaluating native Ollama installation for lower inference latency. Worth measuring before committing to the deployment approach.

- **Silence threshold tuning (Phase 3):** The 2-second silence threshold in MonitorLoop is the single most important UX parameter — it determines how long after the user stops speaking before the AI responds. Too short = premature evaluation while user is mid-thought. Too long = conversational sluggishness. Starting value of 2.0s is research-informed but requires tuning with real usage.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `live_session.py` (2900+ lines, fully read with line references), `stream_composer.py`, `pipeline_frames.py`, `event_bus.py`, `input_classifier.py`, `learner.py`, `push-to-talk.py` — direct code inspection
- [Ollama Python library GitHub](https://github.com/ollama/ollama-python) — AsyncClient API, streaming, structured output
- [Ollama /api/chat docs](https://docs.ollama.com/api/chat) — format, options, keep_alive parameters
- [Ollama structured outputs docs](https://docs.ollama.com/capabilities/structured-outputs) — JSON schema via format parameter, verified for Llama 3.2
- [Ollama FAQ](https://docs.ollama.com/faq) — keep_alive, OLLAMA_MAX_LOADED_MODELS, GPU memory management
- [Llama 3.2 model page](https://ollama.com/library/llama3.2) — 3B size, 2.0GB on disk, 128K context window
- [PipeWire: Echo Cancel Module](https://docs.pipewire.org/page_module_echo_cancel.html) — official WebRTC AEC configuration
- `nvidia-smi` on development machine — RTX 3070, 8192 MB VRAM confirmed
- pip venv inspection — httpx 0.28.1, pydantic 2.12.5, faster-whisper 1.2.1, ollama (not yet installed) confirmed present/absent

### Secondary (MEDIUM confidence)
- [Inner Thoughts: Proactive Conversational Agents (CHI 2025)](https://arxiv.org/html/2501.00383v2) — 8-heuristic proactive participation framework, overt/covert/tonal proactivity parameters
- [Better to Ask Than Assume (CHI 2024)](https://dl.acm.org/doi/10.1145/3613904.3642193) — attention signal pattern ("utterance starter") before proactive VA responses
- [Proactivity Dilemma (CUI 2022)](https://dl.acm.org/doi/10.1145/3543829.3543834) — interruptibility as primary factor in proactive VA acceptance; personal context matters more than content quality
- [Gemini Proactive Audio (Google Cloud)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/proactive-audio) — validates "model decides when to respond" as a production pattern
- [WhisperLive](https://github.com/collabora/WhisperLive) / [whisper_streaming](https://github.com/ufal/whisper_streaming) — rolling-window approach for continuous Whisper transcription
- [distil-whisper HuggingFace](https://huggingface.co/distil-whisper/distil-large-v3) — 6.3x faster, 51% smaller, within 1% WER vs large-v3
- [arXiv 2501.11378: Whisper ASR Hallucinations](https://arxiv.org/abs/2501.11378) — 40.3% hallucination rate on non-speech audio, top hallucinated phrases documented
- [Ollama VRAM guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) — 3B model ~2-3GB VRAM with Q4 quantization
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad) — <1ms per chunk CPU, MIT license, 2MB model, verified performance
- [OpenAI Community: Realtime API feedback loop](https://community.openai.com/t/realtime-api-starts-to-answer-itself-with-mic-speaker-setup/977801) — AI-hears-itself feedback loop documented in production deployments
- [faster-whisper memory leak issue #249](https://github.com/guillaumekln/faster-whisper/issues/249) — memory growth in long transcription sessions
- [GitHub: ollama/ollama Issue #9410](https://github.com/ollama/ollama/issues/9410) — keep_alive GPU loading bug documentation

### Tertiary (LOW confidence)
- VRAM estimates for simultaneous Whisper distil-large-v3 + Ollama Llama 3.2 3B under concurrent inference load — not empirically verified on RTX 3070; community benchmarks only; must measure in Phase 1
- Llama 3.2 3B respond/ignore decision accuracy for ambient monitoring task — estimated from general model capability benchmarks; not tested against this specific classification task
- Ollama Docker GPU passthrough vs native latency — community reports suggest overhead exists; not measured on this hardware

---
*Research completed: 2026-02-21*
*Ready for roadmap: yes*
