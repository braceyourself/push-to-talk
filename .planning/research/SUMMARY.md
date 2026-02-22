# Project Research Summary

**Project:** Push-to-Talk v2.0 -- Deepgram Streaming STT + Local Decision Model
**Domain:** Desktop always-on voice assistant (cloud STT, local LLM, Linux/PipeWire)
**Researched:** 2026-02-22
**Confidence:** HIGH (Deepgram SDK/API docs verified, codebase fully inspected, existing test patterns mapped)
**Supersedes:** Previous SUMMARY.md (2026-02-21, local Whisper-based architecture)

## Executive Summary

This project pivots the push-to-talk voice assistant from local batch-mode Whisper STT to Deepgram Nova-3 streaming STT, while adding a local Ollama-based decision model that monitors conversation and decides when the AI should respond. The pivot trades free local inference for cloud-billed streaming ($0.0077/min of audio sent) but gains transformative latency improvements: end-to-end STT drops from 1.3-2.8s (Whisper batch with silence detection) to ~450ms (Deepgram streaming with endpointing), and removing Whisper from GPU frees ~2GB VRAM, enabling an upgrade from Llama 3.2 3B to Llama 3.1 8B for significantly better classification quality. The existing codebase is well-prepared: the Deepgram SDK is already a dependency, tests already validate the `is_final`/`speech_final` accumulation pattern, the TranscriptBuffer ring buffer requires no changes, and the `CircuitBreaker` class already supports STT fallback.

The recommended approach is a drop-in replacement architecture. A new `DeepgramSTT` class replicates the `ContinuousSTT` interface (same `start()`/`stop()`/`set_playing_audio()` methods, same `TranscriptSegment` output). The `_stt_stage()` in `live_session.py` becomes a thin consumer of Deepgram transcript output instead of running Whisper internally. Everything downstream of STT (LLM stage, StreamComposer, TTS, playback) is unchanged. Whisper stays installed as an offline fallback but is not loaded by default (0 VRAM when idle).

The single most critical risk is VAD gating architecture. The original plan uses Silero VAD to filter silence before sending audio to Deepgram, but Deepgram's engineering team explicitly recommends against per-chunk VAD gating -- it breaks server-side endpointing and degrades accuracy because silence is context for the STT model. The correct approach is to stream audio continuously during active conversation, use KeepAlive during extended idle (free), and disconnect during sleep periods. The second major risk is echo cancellation: with cloud STT, PipeWire AEC residual echo gets transcribed by Deepgram and can create feedback loops. Defense-in-depth is required: AEC + transcript fingerprinting against recent AI speech + playback-aware suppression with extended post-playback cooldown to account for network latency.

## Key Findings

### Recommended Stack

The stack adds one new dependency (`deepgram-sdk` 5.3.2) and upgrades the decision model from Llama 3.2 3B to Llama 3.1 8B. Total new disk usage is ~4.8GB (SDK + 8B model download). The VRAM budget on the RTX 3070 (8GB) goes from ~5-6GB used (Whisper 2GB + 3B 2.5GB) to ~6.5GB (8B model 5GB + overhead 1.5GB), with ~1.7GB headroom. Whisper is removed from the GPU entirely -- Deepgram handles STT in the cloud.

**Core technologies:**
- **Deepgram Nova-3** (cloud streaming STT): ~150ms first-word latency, ~6.84% WER, $0.0077/min of audio sent. KeepAlive and idle WebSocket connections cost nothing. $200 free credit covers 870+ days of light use.
- **deepgram-sdk 5.3.2**: Official Python SDK. Pin `>=5.3,<6.0` -- v6.0.0-rc.2 has breaking API changes. Synchronous client recommended (manages its own WebSocket thread, avoids async event loop conflicts with existing pipeline).
- **Llama 3.1 8B** (Q4_K_M via Ollama, ~5GB VRAM): Replaces 3.2 3B. Meaningfully better reasoning for nuanced "should I respond?" classification (+8 MMLU points). Cap `num_ctx` at 4096 to bound KV cache growth. Fallback to 3B if VRAM pressure occurs.
- **Silero VAD** (ONNX, CPU): Repurposed from barge-in detection to connection lifecycle management. Controls when to stream audio vs send KeepAlive vs disconnect WebSocket entirely. NOT to be used as a per-chunk audio filter before Deepgram.
- **faster-whisper** (kept as offline fallback): Not loaded by default. 0 VRAM when idle. Activates via CircuitBreaker when Deepgram is unreachable.

**Critical version requirements:**
- `deepgram-sdk>=5.3,<6.0` (avoid v6 breaking changes)
- Ollama `num_ctx` capped at 4096 tokens (32K context would exhaust all 8GB VRAM for KV cache alone)
- `OLLAMA_KEEP_ALIVE=-1` to prevent model eviction during idle

### Expected Features

The feature set splits into table stakes (required for always-on to work) and differentiators (what makes this better than existing assistants). The critical path is: TS-1 (WebSocket) -> TS-2 (transcript accumulation) -> TS-3 (cost management) -> TS-8 (echo suppression) -> TS-4 (buffer integration) -> TS-5 (decision engine) -> TS-9 (response backend).

**Must have (table stakes):**
- TS-1: Deepgram WebSocket streaming connection -- foundation for all STT
- TS-2: Interim/final/speech_final transcript accumulation -- core data flow (pattern already tested in test_live_session.py)
- TS-3: VAD-gated cost optimization -- must be connection-lifecycle gating (active/idle/sleep), NOT per-chunk audio filtering
- TS-5: Decision engine via Ollama -- classifies each utterance for respond/don't-respond with structured JSON output
- TS-6: Name-based activation -- deterministic "Hey Russel" trigger via transcript string matching (check both interim and final results)
- TS-7: WebSocket lifecycle management -- KeepAlive (text frame, every 5s), reconnection with exponential backoff, timestamp realignment
- TS-8: Echo suppression -- PipeWire AEC + transcript fingerprinting + playback-aware tagging with extended cooldown
- TS-9: Response backend routing -- Ollama for quick responses, Claude CLI for complex/tool-using queries
- TS-10: Resource management for continuous operation -- bounded buffers, connection health monitoring

**Should have (differentiators):**
- D-1: Real-time transcript display from interim results -- "glass box" feedback showing system is listening
- D-4: Proactive conversation participation -- AI speaks up when relevant, not just when addressed
- D-5: Attention signals before proactive responses -- prevents startling the user
- D-6: Interruptibility detection -- suppress proactive responses during focus time

**Defer to post-MVP:**
- D-2: Word-level timestamp analysis for response timing -- optimization, not core
- D-3: Speaker diarization via Deepgram -- streaming diarization quality unverified, historical issues exist
- D-7: Non-speech event awareness -- interaction with VAD gating and Deepgram is unclear
- D-8: Library growth curator -- independent track, operates on session logs post-session

### Architecture Approach

The integration is a surgical replacement with minimal blast radius. `DeepgramSTT` replaces `ContinuousSTT` with the same external interface. It owns its own pasimple audio capture thread (proven pattern from the class it replaces), runs Silero VAD locally for connection lifecycle decisions, and manages the Deepgram WebSocket connection. The Deepgram SDK's synchronous client manages its own internal WebSocket thread; `on_message` callbacks fire on that thread and must bridge to the asyncio event loop via `loop.call_soon_threadsafe()`. Deepgram's `speech_final` flag maps directly to the existing `END_OF_UTTERANCE` frame type -- the LLM stage does not know or care that transcripts came from Deepgram instead of Whisper.

**Major components:**
1. **DeepgramSTT** (new `deepgram_stt.py`) -- audio capture thread, Silero VAD for lifecycle management, Deepgram WebSocket connection, `is_final`/`speech_final` accumulation, KeepAlive timer, exponential backoff reconnection, `TranscriptSegment` emission
2. **Rewritten `_stt_stage()`** (in `live_session.py`) -- thin consumer of `_deepgram_transcript_q`, applies gating/mute checks, emits `END_OF_UTTERANCE` + `TRANSCRIPT` pipeline frames
3. **Decision Engine** (future phase) -- Ollama + Llama 3.1 8B reading TranscriptBuffer context on each `speech_final`, structured JSON classification, confidence-gated response routing
4. **Unchanged:** TranscriptBuffer, StreamComposer, LLM stage (Claude CLI), TTS/playback pipeline, event bus, dashboard

**Build order:** Step 1 (DeepgramSTT standalone, independently testable) -> Step 2 (rewrite `_stt_stage` to consume Deepgram output) -> Step 3 (clean up old Whisper code) -> Step 4 (decision model integration, separate phase)

### Critical Pitfalls

All five critical pitfalls must be addressed in Phase 1 before building monitoring intelligence.

1. **VAD gating breaks Deepgram endpointing** (CRITICAL) -- Deepgram team explicitly recommends against pre-filtering silence. `speech_final` never fires because Deepgram never sees the silence gap. Stream continuously during active periods; use KeepAlive during idle; disconnect during sleep. This is the foundational architectural decision.

2. **Audio format silent failure** (CRITICAL) -- Deepgram silently returns empty transcripts if `encoding=linear16` and `sample_rate=24000` are not explicitly set. No error message. Must validate on first connection that transcripts arrive within 10 seconds of streaming speech audio.

3. **WebSocket disconnection drops mid-utterance transcript** (CRITICAL) -- Network blips kill the connection mid-speech, losing partial transcripts. Buffer audio locally during reconnection (maxsize=200 chunks, ~17s). Exponential backoff with jitter. Finalize message before intentional disconnect. Timestamp realignment after reconnection.

4. **Cost runaway without lifecycle management** (CRITICAL) -- $332/month if streaming 24/7. Activity-based lifecycle: active mode (streaming audio) -> idle mode (KeepAlive, free) -> sleep mode (disconnected, zero cost). Daily budget cap with Whisper fallback. Cost tracking counter in dashboard.

5. **Echo cancellation feedback loop** (CRITICAL) -- PipeWire AEC residual echo gets transcribed by Deepgram, decision model sees AI's own words. Defense-in-depth: AEC primary + transcript fingerprinting against `_spoken_sentences` secondary + extended post-playback cooldown (200-500ms beyond playback end for network latency) tertiary.

Additional major pitfalls: `speech_final` fails in noisy environments (dual-trigger: endpointing + `utterance_end_ms`); decision model false positives from TV/music (name-activation only at launch); name detection failures (fuzzy matching + Deepgram keyterm prompting + check interim results); interim result mishandling (three-tier strategy: interims for name only, finals for accumulation, speech_final for processing).

## Implications for Roadmap

Based on combined research, the implementation splits into 5 phases with clear dependency chains. All four research files independently converged on this structure.

### Phase 1: Deepgram Streaming Infrastructure
**Rationale:** Everything depends on working, reliable STT. This phase builds and validates the audio pipeline before adding any intelligence. The five critical pitfalls (VAD architecture, audio format, disconnection, cost, echo) must all be resolved here.
**Delivers:** Continuous Deepgram streaming with proper lifecycle management (active/idle/sleep). System transcribes speech reliably but makes no autonomous decisions. Existing PTT mode still works unchanged.
**Addresses:** TS-1 (WebSocket), TS-2 (transcript accumulation), TS-3 (cost optimization), TS-7 (lifecycle), TS-8 (echo suppression), TS-10 (resource management)
**Avoids:** Pitfall 1 (VAD architecture), Pitfall 2 (audio format), Pitfall 3 (disconnection), Pitfall 4 (cost runaway), Pitfall 5 (echo loop), Pitfall 6 (noisy environments)
**Key decision:** VAD role is connection-lifecycle (active/idle/sleep), NOT per-chunk audio filtering. This contradicts the original plan and must be settled before writing code.

### Phase 2: Decision Engine + Name Activation
**Rationale:** With reliable transcripts flowing, add the intelligence layer. Name activation provides a deterministic path while the decision model is being tuned. Initially conservative: high confidence threshold, name-only activation.
**Delivers:** Ollama integration with Llama 3.1 8B, structured JSON classification on each `speech_final`, name detection via fuzzy matching on both interim and final results, transcript buffer context window feeding decisions.
**Addresses:** TS-5 (decision engine), TS-6 (name activation), TS-4 (transcript buffer)
**Avoids:** Pitfall 7 (false positives -- name-only at launch), Pitfall 8 (name detection failures -- fuzzy matching + keyterm prompting)
**Uses:** Ollama + Llama 3.1 8B (STACK.md), `TranscriptBuffer.get_context()`, Deepgram `keyterm` parameter for name boosting

### Phase 3: Response Backend + Full Pipeline Integration
**Rationale:** Decision engine can now say "respond" -- wire the actual response delivery. Full end-to-end loop: STT -> Decision -> Response -> TTS -> Playback.
**Delivers:** Ollama quick responses (~700ms end-to-end), Claude CLI for complex queries, backend routing logic, Whisper fallback for degraded network.
**Addresses:** TS-9 (response backend)
**Avoids:** Pitfall 10 (latency spikes -- Whisper fallback), Pitfall 14 (state management -- pluggable STT interface)

### Phase 4: Proactive Participation
**Rationale:** Core loop proven reliable. Lower decision thresholds to enable AI contributions without being addressed. This is the differentiating feature.
**Delivers:** Proactive participation with attention signals, interruptibility detection, conversation state machine with cooldown after ignored interjections.
**Addresses:** D-4 (proactive participation), D-5 (attention signals), D-6 (interruptibility)
**Avoids:** Pitfall 7 (false positives -- conversation state machine, 5-minute cooldown after unengaged proactive responses)

### Phase 5: Polish + Enrichment
**Rationale:** Core system working. Add observability, analytics, and experimental features.
**Delivers:** Real-time transcript display, word-level timing analysis, optional diarization, non-speech event detection, library growth curator.
**Addresses:** D-1 (display), D-2 (timing), D-3 (diarization), D-7 (non-speech), D-8 (curator)

### Phase Ordering Rationale

- **Phase 1 first** because every other phase depends on working Deepgram streaming. The VAD gating architecture decision cascades through the entire system. Getting this wrong means reworking the audio pipeline.
- **Phase 2 before Phase 3** because the decision engine defines routing logic. Response backends cannot be wired without knowing what triggered them and what kind of response is needed.
- **Phase 3 before Phase 4** because proactive participation requires proven response delivery. Lowering thresholds on an unreliable pipeline creates chaos.
- **Phase 4 before Phase 5** because proactive participation is the core differentiator that defines the product. Polish features are valuable but don't change what the product is.
- **Diarization deferred to Phase 5** despite being a single config parameter, because streaming diarization has historical reliability issues and false positives from TV/music are better solved by name-activation + conversation state machine first.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Must reconcile VAD gating architecture -- STACK/FEATURES research assumes per-chunk filtering, PITFALLS research found Deepgram recommends against it. The lifecycle-management approach (active/idle/sleep) needs validation during implementation. Also: PipeWire AEC effectiveness with cloud STT latency is unverified.
- **Phase 2:** Decision model prompt engineering is inherently experimental. Llama 3.1 8B's classification accuracy on realistic ambient conversation needs benchmarking before committing. Budget time for prompt tuning iterations.
- **Phase 4:** Proactive participation thresholds have no precedent data for this system. The CHI 2025 "Inner Thoughts" framework provides heuristics but they need calibration. Plan for 2-3 iteration cycles.

Phases with standard patterns (skip research-phase):
- **Phase 3:** Response backend routing is well-understood. Ollama and Claude CLI are battle-tested in the existing codebase. Architecture is fully specified.
- **Phase 5:** All features have clear implementations. Real-time display is a new data source for the existing overlay. Diarization is a single Deepgram parameter. Curator follows existing `learner.py` daemon pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Deepgram SDK verified on PyPI (v5.3.2, 2026-01-29), pricing confirmed by Deepgram staff on GitHub Discussion #1423, VRAM estimates from community benchmarks (MEDIUM sub-confidence -- not empirically verified on RTX 3070 with 8B model loaded) |
| Features | HIGH | Deepgram API docs verified, existing test suite validates transcript accumulation patterns (lines 132-178), latency numbers from official Deepgram benchmarks, billing model confirmed by staff |
| Architecture | HIGH | Full codebase inspection with exact line references, drop-in replacement design with minimal blast radius (only `_stt_stage` rewritten, stages 3-5 unchanged), thread/async model mapped precisely including SDK internal thread behavior |
| Pitfalls | HIGH | 19 pitfalls documented from Deepgram official docs + SDK issue tracker + community discussions + codebase analysis. VAD gating pitfall (#1) sourced directly from Deepgram team member recommendation. Echo cancellation path analyzed with network latency implications. |

**Overall confidence:** HIGH

### Gaps to Address

- **VAD gating vs continuous streaming reconciliation:** STACK and FEATURES research assume per-chunk VAD gating for cost savings. PITFALLS research found Deepgram's team recommends against this. The reconciled approach (lifecycle-level gating: active/idle/sleep) needs implementation validation during Phase 1. This is the most consequential gap.

- **Llama 3.1 8B VRAM under load:** The ~5GB estimate is from community benchmarks. With KV cache growth at `num_ctx=4096`, actual usage may approach the 8GB limit on the RTX 3070. Must empirically validate in Phase 2 setup. Fallback to 3B must remain viable.

- **Streaming diarization quality:** Deepgram has a historical GitHub issue (#108) where speaker IDs always returned 0 in streaming mode. May be resolved in Nova-3 but needs verification before relying on it (deferred to Phase 5).

- **Deepgram SDK sync client reconnection behavior:** Architecture recommends sync client to avoid async conflicts, but behavior under repeated reconnection (exponential backoff + buffered audio replay) needs empirical testing during Phase 1.

- **PipeWire AEC effectiveness with cloud STT:** Echo cancellation is configured but untested in the cloud STT path. Network latency widens the timing window for residual echo compared to local Whisper processing. Must validate during Phase 1 before connecting Deepgram output to the decision model.

- **Endpointing in noisy environments:** `speech_final` may never fire with background noise (HVAC, fan). The dual-trigger pattern (`speech_final` OR `UtteranceEnd` OR local timeout) is designed but untested. Must verify during Phase 1 integration testing.

## Sources

### Primary (HIGH confidence)
- [Deepgram Streaming API Reference](https://developers.deepgram.com/reference/speech-to-text/listen-streaming) -- parameters, response format, message types
- [Deepgram Endpointing + Interim Results](https://developers.deepgram.com/docs/understand-endpointing-interim-results) -- `is_final`/`speech_final` lifecycle, configuration
- [Deepgram Audio Keep Alive](https://developers.deepgram.com/docs/audio-keep-alive) -- KeepAlive format, 10s timeout, text frame requirement
- [Deepgram Connection Recovery](https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio) -- reconnection, audio buffering, timestamp realignment
- [Deepgram Billing Discussion #1423](https://github.com/orgs/deepgram/discussions/1423) -- per-audio-second billing confirmed by staff, KeepAlive costs $0
- [Deepgram VAD Discussion #1216](https://github.com/orgs/deepgram/discussions/1216) -- team member recommends against pre-filtering silence
- [Deepgram Pricing](https://deepgram.com/pricing) -- $0.0077/min Nova-3 streaming, $200 free credit no expiry
- [Deepgram Python SDK (PyPI)](https://pypi.org/project/deepgram-sdk/) -- v5.3.2 stable, v6.0.0-rc.2 pre-release
- [Deepgram Nova-3 Benchmarks](https://deepgram.com/learn/speech-to-text-benchmarks) -- WER ~6.84% streaming
- Codebase: `live_session.py`, `continuous_stt.py`, `transcript_buffer.py`, `test_live_session.py`, `pipeline_frames.py`, `event_bus.py`

### Secondary (MEDIUM confidence)
- [Ollama VRAM Requirements](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) -- 8B Q4 ~5-6GB community benchmarks
- [Llama 3.1 8B vs 3.2 3B Comparison](https://medium.com/@marketing_novita.ai/llama-3-1-8b-vs-llama-3-2-3b) -- MMLU benchmark comparison
- [Ollama Performance on 8GB GPUs](https://aimuse.blog/article/2025/06/08/ollama-performance-tuning-on-8gb-gpus) -- KV cache growth, 7.6GB cliff on RTX 3070
- [Deepgram SDK Issue #493](https://github.com/deepgram/deepgram-python-sdk/issues/493) -- API key passing regression, still present in v5.3.2
- [Deepgram SDK Issue #279](https://github.com/deepgram/deepgram-python-sdk/issues/279) -- SDK 3.1.1 broke WebSocket connections
- [Twilio Voice Agent Latency Guide](https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents) -- STT 100-300ms, LLM 375ms, TTS 100ms typical
- [Tavus Turn-Taking Guide](https://www.tavus.io/post/ai-turn-taking) -- 200ms natural response gap

### Tertiary (LOW confidence)
- Streaming diarization quality (historical issue #108, may be resolved in Nova-3) -- needs empirical validation
- Llama 3.1 8B VRAM with KV cache at `num_ctx=4096` on RTX 3070 -- community estimates only, needs hardware testing
- Deepgram SDK sync client behavior under repeated reconnection -- undocumented edge cases possible

---
*Research completed: 2026-02-22*
*Supersedes: Previous SUMMARY.md (2026-02-21, local Whisper-based architecture)*
*Ready for roadmap: yes*
