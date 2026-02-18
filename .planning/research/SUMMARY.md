# Research Summary: Adaptive Quick Responses (v1.2)

**Project:** Push-to-Talk Voice Assistant - v1.2 Adaptive Quick Response Library
**Domain:** Context-aware quick response system for existing voice AI assistant
**Researched:** 2026-02-18
**Confidence:** HIGH

## Executive Summary

The v1.2 milestone replaces the current random acknowledgment system with context-aware quick responses. Research confirms this is a surgically scoped enhancement to the existing 5-stage pipeline — not a major architectural change. The integration point is narrow: modify the `_filler_manager()` method in `live_session.py` to classify user input and select appropriate responses from a categorized library.

The recommended approach uses **pattern-matching classification** (not ML) for sub-millisecond response selection, **model2vec** for semantic similarity when exact matches fail, and a **JSON-based response library** that grows organically via a post-session curator daemon (following the same pattern as `learner.py`). The existing Whisper STT, Piper TTS, and clip factory infrastructure provide everything needed — only two small additions are required: model2vec (8MB, numpy-only dependency) for semantic matching, and a curator daemon for library growth.

The critical risk is **timing**: classification must complete in under 50ms to fit within the existing 500ms filler gate. Using keyword/pattern matching for primary classification (1ms) with model2vec as fallback (5-10ms) keeps total latency under 20ms. Secondary risks include audio quality discontinuity between pre-generated clips and live TTS (solved by parameter normalization), and classification accuracy degradation with too many categories (solved by starting with 5-7 broad categories and expanding only with data).

## Key Findings

### Recommended Stack

The existing stack (Piper TTS, Whisper STT, asyncio pipeline) already provides nearly everything needed. Only two additions are recommended:

**Core technologies:**
- **model2vec** (potion-base-8M, 8MB): Ultra-fast semantic similarity matching (20,000+ sentences/sec on CPU) for when keyword patterns don't match — only dependency is numpy (already installed)
- **Python sqlite3** (stdlib): Response library metadata storage — considered but JSON is actually better for this scale (50-200 entries); JSON keeps backward compatibility with existing `ack_pool.json` pattern
- **Piper TTS** (existing): Pre-generate response clips using the same `clip_factory.py` infrastructure — identical pattern to existing acknowledgment pool

**What NOT to add:**
- **No PyTorch/sentence-transformers**: model2vec uses numpy only; sentence-transformers would add 2GB dependency for marginal accuracy improvement
- **No cloud APIs**: Classification must happen in <50ms; any HTTP call exceeds the budget
- **No separate audio classifier**: Whisper already produces non-speech detection signals via `no_speech_prob`, `avg_logprob`, and bracketed annotations like `[Laughter]`

### Expected Features

**Must have (table stakes):**
- **Intent-based category routing** — classify user input (question/command/conversational/emotional) and pick fillers from matching category; without this, the system is worse than random because users notice wrong responses
- **Multi-category clip pools** — separate pools for different situations (task-oriented "on it", conversational "hmm", social "hey"); LOW complexity, extends existing clip_factory
- **Classification speed <200ms** — must fit within the 500ms filler gate; this constraint rules out API calls and large models
- **Graceful fallback to random** — if classification fails or takes too long, use existing acknowledgment pool rather than silence
- **Clip variety within categories** — 5-8 clips per category minimum to avoid repetition fatigue

**Should have (differentiators):**
- **Non-speech event responses** — detect coughs/sighs/laughter via STT rejection metadata and respond playfully ("excuse you", sympathetic acknowledgment); leverages existing multi-layer filtering
- **Emotional tone matching** — detect excitement/frustration in text and match filler energy (exclamation marks, sentiment keywords)
- **Conversational vs. task mode detection** — "What's your name?" gets "hmm" not "checking now"; falls naturally out of category taxonomy
- **Silence as valid response** — for very short inputs ("yes", "ok"), no filler at all; explicit classification of "no-filler-needed" inputs

**Defer (v2+):**
- **Dynamic TTS fillers** — on-the-fly generation with contextual text ("hmm, let me think about [topic]"); Piper latency (200-400ms) may blow timing budget
- **Learned response preferences** — track which fillers get interrupted and down-weight them; needs session logging infrastructure
- **Multi-turn context** — track conversation history to pick fillers based on flow; over-engineering for a 1-second audio clip

### Architecture Approach

The architecture adds three new components while modifying only two existing ones. The key insight: the integration point is surgically narrow — `_filler_manager()` is the only method in the hot path that changes. Library growth happens entirely outside the pipeline via a post-session curator daemon.

**Major components:**

1. **InputClassifier** (new, in-process) — Pattern matching + optional model2vec fallback for semantic similarity; produces ClassifiedInput with category/confidence in <50ms; purely synchronous, no async complexity

2. **ResponseLibrary** (new, in-process) — JSON-based lookup table mapping categories to pre-loaded PCM clips; O(1) lookup via dict index, entire library loads at startup (same pattern as `_load_filler_clips()`); tracks usage for post-session analysis

3. **LibraryCurator** (new, daemon) — Post-session subprocess (like `learner.py`) that reads conversation log, identifies response gaps via Claude, generates new clips via Piper, and prunes low-quality entries; writes to `library.json`, next session picks up changes

4. **Modified: `_filler_manager()`** — Classify input → lookup response → wait gate → play clip or fallback; classification happens before gate (consumes negligible time from 500ms budget); falls back to existing random acknowledgment if no match

5. **Modified: STT stage** — Forward non-speech events (currently rejected silently) as new `NON_SPEECH` frame type carrying rejection metadata (`no_speech_prob`, `avg_logprob`) for playful responses

**Data flow:**
- User speaks → Whisper transcribes → Classify input → Lookup clip → Wait 500ms gate → Play if LLM hasn't responded → Cancel on first LLM text
- Non-speech (cough/sigh) → Whisper rejects → Emit NON_SPEECH frame → Classify event → Play appropriate response (no LLM call)
- Session ends → Curator analyzes gaps → Generates new clips via Piper → Updates library.json

### Critical Pitfalls

1. **Classification latency exceeds filler window** — If classification takes 200ms+, the filler arrives too late or collides with LLM response. Industry benchmarks: 200-300ms is the human-perceivable response window. **Mitigation:** Use pattern matching (<1ms) with model2vec fallback (5-10ms). Never use LLM or API calls for classification.

2. **Quick response collides with LLM's first audio** — Clip starts playing but LLM responds faster than expected; tail of clip overlaps with TTS. **Mitigation:** Two-layer cancellation: source-side stops pushing new chunks (existing), sink-side drains pending FILLER frames from queue when cancel fires (add to playback stage).

3. **Over-classifying intent creates brittle taxonomy** — 30+ categories degrades accuracy; 60% accuracy means 40% of responses feel wrong (worse than generic). **Mitigation:** Start with 5-7 broad categories maximum. Use "acknowledgment" as fallback for low-confidence classifications. Add categories only with data-driven evidence.

4. **Non-speech detection misclassifies speech** — Cough detector triggers on breathy speech, laugh detector on mid-sentence chuckle. Whisper's `no_speech_prob` has 40.3% hallucination rate on non-speech audio. **Mitigation:** Defer non-speech detection to Phase 3+. Require very high confidence (≥0.85) when implemented. Make responses safe even if detection is wrong.

5. **Uncanny valley — contextually correct but emotionally wrong** — Classifier picks "I'm sorry to hear that" for sad news but Piper TTS sounds robotic, creating intelligence/voice quality mismatch. **Mitigation:** Avoid emotionally loaded categories. Stick to neutral/functional (acknowledgments, greetings, reactions). Normalize Piper parameters between clips and live TTS.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Core Classification + Playback
**Rationale:** Establishes the foundation — classification and response library — before adding complexity. Pattern-matching classification is simple enough to build and test independently before integration. This phase addresses the core user complaint: task-oriented fillers playing for non-task inputs.

**Delivers:**
- InputClassifier with pattern/keyword matching (<1ms classification)
- ResponseLibrary JSON schema and in-memory index
- Modified `_filler_manager()` using classifier + library
- Seed library with 5 categories: task, question, conversational, social, acknowledgment
- 5-8 clips per category (30-40 clips total)
- Graceful fallback to existing acknowledgment pool

**Addresses features:**
- Intent-based category routing (must-have)
- Multi-category clip pools (must-have)
- Classification speed <200ms (must-have)
- Graceful fallback (must-have)

**Avoids pitfalls:**
- #1 (classification latency) — pattern matching guarantees <1ms
- #3 (over-classification) — only 5 categories initially
- #7 (cold start) — seed library ships with repo

**Research flags:** Standard pattern matching and JSON storage — no research needed.

### Phase 2: Semantic Matching + Barge-in Polish
**Rationale:** Adds model2vec for semantic similarity when keyword patterns don't match. This improves classification accuracy without adding latency. Barge-in integration addresses edge cases where quick responses are interrupted.

**Delivers:**
- model2vec integration as classification fallback
- Semantic similarity matching (embed user text, cosine similarity to category exemplars)
- Barge-in tracking for quick response clips (don't include in LLM annotation)
- Custom barge-in behavior: skip trailing acknowledgment when quick response is interrupted
- Silence-as-response for very short inputs (<3 words)

**Addresses features:**
- Emotional tone matching (should-have) — semantic matching catches sentiment
- Conversational vs. task mode (should-have) — better discrimination
- Silence as valid response (should-have)

**Avoids pitfalls:**
- #9 (barge-in confusion) — track quick response state separately from LLM state
- #13 (unnecessary quick responses) — suppress for short utterances

**Research flags:** model2vec API is verified, integration is straightforward — no research needed.

### Phase 3: Library Growth + Pruning
**Rationale:** Once the system works with a static library, add organic growth via the curator daemon. This follows the exact pattern as `learner.py`, reducing implementation risk.

**Delivers:**
- LibraryCurator daemon (subprocess spawned at session start)
- Post-session gap analysis via `claude -p`
- Automated clip generation via Piper (reuses `clip_factory.py` functions)
- Quality pruning (remove frequently-interrupted clips)
- Usage logging (which clips played, barge-in events)

**Addresses features:**
- Learned response preferences (should-have)
- Library expansion based on actual usage patterns

**Avoids pitfalls:**
- #6 (library bloat) — cap at 7 clips per category, prune low-effectiveness entries
- #11 (personality drift) — include personality prompt in generation context

**Research flags:** Standard daemon pattern from `learner.py` — no research needed.

### Phase 4: Non-Speech Event Detection (Optional)
**Rationale:** Defer to last because of high false-positive risk. Only implement if Phases 1-3 demonstrate that text-based classification leaves coverage gaps.

**Delivers:**
- NON_SPEECH frame type in pipeline
- STT stage forwarding of rejected segments with metadata
- Non-speech event classification (cough/sigh/laugh)
- Playful responses for detected events ("excuse you", empathetic acknowledgment)

**Addresses features:**
- Non-speech event responses (should-have)

**Avoids pitfalls:**
- #4 (misclassification) — require high confidence, make responses safe

**Research flags:** May need `/gsd:research-phase` for non-speech classification accuracy tuning if Whisper's existing signals prove insufficient.

### Phase Ordering Rationale

- **Phase 1 first** because classification and library are dependencies for everything else; they can be built and tested independently before pipeline integration
- **Phase 2 before Phase 3** because semantic matching improves classification quality, which helps the curator learn what "good" responses look like
- **Phase 3 before Phase 4** because library growth establishes the curator infrastructure; non-speech detection can reuse this for generating event-specific clips
- **Phase 4 last** (and optional) because non-speech detection has the highest false-positive risk and may not be needed if text-based classification covers 95%+ of cases

**Grouping by architectural layer:**
- Phase 1: Data layer (library) + decision layer (classifier)
- Phase 2: Decision quality (semantic matching) + interaction layer (barge-in)
- Phase 3: Growth layer (curator)
- Phase 4: Input layer (non-speech events)

This ordering minimizes rework — each phase builds on verified infrastructure from previous phases.

### Research Flags

**Phases with standard patterns (skip research-phase):**
- **Phase 1:** Pattern matching is simple; JSON storage follows `ack_pool.json` pattern; pipeline integration point is well-understood from codebase reading
- **Phase 2:** model2vec API is verified; barge-in logic already exists (just needs extension)
- **Phase 3:** Curator daemon follows `learner.py` pattern exactly; clip generation reuses `clip_factory.py` functions

**Phases likely needing deeper research:**
- **Phase 4:** Non-speech detection accuracy may need experimentation; if Whisper's existing signals (`no_speech_prob`, bracketed annotations) prove insufficient, may need to research dedicated audio classifiers (Whisper-AT, SenseVoice) — but defer this research until Phase 4 is prioritized

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified via GitHub docs (model2vec), codebase reading (existing Piper/Whisper integration), stdlib availability (sqlite3/JSON); only 1 new external dependency (model2vec, 8MB, numpy-only) |
| Features | HIGH | Well-researched domain (backchannel linguistics, Google Duplex research, NVIDIA PersonaPlex, industry latency benchmarks); clear distinction between table stakes and differentiators |
| Architecture | HIGH | Complete codebase reading of `live_session.py`, `clip_factory.py`, `learner.py`; integration points verified in source code; proposed changes are surgical (only `_filler_manager()` in hot path) |
| Pitfalls | HIGH | Verified against academic research (arXiv papers on Whisper hallucination), industry analysis (AssemblyAI/Sierra latency benchmarks), and existing codebase patterns (timing constraints, audio pipeline) |

**Overall confidence:** HIGH

The research is based on complete source code reading (not API assumptions), verified third-party libraries (model2vec confirmed on GitHub/PyPI), and well-documented voice UX domain knowledge (Google Duplex research, backchannel linguistics). The only area with lower confidence is non-speech detection accuracy (40.3% hallucination rate from research), which is why Phase 4 is deferred and optional.

### Gaps to Address

- **Non-speech detection accuracy**: Whisper's `no_speech_prob` and bracketed annotations are inconsistent (trained on YouTube subtitles, not designed for this). The research identifies this as a known hard problem (7% error rate at best). **Handling:** Defer to Phase 4 (optional). If implemented, require very high confidence thresholds and make responses safe even if detection is wrong.

- **Model2vec inference latency on target hardware**: Benchmark claims 20,000+ sentences/sec on CPU, but this wasn't verified on the actual deployment hardware. **Handling:** Phase 2 includes a benchmark step before integration. If model2vec exceeds 20ms per classification, fall back to pattern matching only (still functional, just less accurate on paraphrased inputs).

- **Clip factory batch generation time**: Current `clip_factory.py` generates ~1 clip/sec via Piper. A full 7-category library with 7 clips each = 49 clips = ~50 seconds. **Handling:** Ship a seed library with 2-3 clips per category (pre-generated, committed to repo). Curator adds variety over time. System is functional from first launch.

- **Piper TTS emotional expressivity**: Research confirms Piper (en_US-lessac-medium) produces intelligible but not emotionally expressive speech. **Handling:** Avoid emotionally loaded categories (sympathy, excitement, humor) entirely. Stick to neutral/functional categories where flat prosody is acceptable.

## Sources

### Primary (HIGH confidence)
- **Codebase reading** (complete): `live_session.py` (pipeline architecture, filler system, STT gating, barge-in logic, tool-use flow), `clip_factory.py` (clip generation, quality evaluation, pool management), `learner.py` (daemon pattern), `pipeline_frames.py` (frame types), `ack_pool.json` (metadata schema)
- **[model2vec GitHub](https://github.com/MinishLab/model2vec)** — Verified API, model sizes, dependencies, inference speed (20,000+ sent/sec on CPU)
- **[model2vec PyPI](https://pypi.org/project/model2vec/)** — Latest version, release history
- **[minishlab/potion-base-8M on HuggingFace](https://huggingface.co/minishlab/potion-base-8M)** — Model specs: 8MB, 256 dimensions
- **[Python sqlite3 stdlib](https://docs.python.org/3.12/library/sqlite3.html)** — Verified FTS5 available on system
- **[OpenAI Whisper GitHub](https://github.com/openai/whisper)** — `no_speech_prob`, `avg_logprob`, `compression_ratio` segment fields
- **[Google Duplex Research Blog](https://research.google/blog/google-duplex-an-ai-system-for-accomplishing-real-world-tasks-over-the-phone/)** — Filler design, latency management, context-aware disfluencies

### Secondary (MEDIUM confidence)
- **[AssemblyAI: The 300ms Rule](https://www.assemblyai.com/blog/low-latency-voice-ai)** — Latency perception thresholds (200-300ms human-perceivable window)
- **[Sierra: Engineering Low-Latency Voice Agents](https://sierra.ai/blog/voice-latency)** — Filler audio as latency masking
- **[Retell AI: Backchanneling](https://www.retellai.com/blog/how-backchanneling-improves-user-experience-in-ai-powered-voice-agents)** — Context-aware acknowledgment selection
- **[NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/)** — Backchannel training on 7,303 real conversations
- **[Wikipedia: Backchannel (linguistics)](https://en.wikipedia.org/wiki/Backchannel_(linguistics))** — Linguistic framework for generic vs. specific backchannels
- **[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)** — Non-speech event detection (cough, laugh, sneeze)
- **[arXiv 2501.11378: Whisper ASR Hallucinations](https://arxiv.org/abs/2501.11378)** — 40.3% hallucination rate on non-speech audio
- **[Talkdesk: Voice Design Uncanny Valley](https://www.talkdesk.com/blog/voice-design/)** — Consistency between voice quality and intelligence
- **[Nonspeech7k Dataset](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/sil2.12233)** — Non-speech sound classification research

### Tertiary (LOW confidence)
- **[Intent Classification in <1ms](https://medium.com/@durgeshrathod.777/intent-classification-in-1ms-how-we-built-a-lightning-fast-classifier-with-embeddings-db76bfb6d964)** — Embedding-based fast classification (not verified, but architecture pattern is sound)
- **[Whisper-AT PyPI](https://pypi.org/project/whisper-at/)** — 152 weekly downloads (low adoption signal, not recommended but documented as alternative)

---
*Research completed: 2026-02-18*
*Ready for roadmap: yes*
