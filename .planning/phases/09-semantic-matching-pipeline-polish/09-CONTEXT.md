# Phase 9: Semantic Matching + Pipeline Polish - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Classification handles paraphrased and ambiguous inputs via semantic fallback (model2vec). Quick response clips integrate cleanly with barge-in and LLM playback transitions through a new stream composer / cadence manager. Sentence-segmented TTS output plays with natural pacing.

Library growth and learning are Phase 10. Non-speech awareness is Phase 11.

</domain>

<decisions>
## Implementation Decisions

### Trivial input handling
- Short affirmations ("yes", "ok", "mhm", "yeah sure", "okay cool") are treated as trivial — no filler clip plays, natural silence instead
- Trivial inputs are still sent to LLM — only filler clip is suppressed
- Trivial detection is context-dependent: if Claude just asked a question, the next utterance is treated as a real answer regardless of length
- Subtle visual-only indicator (brief status flash) when trivial input detected — no audio feedback
- Initial rules ship in Phase 9; Phase 10 curator refines trivial detection over sessions using learning loop
- Boundary: anything with a verb or directive is real input, pure confirmational phrases are trivial
- Researcher should document conversational backchannel patterns to inform initial rule design

### Barge-in during clips
- Quick response clips (short, <1-2s): let clip finish playing, then process new input — they're too short to interrupt
- LLM TTS output: sentence-segmented, barge-in stops at sentence boundary (finishes current sentence, then stops)
- Remaining unsaid sentences are held in memory — LLM can decide to resume, modify, or discard based on what the user said
- Feels like the AI pauses to listen, not like it was cut off

### Clip-to-LLM transition — Stream Composer
- **New subprocess: stream composer / cadence manager** — separate from classifier daemon
- Manages a **unified sentence queue**: quick response clip is sentence 1, LLM TTS sentences follow as one continuous stream
- Barge-in works the same way across all segments in the queue
- **Natural pause between segments** — variable duration, chosen by the cadence manager (pauses are explicit elements in the stream)
- **Same voice continuity** — clip and LLM response should feel like one person speaking
- **Text-level sentence splitting** — LLM text response split into sentences first, TTS generated per sentence (enables streaming — first sentence plays while later ones generate)
- Cadence manager decides buffering/playback timing — no hard-coded rules
- Cadence manager has awareness of recent conversation context (last few turns) for pacing decisions
- **Full audio palette in Phase 9** — cadence manager can insert silence pauses, thinking sounds, breath-like pauses, subtle tonal cues
- Non-speech audio elements sourced from **pre-recorded clips** (shipped as library files)

### Semantic fallback behavior
- **Best guess wins** — go with the most likely category even at moderate confidence. The system learns over time (core principle)
- Zero-confidence (completely novel input): fall back to generic acknowledgment — never silence
- When keyword matching and semantic matching disagree: **higher confidence wins** (not automatic keyword priority)
- **Log everything** — every classification decision logged: keyword match, semantic match, final choice, confidence scores. Feeds Phase 10 learning loop

### Claude's Discretion
- Exact model2vec confidence thresholds (tune based on testing)
- Specific trivial input word list and boundary rules
- Stream composer internal architecture and buffering strategy
- Pre-recorded non-speech clip selection and variety
- Sentence splitting algorithm details

</decisions>

<specifics>
## Specific Ideas

- AI speech should "build on itself, as if it's speaking carefully, with poise" — sentence-segmented delivery, not a wall of audio
- Pauses are first-class elements in the audio stream, chosen by the AI subprocess — not afterthoughts
- The system should learn from itself over time — this is a core design principle that pervades all decisions
- Cadence manager is context-aware: rapid back-and-forth gets shorter pauses, deep explanations get longer ones

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-semantic-matching-pipeline-polish*
*Context gathered: 2026-02-18*
