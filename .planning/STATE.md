# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.2 Adaptive Quick Responses -- Phase 9 in progress

## Current Position

Milestone: v1.2 Adaptive Quick Responses
Phase: 9 of 11 (Semantic Matching + Pipeline Polish)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-02-19 -- Completed 09-01-PLAN.md

Progress: [##########                    ] 33% (4/12 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 4 (this milestone)
- Average duration: 3.5min
- Total execution time: 14min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 8 | 3/3 | 10min | 3.3min |
| 9 | 1/3 | 4min | 4min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carried forward from v1.1:
- Pipeline architecture: 5-stage asyncio (audio_capture -> STT -> LLM -> TTS -> playback)
- Local Whisper STT + Piper TTS
- Acknowledgment phrase fillers (nonverbal clips don't work with Piper)
- Barge-in via STT gating + VAD

v1.2 research decisions:
- Heuristic pattern matching first (<1ms), model2vec semantic fallback second (5-10ms)
- JSON-based response library (not sqlite -- 50-200 entries, follows existing ack_pool.json pattern)
- 5-7 broad categories max (accuracy drops with 30+ categories)
- Non-speech detection deferred to Phase 11 (40% Whisper hallucination rate on non-speech)
- Curator daemon follows learner.py subprocess pattern
- Seed clips ship in repo, pre-generated

v1.2 execution decisions (08-01):
- Default classifier fallback is acknowledgment (not task) -- safest for any input
- Emotional words "nice"/"sick" require standalone context to avoid false positives
- Acknowledgment patterns use anchored regex to only match full-text acknowledgments
- Classifier daemon uses CLASSIFIER_READY stdout signal for readiness synchronization

v1.2 execution decisions (08-02):
- Piper defaults for seed clips (not randomized) to match live TTS voice quality
- Relaxed silence threshold (0.7) for short clips (<1s) to account for Piper padding
- Subcategory metadata as underscore-prefixed maps in seed_phrases.json

v1.2 execution decisions (08-03):
- Classification before 500ms gate (time absorbed into wait, no added latency)
- Confidence < 0.4 falls back to acknowledgment category
- Response library clips resampled 22050->24000Hz for playback compatibility

v1.2 execution decisions (09-01):
- Heuristic tiebreak: "could you"/"can you"/"would you" framing prefers task over question
- Confidence normalization: cosine>=0.6 maps to 0.8-0.9, 0.4-0.6 maps to 0.5-0.7, <0.4 maps to 0.2-0.4
- CLASSIFIER_READY before semantic model loads (background thread graceful degradation)
- TRIVIAL_PATTERNS as frozenset for O(1) lookup

### Pending Todos

- Assistant tool creation feature (user idea captured during Phase 8 execution)

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 09-01-PLAN.md
Resume file: None
