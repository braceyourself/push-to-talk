# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.2 Adaptive Quick Responses -- Phase 8 in progress

## Current Position

Milestone: v1.2 Adaptive Quick Responses
Phase: 8 of 11 (Core Classification + Response Library)
Plan: 1 of 3 complete
Status: In progress
Last activity: 2026-02-19 -- Completed 08-01-PLAN.md

Progress: [##                            ] 8% (1/12 plans estimated)

## Performance Metrics

**Velocity:**
- Total plans completed: 1 (this milestone)
- Average duration: 4min
- Total execution time: 4min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 8 | 1/3 | 4min | 4min |

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

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 08-01-PLAN.md
Resume file: None
