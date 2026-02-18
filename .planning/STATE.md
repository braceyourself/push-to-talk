# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.2 Adaptive Quick Responses -- Phase 8 ready to plan

## Current Position

Milestone: v1.2 Adaptive Quick Responses
Phase: 8 of 11 (Core Classification + Response Library)
Plan: Not started
Status: Ready to plan
Last activity: 2026-02-18 -- Roadmap created for v1.2

Progress: [                              ] 0% (0/4 phases)

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (this milestone)
- Average duration: -
- Total execution time: -

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

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

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18
Stopped at: Roadmap created for v1.2, ready to plan Phase 8
Resume file: None
