# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.1 Voice UX Polish — Phase 4: Filler System Overhaul

## Current Position

Milestone: v1.1 Voice UX Polish
Phase: 4 of 6 (Filler System Overhaul)
Plan: 1 of 2
Status: In progress
Last activity: 2026-02-17 — Completed 04-01-PLAN.md (clip factory daemon + seeded pool)

Progress: [███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10% (1/2 Phase 4 plans complete)

## Performance Metrics

**v1.0 Velocity:**
- Total plans completed: 5
- Average duration: ~10 minutes
- Total execution time: ~48 minutes

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2/2 | ~42min | ~21min |
| 02 | 1/1 | ~3min | ~3min |
| 03 | 2/2 | ~3min | ~3min |
| 04 | 1/2 | ~2.5min | ~2.5min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Carried forward from v1.0:

- Pipeline architecture: 5-stage asyncio (audio_capture → STT → LLM → TTS → playback)
- Claude CLI via stream-json protocol, not OpenAI Realtime API
- Local Whisper STT + Piper TTS (zero cloud latency dependency)
- Overlay communicates with session via signal files
- Filler system needs overhaul: smart fillers conflict with LLM response
- Barge-in: gate STT instead of mic mute (mic must stay live for VAD)
- Clip factory: single nonverbal/ category, synchronous subprocess, numpy quality evaluation
- Non-verbal clip quality gate: duration 0.2-2.0s, RMS > 300, clipping < 1%, silence < 70%

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-17
Stopped at: Completed 04-01-PLAN.md — clip factory and seeded pool ready
Resume file: .planning/phases/04-filler-system-overhaul/04-02-PLAN.md
