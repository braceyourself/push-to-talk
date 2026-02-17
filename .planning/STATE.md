# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.1 Voice UX Polish — Phase 5 in progress (Barge-in)

## Current Position

Milestone: v1.1 Voice UX Polish
Phase: 5 of 6 (Barge-in)
Plan: 1 of 2
Status: In progress
Last activity: 2026-02-17 — Completed 05-01-PLAN.md (core barge-in mechanism)

Progress: [█████████████████░░░░░░░░░░░░░░] 56% (5/9 v1.1 plans complete)

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
| 04 | 2/2 | ~7.5min | ~3.75min |
| 05 | 1/2 | ~3min | ~3min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Carried forward from v1.0:

- Pipeline architecture: 5-stage asyncio (audio_capture -> STT -> LLM -> TTS -> playback)
- Claude CLI via stream-json protocol, not OpenAI Realtime API
- Local Whisper STT + Piper TTS (zero cloud latency dependency)
- Overlay communicates with session via signal files
- Filler system overhauled: non-verbal clips only, no more Ollama/LLM smart fillers
- Barge-in: gate STT instead of mic mute (mic must stay live for VAD)
- Clip factory: single nonverbal/ category, synchronous subprocess, numpy quality evaluation
- Non-verbal clip quality gate: duration 0.2-2.0s, RMS > 300, clipping < 1%, silence < 70%

Phase 5 additions:
- VAD runs inline in STT stage (Branch 2) rather than separate monitor stage
- 6 consecutive VAD-positive chunks (~0.5s) threshold for barge-in trigger
- 1.5s cooldown after barge-in prevents rapid-fire re-triggers
- Trailing filler: 150ms nonverbal clip with 0.8->0.0 linear fade for natural cutoff
- Three-branch audio processing in _stt_stage: muted / gated (VAD) / normal

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-17T23:21:54Z
Stopped at: Completed 05-01-PLAN.md — core barge-in mechanism implemented
Resume file: None
