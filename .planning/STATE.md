# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.1 Voice UX Polish — Phase 5 complete (Barge-in)

## Current Position

Milestone: v1.1 Voice UX Polish
Phase: 5 of 6 (Barge-in)
Plan: 2 of 2
Status: Phase complete
Last activity: 2026-02-17 — Completed 05-02-PLAN.md (barge-in intelligence layer)

Progress: [████████████████████░░░░░░░░░░░] 67% (6/9 v1.1 plans complete)

## Performance Metrics

**v1.0 Velocity:**
- Total plans completed: 6
- Average duration: ~9 minutes
- Total execution time: ~51 minutes

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2/2 | ~42min | ~21min |
| 02 | 1/1 | ~3min | ~3min |
| 03 | 2/2 | ~3min | ~3min |
| 04 | 2/2 | ~7.5min | ~3.75min |
| 05 | 2/2 | ~5.5min | ~2.75min |

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
- Sentence tracking at 3 flush sites for spoken/unspoken annotation on barge-in
- Annotation prepended to next user message (not separate turn) so AI knows it was cut off
- Post-barge-in silence threshold 0.4s (vs 0.8s normal) for faster response after interruption
- No separate overlay state for barge-in -- status transition is the visual acknowledgment

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-17T23:27:15Z
Stopped at: Completed 05-02-PLAN.md — barge-in intelligence layer (phase 5 complete)
Resume file: None
