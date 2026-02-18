# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.1 Voice UX Polish — Phase 6 in progress (Polish & Verification)

## Current Position

Milestone: v1.1 Voice UX Polish
Phase: 6 of 6 (Polish & Verification)
Plan: 1 of 3
Status: In progress
Last activity: 2026-02-18 — Completed 06-01-PLAN.md (STT filtering + rejection flash)

Progress: [██████████████████████░░░░░░░░░] 78% (7/9 v1.1 plans complete)

## Performance Metrics

**v1.0 Velocity:**
- Total plans completed: 7
- Average duration: ~7.5 minutes
- Total execution time: ~52 minutes

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2/2 | ~42min | ~21min |
| 02 | 1/1 | ~3min | ~3min |
| 03 | 2/2 | ~3min | ~3min |
| 04 | 2/2 | ~7.5min | ~3.75min |
| 05 | 2/2 | ~5.5min | ~2.75min |
| 06 | 1/3 | ~1min | ~1min |

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

Phase 6 additions:
- 3-layer Whisper segment filtering: no_speech_prob >= 0.6, avg_logprob < -1.0, compression_ratio > 2.4
- stt_rejected as transient overlay flash (300ms dot dim), not a state transition

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18T12:48:40Z
Stopped at: Completed 06-01-PLAN.md — STT filtering + rejection flash
Resume file: None
