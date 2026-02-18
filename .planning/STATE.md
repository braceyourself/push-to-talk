# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** v1.1 Voice UX Polish — Complete

## Current Position

Milestone: v1.1 Voice UX Polish
Phase: 7 of 7 (Nonverbal System Cleanup)
Plan: 1 of 1
Status: Milestone complete
Last activity: 2026-02-18 — Completed 07-01-PLAN.md (nonverbal system cleanup)

Progress: [██████████████████████████████] 100% (10/10 v1.1 plans complete)

## Performance Metrics

**v1.1 Velocity:**
- Total plans completed: 10
- Average duration: ~6 minutes
- Total execution time: ~63 minutes

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2/2 | ~42min | ~21min |
| 02 | 1/1 | ~3min | ~3min |
| 03 | 2/2 | ~3min | ~3min |
| 04 | 2/2 | ~7.5min | ~3.75min |
| 05 | 2/2 | ~5.5min | ~2.75min |
| 06 | 3/3 | ~20min | ~6.5min |
| 07 | 1/1 | ~3min | ~3min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Carried forward from v1.0:

- Pipeline architecture: 5-stage asyncio (audio_capture -> STT -> LLM -> TTS -> playback)
- Claude CLI via stream-json protocol, not OpenAI Realtime API
- Local Whisper STT + Piper TTS (zero cloud latency dependency)
- Overlay communicates with session via signal files
- Filler system overhauled: acknowledgment phrase clips only, no more Ollama/LLM smart fillers
- Barge-in: gate STT instead of mic mute (mic must stay live for VAD)
- Clip factory: acknowledgment-only, synchronous subprocess, numpy quality evaluation
- Acknowledgment clip quality gate: duration 0.3-4.0s, RMS > 200, clipping < 1%, silence < 50%

Phase 5 additions:
- VAD runs inline in STT stage (Branch 2) rather than separate monitor stage
- 6 consecutive VAD-positive chunks (~0.5s) threshold for barge-in trigger
- 1.5s cooldown after barge-in prevents rapid-fire re-triggers
- Trailing filler: 150ms acknowledgment clip with 0.8->0.0 linear fade for natural cutoff
- Three-branch audio processing in _stt_stage: muted / gated (VAD) / normal
- Sentence tracking at 3 flush sites for spoken/unspoken annotation on barge-in
- Annotation prepended to next user message (not separate turn) so AI knows it was cut off
- Post-barge-in silence threshold 0.4s (vs 0.8s normal) for faster response after interruption
- No separate overlay state for barge-in -- status transition is the visual acknowledgment

Phase 6 additions:
- 3-layer Whisper segment filtering: no_speech_prob >= 0.6, avg_logprob < -1.0, compression_ratio > 2.4
- stt_rejected as transient overlay flash (300ms dot dim), not a state transition
- Clip factory refactored to generic _top_up helper supporting multiple pool categories
- Acknowledgment clip pool: 10-15 verbal phrases, relaxed quality thresholds (0.3-4.0s, RMS > 200)
- Nonverbal filler clips dropped — Piper TTS can't produce natural interjections
- Acknowledgment clips used for all filler roles (pre-response + pre-tool)
- Gated pre-tool acknowledgment: 300ms asyncio gate, skips if tool completes fast
- TOOL_INTENT_MAP maps MCP tool names to human-readable intents (with MCP prefix stripping)
- _set_status accepts optional metadata dict, serialized as JSON through existing callback chain
- Overlay parses JSON status for tool intent display, coalesces consecutive tool_use history entries

Phase 7 additions:
- Barge-in trailing filler fixed: uses acknowledgment clips instead of broken nonverbal lookup
- All nonverbal code removed from clip_factory.py (constants, functions, evaluate_clip else-branch)
- Nonverbal WAV files and pool.json cleaned from disk and git
- FILL-02 requirement updated to describe acknowledgment phrase behavior

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18T19:38:00Z
Stopped at: Completed 07-01-PLAN.md — v1.1 milestone complete
Resume file: None
