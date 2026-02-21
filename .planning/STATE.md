# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** An always-present AI that listens, understands context, and contributes when it has something useful to add
**Current focus:** v2.0 Always-On Observer — Defining requirements

## Current Position

Milestone: v2.0 Always-On Observer
Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-21 — Milestone v2.0 started

## Performance Metrics

**Velocity (v1.2):**
- Total plans completed: 5
- Average duration: 3.6min
- Total execution time: 18min

*Reset for v2.0 after first plan completes*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carried forward from v1.1:
- Pipeline architecture: 5-stage asyncio (audio_capture -> STT -> LLM -> TTS -> playback)
- Local Whisper STT + Piper TTS
- Acknowledgment phrase fillers (nonverbal clips don't work with Piper)
- Barge-in via STT gating + VAD

Carried forward from v1.2:
- Heuristic pattern matching first (<1ms), model2vec semantic fallback second (5-10ms)
- JSON-based response library (50-200 entries)
- StreamComposer for unified audio queue with pre-buffering, cadence control, barge-in
- Configurable idle timeout (0 = always-on)

v2.0 milestone decisions:
- Decouple inputs from LLM processing (independent input stream)
- Ollama + Llama 3.2 3B for monitoring layer (free, local, ~200ms)
- Configurable response backend (Claude CLI / Ollama), auto-selected
- PTT replaced entirely by always-on listening
- Name-based interruption ("hey Russel") replaces PTT-based barge-in trigger
- Proactive AI participation (more aggressive — joins even casual conversations)
- Library growth + non-speech awareness folded from v1.2 phases 10-11

### Pending Todos

- Assistant tool creation feature (user idea captured during Phase 8 execution)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Adapt Clawdbot personality system (Russel identity, soul values, user prefs) | 2026-02-20 | 0f70b4d | [001-adapt-clawdbot-personality-system](./quick/001-adapt-clawdbot-personality-system/) |
| 002 | Configurable idle timeout with always-on default (0 = never disconnect) | 2026-02-20 | d5450f8 | [002-always-on-listening-smart-responding](./quick/002-always-on-listening-smart-responding/) |

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-21
Stopped at: Milestone v2.0 initialized, defining requirements
Resume file: None
