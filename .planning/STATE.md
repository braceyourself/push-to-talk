# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Natural, low-friction voice conversation with Claude that feels like talking to a person
**Current focus:** Planning v1.2 Adaptive Quick Responses

## Current Position

Milestone: v1.2 Adaptive Quick Responses
Phase: Not started
Plan: Not started
Status: Ready to plan
Last activity: 2026-02-18 — v1.1 milestone complete

Progress: [                              ] 0% (milestone not yet planned)

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carried forward:
- Pipeline architecture: 5-stage asyncio (audio_capture -> STT -> LLM -> TTS -> playback)
- Claude CLI via stream-json protocol
- Local Whisper STT + Piper TTS
- Acknowledgment phrase fillers (nonverbal clips don't work with Piper)
- Barge-in via STT gating + VAD
- 3-layer Whisper segment filtering
- Tool intent overlay with JSON status protocol

v1.2 direction:
- Replace random filler selection with AI-driven quick response library
- System should understand context and choose appropriate response
- Non-speech events (coughs, sighs) should get contextual responses
- Library grows and prunes across sessions

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18
Stopped at: v1.1 milestone archived, v1.2 seeded
Resume file: None
