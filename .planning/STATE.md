# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** An always-present AI that listens, understands context, and contributes when it has something useful to add
**Current focus:** v2.0 Always-On Observer — Phase 12: Infrastructure + Safety Net

## Current Position

Milestone: v2.0 Always-On Observer
Phase: 12 of 16 (Infrastructure + Safety Net)
Plan: 2 of 3
Status: In progress
Last activity: 2026-02-21 — Completed 12-02-PLAN.md (TranscriptBuffer, hallucination filter)

Progress: [██░░░░░░░░] ~7% (2 of ~30 total plans)

## Performance Metrics

**Velocity (v2.0):**
- Total plans completed: 2
- Average duration: 8min
- Total execution time: 16min

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carried forward from v1.2:
- StreamComposer for unified audio queue with pre-buffering, cadence control, barge-in
- Heuristic pattern matching first (<1ms), model2vec semantic fallback second (5-10ms)
- Configurable idle timeout (0 = always-on)

v2.0 milestone decisions:
- Decouple inputs from LLM processing (independent input stream)
- Ollama + Llama 3.2 3B for monitoring layer (free, local, ~200ms)
- Configurable response backend (Claude CLI / Ollama), auto-selected
- Whisper distil-large-v3 for continuous STT (~1.5GB VRAM vs 3.5GB for large-v3)
- PipeWire echo cancellation as primary feedback loop prevention
- Start conservative (name-activation only), expand to proactive after validation

Phase 12-01 decisions:
- VRAM GO: Whisper+Ollama concurrent peak 5421MB (66%) leaves 2771MB headroom on RTX 3070
- AEC device name: "Echo Cancellation Source" (with spaces) for pasimple device_name param
- VRAMMonitor uses factory method create() returning None for graceful GPU-less operation

Phase 12-02 decisions:
- get_context() always returns at least one segment even if it exceeds token budget
- 46 hallucination phrases (18 existing + 28 research-backed from arXiv 2501.11378)
- TranscriptSegment is frozen (immutable) for thread safety

### Research Flags

- **Phase 12:** ~~PipeWire AEC device selection needs 30-min spike on this machine~~ DONE: "Echo Cancellation Source"
- **Phase 12:** ~~VRAM validation is go/no-go gate~~ DONE: GO (5421MB peak, 2771MB headroom)
- **Phase 13:** Llama 3.2 3B decision quality unknown — benchmark 20-30 scenarios before committing

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
Stopped at: Completed 12-02-PLAN.md
Resume file: None
