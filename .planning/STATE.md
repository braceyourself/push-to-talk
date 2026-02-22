# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** An always-present AI that listens, understands context, and contributes when it has something useful to add
**Current focus:** v2.0 Always-On Observer (Refreshed) — Defining requirements

## Current Position

Milestone: v2.0 Always-On Observer (Refreshed)
Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-22 — Architectural pivot: Deepgram streaming STT + local decision model

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity (v2.0):**
- Total plans completed: 0 (reset after pivot)
- Average duration: —
- Total execution time: —

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carried forward from v1.2:
- StreamComposer for unified audio queue with pre-buffering, cadence control, barge-in
- Heuristic pattern matching first (<1ms), model2vec semantic fallback second (5-10ms)
- Configurable idle timeout (0 = always-on)

v2.0 milestone decisions (original, still valid):
- Decouple inputs from LLM processing (independent input stream)
- Configurable response backend (Claude CLI / Ollama), auto-selected
- PipeWire echo cancellation as primary feedback loop prevention
- Start conservative (name-activation only), expand to proactive after validation

v2.0 architectural pivot (2026-02-22):
- Deepgram Nova-3 streaming replaces local Whisper (~150ms vs 1.5-3s latency)
- Silero VAD stays as local cost gate (only stream speech segments to Deepgram)
- Local decision model for "should I respond?" (more VRAM available with Whisper off GPU)
- TranscriptBuffer stays (already built and committed)
- VRAMMonitor stays (already built and committed)
- ContinuousSTT module replaced by Deepgram streaming integration

### Artifacts retained from original Phase 12:
- `transcript_buffer.py` — TranscriptSegment, TranscriptBuffer, is_hallucination() (committed: 4bb48ff)
- `vram_monitor.py` — VRAMMonitor with NVML (committed: 1f7c396)
- `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf` — PipeWire AEC (committed: 1f7c396)

### Research Flags

- **Decision model quality:** Which local model works best for respond/wait/ignore classification? Benchmark needed.
- **Deepgram SDK integration:** WebSocket streaming, KeepAlive patterns, error handling, reconnection.

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

Last session: 2026-02-22
Stopped at: Architectural pivot, spawning researchers
Resume file: None
