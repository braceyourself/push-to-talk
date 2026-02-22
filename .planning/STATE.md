# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** An always-present AI that listens, understands context, and contributes when it has something useful to add
**Current focus:** v2.0 Always-On Observer -- Phase 12 (Deepgram Streaming Infrastructure)

## Current Position

Milestone: v2.0 Always-On Observer (Refreshed)
Phase: 12 of 16 (Deepgram Streaming Infrastructure)
Plan: --
Status: Ready to plan
Last activity: 2026-02-22 -- Roadmap created (5 phases, 29 requirements mapped)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity (v2.0):**
- Total plans completed: 0 (reset after pivot)
- Average duration: --
- Total execution time: --

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Carried forward from v1.2:
- StreamComposer for unified audio queue with pre-buffering, cadence control, barge-in
- Heuristic pattern matching first (<1ms), model2vec semantic fallback second (5-10ms)
- Configurable idle timeout (0 = always-on)

v2.0 milestone decisions:
- Deepgram Nova-3 streaming replaces local Whisper (~150ms vs 1.5-3s latency)
- Silero VAD manages connection lifecycle (active/idle/sleep), NOT per-chunk audio gating
- Llama 3.1 8B replaces 3.2 3B (more VRAM available with Whisper off GPU)
- TranscriptBuffer and VRAMMonitor retained from original Phase 12 (committed artifacts)
- Start conservative (name-activation only in Phase 13), expand to proactive in Phase 15
- deepgram-sdk pinned >=5.3,<6.0 (v6 has breaking API changes)

### Artifacts retained from original Phase 12:
- `transcript_buffer.py` -- TranscriptSegment, TranscriptBuffer, is_hallucination() (committed: 4bb48ff)
- `vram_monitor.py` -- VRAMMonitor with NVML (committed: 1f7c396)
- `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf` -- PipeWire AEC (committed: 1f7c396)

### Research Flags

- **Phase 12:** VAD lifecycle gating (active/idle/sleep) needs implementation validation. PipeWire AEC effectiveness with cloud STT latency unverified.
- **Phase 13:** Llama 3.1 8B classification accuracy on ambient conversation needs benchmarking. Prompt tuning iterations expected.
- **Phase 15:** Proactive participation thresholds have no precedent data. Plan for calibration cycles.

### Pending Todos

- Assistant tool creation feature (user idea captured during Phase 8 execution)

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-22
Stopped at: Roadmap created, ready to plan Phase 12
Resume file: None
