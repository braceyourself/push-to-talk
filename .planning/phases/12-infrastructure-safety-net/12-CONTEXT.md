# Phase 12: Infrastructure + Safety Net - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Continuous audio capture, speech-to-text pipeline, echo cancellation, transcript buffer, VRAM validation, and hallucination filtering. This phase builds the always-on listening foundation — the infrastructure that Phase 13's decision engine will consume. No decision-making, no response generation, no barge-in logic.

</domain>

<decisions>
## Implementation Decisions

### Continuous capture behavior
- Always-on replaces PTT entirely — no push-to-talk button in v2.0
- Visible transcript log — user sees a running stream of what the system heard
- Whisper runs continuously (no VAD gating) — always transcribing regardless of audio content
- Capture begins immediately on service startup — true always-on, no manual start needed
- Buffer holds ~5 minutes and rotates automatically (from success criteria)

### Echo cancellation approach
- PipeWire AEC as primary echo cancellation method
- Fallback: software-level filtering (audio fingerprinting or timing-based exclusion) if PipeWire AEC doesn't work well
- Capture pipeline keeps running during AI speech playback — enables barge-in detection for Phase 14
- On echo cancellation failure (system hearing itself): log warning only, let downstream decision engine handle it

### Hallucination filtering
- Conservative filtering — only filter known hallucination phrases and repeated tokens, let borderline segments through
- Auto-tuning: ambient-based (measures noise floor, adjusts confidence thresholds automatically)
- User feedback enhancement: ability to mark segments as "real" or "noise" to improve filter over time
- New filter built from scratch — continuous capture has different characteristics than PTT
- Filtered segments: silent by default, debug mode config toggle to show them marked as [filtered]

### Resource management
- Transcription (Whisper) has priority over decision engine (Ollama) when VRAM is constrained
- Graceful degradation on OOM: fall back to CPU-only Whisper, log warning, resume GPU when memory frees
- Proactive VRAM management: monitor usage and pre-emptively unload/reload models before hitting limits
- Visible resource stats in terminal log: VRAM usage, Whisper latency, buffer depth

### Claude's Discretion
- Specific transcript log format and update frequency
- PipeWire AEC device configuration details
- VRAM threshold values for proactive management
- Exact hallucination phrase list for initial conservative filter
- CPU fallback Whisper model selection (may need smaller model for CPU)

</decisions>

<specifics>
## Specific Ideas

- User wants the feedback loop for hallucination filtering (mark as real/noise) but ambient-based auto-tuning is the primary mechanism — feedback is an enhancement, not the core
- Echo cancellation failure should be non-disruptive — just warn, don't auto-mute or stop the pipeline
- Resource stats should be at-a-glance useful, not cluttering the transcript log

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-infrastructure-safety-net*
*Context gathered: 2026-02-21*
