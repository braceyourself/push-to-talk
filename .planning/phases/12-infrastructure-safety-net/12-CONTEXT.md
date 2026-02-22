# Phase 12: Deepgram Streaming Infrastructure - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Always-on audio capture via Deepgram Nova-3 cloud STT with VAD-driven connection lifecycle management (active/idle/sleep), echo suppression (PipeWire AEC + transcript fingerprinting), cost control, and Whisper fallback for network outages. This phase builds the streaming transcript infrastructure that Phase 13's decision engine will consume. No decision-making, no response generation, no barge-in logic.

</domain>

<decisions>
## Implementation Decisions

### Connection lifecycle
- Silero VAD drives state transitions: active (streaming audio) -> idle (KeepAlive) -> sleep (disconnected)
- Active to idle transition: moderate timeout (5-10 seconds of silence) to accommodate natural pauses between sentences
- Idle to sleep transition: short timeout (30-60 seconds) to disconnect quickly after conversation ends
- Sleep to active: buffer audio locally during ~1-2 second reconnection window, send buffered audio once WebSocket is up -- no words lost
- Lifecycle timeouts exposed in config.yaml (idle_timeout, sleep_timeout) for power users to tune

### Degradation & fallback
- Visual indicator only when switching to Whisper fallback -- dashboard shows "STT: Whisper (fallback)", no audio cue
- Fast fallback: 3-5 seconds of failed Deepgram reconnection before loading Whisper
- Buffer all audio during the fallback gap (before Whisper loads) -- transcribe the backlog once Whisper is ready, no words lost
- Periodic retry: try reconnecting to Deepgram every 30-60 seconds in background while on Whisper, switch back automatically when available

### Transcript presentation
- Real-time transcript displayed in GTK dashboard panel (not terminal log)
- Show both interim and final results: interim results appear visually distinct (gray/italic) as words stream in, then solidify to final text
- Each segment shows timestamp + STT source (Deepgram/Whisper) -- useful for debugging and system awareness
- Visible buffer depth indicator (e.g., "5:00 / 5:00") so user knows transcript capacity

### Echo handling
- Keep streaming audio to Deepgram during AI speech playback -- enables barge-in detection for Phase 14
- Defense in depth: PipeWire AEC handles acoustic echo cancellation, transcript fingerprinting catches anything that leaks through -- both always active
- Partial echo failure: filter echoed segments silently in normal mode, show as [echo] in debug mode
- Complete echo failure: degrade gracefully -- log warning, mark echoed segments, keep pipeline running, let downstream decision engine filter the rest

### Claude's Discretion
- Exact Silero VAD sensitivity thresholds for speech detection
- Transcript fingerprinting algorithm (fuzzy string matching, timing window, etc.)
- Deepgram API parameters (model, encoding, channels, sample rate)
- Whisper model selection for fallback (may need smaller model for fast loading)
- Dashboard panel layout and styling for transcript stream
- Reconnection backoff strategy for Deepgram retries

</decisions>

<specifics>
## Specific Ideas

- Interim results should feel like "watching someone type" -- gray/italic text that solidifies when finalized
- Buffer depth indicator gives user awareness of system capacity without being obtrusive
- Echo handling should never stop the pipeline -- worst case is some echoed text gets through, which downstream filters can handle
- "No words lost" is the guiding principle for all transition scenarios (idle->active, Deepgram->Whisper, reconnection gaps)

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 12-deepgram-streaming-infrastructure*
*Context gathered: 2026-02-22*
