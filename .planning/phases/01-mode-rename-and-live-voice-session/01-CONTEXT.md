# Phase 1: Mode Rename and Live Voice Session - Context

**Gathered:** 2026-02-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Rename the existing "live" dictation mode to "dictate" across all code, config, and UI. Then create a new "live" mode that opens an OpenAI Realtime voice session for real-time voice conversation. No tool access or task management in this phase — pure voice conversation.

</domain>

<decisions>
## Implementation Decisions

### Voice & response style
- Voice: Configurable in settings, default to Ash (global setting, not per-session)
- Verbosity: Concise — short, punchy responses, a few sentences max
- Tone: Focused assistant — friendly but efficient, stays on topic, gets to the point
- Personality: Direct and opinionated, dry humor and wit, task-oriented memory (connects dots between topics)
- Personality modeled after clawdbot — researcher should study `laptop:~/clawdbot` for the multi-file personality system
- Personality system: Multiple files drive behavior (not a single hardcoded prompt)
- Fillers allowed — they signal listening/processing and give the AI time to think

### Session lifecycle
- Session auto-starts when user selects Live mode in settings combo box
- Session ends on mode switch (no voice command to end)
- Auto-listen after AI responds — mic opens automatically for natural back-and-forth
- Idle timeout with reconnect — drops after silence, reconnects automatically on next PTT press

### Audio/visual feedback
- Voice announcement on connect/disconnect (AI says "Connected" / "Session ended")
- Overlay widget showing: colored dot (green=listening, blue=speaking, gray=idle/disconnected), waveform visualization, and status text
- Overlay is draggable, remembers last position

### Conversation scope
- General purpose — no domain restrictions, talk about anything
- Pure voice only in Phase 1 — no tools (tools come in Phase 3)
- Sliding window for active context (older exchanges fade) but full conversation history stored for reference
- Sessions load recent context/key points from previous sessions as seed

### Claude's Discretion
- Idle timeout duration (how long before disconnect)
- Overlay widget styling and default position
- Waveform visualization implementation
- Reconnection strategy details
- How to summarize/load recent context at session start

</decisions>

<specifics>
## Specific Ideas

- Personality system modeled after clawdbot — multi-file approach at `laptop:~/clawdbot` (researcher must investigate)
- Learnings system with cron exists on this machine — researcher should investigate for conversation history/context loading
- Fillers serve a UX purpose: they let the user know the AI is listening and acting, not just silent

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-mode-rename-and-live-voice-session*
*Context gathered: 2026-02-13*
