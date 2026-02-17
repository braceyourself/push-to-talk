# Phase 5: Barge-in - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

User can interrupt AI mid-speech by speaking, which cancels current TTS playback and queued audio, allowing the conversation to continue naturally. Assumes headphones (no echo cancellation in this phase).

</domain>

<decisions>
## Implementation Decisions

### Detection sensitivity
- Headphones assumed — no echo cancellation needed this phase (speaker bleed is a future milestone)
- Barge-in is always active during AI speech — no key press required
- Require ~0.5s of sustained speech before triggering barge-in to filter out coughs, throat clears, and brief noises

### Interruption behavior
- Quick fade (~100-200ms) on the AI's speech, not a hard cut
- Play a contextually appropriate filler clip during fade-out (trailing "mm", breath) to simulate natural interruption — smart selection scoped to interruption moments, using existing clip pool
- Cancel playback only — let LLM/TTS finish in background (text is generated, just not spoken)
- Brief cooldown (~1-2s) after a barge-in before allowing another one to avoid rapid-fire interruption chaos

### Context handling
- Keep full LLM response text in conversation history but annotate where the user interrupted (e.g., `[interrupted here]`)
- AI is aware it was interrupted — system prompt or context annotation tells it, so it can choose to address it or pick up where it left off
- Tool call results: note partial delivery — mark which results were spoken vs. not, so AI can offer to continue
- No limit on interrupted turns — all stay in context with their markers

### Recovery flow
- Immediately start listening after barge-in triggers — don't wait for fade/filler to finish
- Brief visual pulse/flash on the overlay when barge-in activates — momentary acknowledgment, then transitions to listening state (no new overlay state)
- Shortened silence threshold after interruption — assume the user has a quick correction or redirect, respond faster than normal turn-taking

### Claude's Discretion
- Exact VAD threshold values for the ~0.5s sustained speech requirement
- Cooldown duration tuning (1-2s range)
- Fade-out duration and curve
- How to select "contextually appropriate" filler clips for interruptions
- Exact silence threshold reduction for post-interrupt faster response
- Implementation of the `[interrupted here]` annotation format

</decisions>

<specifics>
## Specific Ideas

- Interruption should feel like interrupting a real person — the fade + trailing filler clip is key to naturalness
- User wants the AI to be a smart conversational partner that knows when it was cut off and can adapt

</specifics>

<deferred>
## Deferred Ideas

- Echo cancellation for speaker mode (no headphones) — future milestone
- Smart clip factory overhaul (broader intelligent filler selection beyond just interruption moments) — future phase

</deferred>

---

*Phase: 05-barge-in*
*Context gathered: 2026-02-17*
