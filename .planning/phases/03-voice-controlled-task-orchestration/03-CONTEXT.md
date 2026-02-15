# Phase 3: Voice-Controlled Task Orchestration - Context

**Gathered:** 2026-02-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire task management into the live voice session so users can spawn, query, cancel, and receive results from Claude CLI tasks through natural conversation. The AI acts as a conversational assistant that happens to manage background tasks — conversation system first, task management second.

</domain>

<decisions>
## Implementation Decisions

### Task spawning voice UX
- Natural language detection — AI uses judgment from conversation context to decide when to spawn a task vs just answer a question
- Brief acknowledgment on spawn — "Got it, I'll have Claude work on that." then continue conversation naturally
- No concurrency limit — user can spawn as many tasks as they want
- Multi-context project awareness — AI learns projects over time as tasks are spawned in different directories; builds a mental map and can route future tasks by project name

### Notification & result delivery
- Wait for a natural pause in conversation before delivering task completion/failure notifications
- Short summary on completion — conservative speech, focused on conversational fluency, never a wall of text
- No sound cues — speech is the only notification mechanism
- Failures get same delivery style as success, just different content (no elevated priority)

### Task naming & referencing
- AI auto-generates short descriptive names from the request (e.g., "auth refactor", "fix tests")
- User can rename: "call that one X instead"
- User can refer to tasks by name, description, or number — AI accepts whatever is natural
- When reference is ambiguous (matches multiple tasks), AI asks to clarify
- All tasks from the session stay referenceable for the entire session duration

### Status & listing behavior
- Status queries include active tasks + recently completed (since last asked)
- AI speaks status conversationally and word-efficiently — spoken time is real time, every word costs seconds
- User can ask about a running task's progress — AI tails recent output and summarizes what it's doing
- AI proactively mentions task status when conversationally relevant (e.g., "that refactor has been running a while") — but mindful of timing and whether the topic has come up recently
- Completed task results are re-read from output each time (not from memory) — enables specific follow-up questions
- Cancellation is immediate, no confirmation — AI just does it
- After cancellation, AI reports what was accomplished only if it fits naturally in the conversation

### Core design principle
- **Conversation system first, task management second** — encoded in both the system prompt and tool descriptions
- System prompt sets the principle: be conversational, word-efficient, aware of spoken time cost
- Tool descriptions reinforce it: keep responses brief, prioritize fluency

### Claude's Discretion
- Exact wording of spawn acknowledgments and status summaries
- How to naturally weave task notifications into ongoing conversation
- When a proactive status mention adds value vs interrupts

</decisions>

<specifics>
## Specific Ideas

- "AI speech should always be conservative and focused on keeping conversational fluency. We want to be aware of the human and avoid spamming them with a wall of text/speech"
- "This is not a task notification issue, this is a realtime communication issue" — spoken status must account for how long saying something takes
- "Conversation system first, task management system second" — the AI is a conversation partner that also manages tasks, not a task dashboard that talks
- Multi-context: "We need to build a multi context switching/understanding system for the main orchestrator/agent" — the AI learns which projects exist as you work with them, rather than requiring pre-configuration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-voice-controlled-task-orchestration*
*Context gathered: 2026-02-15*
