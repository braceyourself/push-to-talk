# Feature Research

**Domain:** Voice-controlled async AI task orchestrator (live mode for push-to-talk)
**Researched:** 2026-02-13
**Confidence:** MEDIUM-HIGH (strong codebase understanding + ecosystem research; novel domain means fewer direct precedents)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels broken on first use.

#### Task Lifecycle

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Spawn task by voice | Core premise -- "ask Claude to do X" must work on first try | MEDIUM | Already have `ask_claude` tool in `openai_realtime.py` but it blocks synchronously (120s timeout). Must become async with `subprocess.Popen` or `asyncio.create_subprocess_exec` instead of `subprocess.run`. Each task needs a name/ID for reference. |
| Task completion notification | Users need to know when background work finishes | LOW | Audio chime + brief spoken summary ("Your auth refactor is done"). Without this, users have no idea tasks finished and the whole async model falls apart. |
| Task failure notification | Errors can't be silent -- users must hear about them | LOW | Distinct audio cue (different tone from success) + spoken error summary. Claude CLI exit code != 0 or stderr output triggers this. |
| Ask for task status | "What's happening with my tasks?" must return a useful answer | LOW | Expose a `get_task_status` tool to the Realtime API. Returns list of tasks with state (running/completed/failed), elapsed time, and last output line. The AI then speaks a summary. |
| Cancel a running task | Users must be able to stop work they no longer want | LOW | `process.terminate()` or `process.kill()` on the subprocess. Expose as `cancel_task` tool. Must handle cleanup (temp files, partial git changes). |
| Task result retrieval | "What did Claude say about the auth module?" | LOW | Store stdout/stderr in memory or temp file per task. Expose as `get_task_result` tool. Truncate to reasonable size for voice summary (last N lines or AI-summarized). |
| Session start/stop | Clear boundary for when live mode is active | LOW | Already have `start_realtime_session()` / `stop_realtime_session()`. Extend to initialize task registry and clean up on stop. |
| Persistent conversation context | AI remembers what was discussed within a session | LOW | OpenAI Realtime API maintains conversation history across turns within a WebSocket session. Already works. Need to ensure task context (what was spawned, results) persists in the conversation. |

#### Voice UX

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Non-blocking conversation during tasks | Must be able to keep talking while Claude works in background | MEDIUM | Current `ask_claude` blocks the event loop for up to 120s. Must run Claude CLI as a background subprocess with async monitoring. The Realtime WebSocket session must stay responsive. |
| Interrupt AI speech | Press Escape or start talking to stop current response | LOW | Already implemented via `_interrupt_requested` and `input_audio_buffer.speech_started` handling in `openai_realtime.py`. |
| Clear status indicator | Visual indicator showing what mode is active and task counts | LOW | Extend existing status dot system. Add task count to status file or indicator tooltip. Color already encodes state (red=recording, purple=speaking, blue=listening). |

### Differentiators (Competitive Advantage)

Features that set this apart from Serenade, Copilot Voice, SystemPrompt, or just using Claude CLI directly.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Named task context switching | "Switch to the database task" / "What's the auth task doing?" -- refer to tasks by natural name, not ID | MEDIUM | The AI assigns human-readable names based on the task description. Internal registry maps names to process handles. This is what makes voice orchestration usable -- you can't read a task ID aloud comfortably. |
| Proactive status announcements | AI volunteers updates: "Hey, your refactor just finished" during natural conversation pauses | HIGH | Monitor task completion in background. When a task finishes and the AI is not speaking/listening, inject a conversation item with the result summary. Tricky timing -- must not interrupt user mid-sentence. Use `input_audio_buffer.speech_started` / `speech_stopped` events to find safe windows. |
| Context-isolated concurrent tasks | Multiple Claude CLI tasks run simultaneously without bleeding context | MEDIUM | Each task gets its own working directory or worktree. Prevent one task from seeing another's partial changes. Claude CLI already supports `--add-dir` for context scoping. Could use git worktrees for true isolation on same-repo tasks. |
| Voice-driven task composition | "Take the output from the API task and feed it into the frontend task" | HIGH | Chain task results as input to new tasks. Requires the AI to understand task dependency graphs. Defer to v2 -- complex and rarely needed in v1. |
| Smart result summarization | AI reads full Claude CLI output but speaks a concise 2-3 sentence summary | MEDIUM | Claude CLI can produce hundreds of lines of output. The Realtime API AI must condense this for voice. Could either (a) truncate to last N lines and let the AI summarize, or (b) run a quick summarization pass. Option (a) is simpler and leverages the AI's native capability. |
| Ambient task awareness | AI automatically knows about running/completed tasks without being asked | LOW | Inject task registry state into the system prompt or as a periodic context update. When user asks "can you also fix the tests?" the AI knows what's already running and avoids duplicate work. |
| Audio notification differentiation | Different sounds for: task started, task completed, task failed, task needs attention | LOW | Simple audio cues (short wav files) played via `aplay`. Provides eyes-free awareness. Users learn the sounds and know what happened without the AI speaking. |
| Session persistence across reconnects | If WebSocket drops, reconnect and preserve task state | MEDIUM | Tasks run as OS processes independent of the WebSocket. Task registry persists to disk. On reconnect, re-inject task state into the new session's context. WebSocket drops shouldn't kill background tasks. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems in a voice-controlled context.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Interactive Claude CLI steering by voice | "Now tell that Claude to also check the tests" -- sending follow-up prompts to a running task | Running Claude CLI sessions don't accept stdin mid-execution. Would require a fundamentally different architecture (MCP server, agent teams protocol). Massive complexity for marginal value. | Spawn a new task with the refined prompt. Tasks are cheap. Let the user say "also do X" and spawn a fresh Claude with the additional context. |
| Real-time streaming of Claude CLI output | Show/speak every line as Claude works | Claude CLI produces verbose tool-use output (file reads, diffs, thinking). Speaking this aloud would be overwhelming and unusable. Visual streaming in a terminal is fine; audio streaming is not. | Store output silently, summarize on completion. User can ask "what's the auth task doing?" for a status check at any time. |
| Arbitrary shell command execution in live mode | "Run npm test" / "Deploy to staging" | Already exists as `run_command` tool but is dangerous without guardrails. In a voice context, misheard commands could be destructive. "Delete the logs" could become "delete the docs." | Keep `run_command` for read-only queries (git status, ls, date). Route destructive operations through Claude CLI which has its own permission model and confirmation flow. |
| Always-on listening (no PTT) | Hands-free convenience | Accidental activation, privacy concerns, battery/CPU drain from continuous STT, false positives triggering task spawns. The existing Realtime API uses server VAD which already provides continuous listening within a session. | Server VAD within the Realtime session handles the "always listening while session is active" need. The PTT key starts/stops the session, not individual utterances. This is already the design. |
| Visual task dashboard / TUI | Terminal-based task monitoring with live output | Defeats the purpose of voice-first interaction. If you're looking at a terminal, just use Claude CLI directly. The whole point is hands-free/eyes-free operation. | Status indicator dot with tooltip showing task count. Voice-queryable status ("what's running?"). Desktop notifications for completions. |
| Persistent tasks across sessions | Tasks survive session end and resume on next session start | Scope creep into job scheduler territory. Claude CLI processes are tied to their execution context. Resuming a Claude mid-thought is not possible. | Each session starts fresh. Completed results could persist to disk for reference, but running tasks end when the session ends. Warn user of running tasks on session stop. |
| Multi-user / shared task orchestration | Multiple people controlling the same task pool | Single-user desktop app. Adding multi-user adds auth, conflict resolution, permission models. Zero users have asked for this. | Not applicable -- this is a personal productivity tool. |

## Feature Dependencies

```
[Session Management (start/stop live mode)]
    |
    +---> [Task Registry (in-memory task tracking)]
    |         |
    |         +---> [Spawn Task (async Claude CLI)]
    |         |         |
    |         |         +---> [Task Completion Notification]
    |         |         +---> [Task Failure Notification]
    |         |         +---> [Task Result Storage]
    |         |
    |         +---> [Ask Task Status]
    |         +---> [Cancel Task]
    |         +---> [Get Task Result]
    |         +---> [Named Context Switching]
    |
    +---> [Non-blocking Conversation]
              |
              +---> [Proactive Status Announcements]
              +---> [Ambient Task Awareness]

[Context Isolation]
    +---> independent of task registry, but enhances [Spawn Task]
    +---> requires decision: subdirectory vs git worktree vs temp dir

[Audio Notification Differentiation]
    +---> independent, enhances [Task Completion/Failure Notification]

[Session Persistence Across Reconnects]
    +---> requires [Task Registry] to persist to disk
    +---> enhances [Session Management]
```

### Dependency Notes

- **Spawn Task requires Task Registry:** Can't manage what you can't track. The registry must exist before any task spawning.
- **Named Context Switching requires Task Registry:** Names are just a lookup key into the registry.
- **Proactive Status Announcements require Non-blocking Conversation:** Can't announce anything if the conversation loop is blocked waiting for a task.
- **Ambient Task Awareness enhances Spawn Task:** The AI needs registry access in its tool set to avoid spawning duplicate tasks.
- **Context Isolation is independent:** Can be added at any phase. Starts as simple (separate working dirs) and evolves (git worktrees) based on need.
- **Session Persistence requires disk-backed Task Registry:** In-memory registry is lost on crash. Persisting to disk enables reconnect recovery.

## MVP Definition

### Launch With (v1)

Minimum viable product -- what's needed to validate that voice-controlled task orchestration is useful.

- [ ] **Async task spawning** -- "Ask Claude to refactor the auth module" spawns a background Claude CLI process and immediately returns control to the conversation
- [ ] **Task registry** -- In-memory dict mapping task_id to {name, process, status, start_time, stdout, stderr}
- [ ] **Task completion/failure notification** -- Audio chime + spoken summary when a task finishes or fails
- [ ] **Status query** -- "What are my tasks doing?" returns spoken summary of all task states
- [ ] **Task cancellation** -- "Cancel the auth task" kills the subprocess
- [ ] **Task result retrieval** -- "What did the database task produce?" speaks a summary of stored output
- [ ] **Non-blocking conversation** -- User can keep talking while tasks run in background

### Add After Validation (v1.x)

Features to add once the core async loop is proven useful.

- [ ] **Named context switching** -- Refer to tasks by descriptive name instead of requiring exact wording. Trigger: users struggle to reference tasks.
- [ ] **Audio notification differentiation** -- Different sounds for start/complete/fail. Trigger: users want faster awareness without waiting for AI to speak.
- [ ] **Ambient task awareness in system prompt** -- AI automatically knows running tasks. Trigger: users repeatedly spawn duplicate tasks because AI doesn't know what's already running.
- [ ] **Context isolation via separate directories** -- Each task gets its own working directory. Trigger: concurrent tasks on same repo cause conflicts.
- [ ] **Smart result summarization** -- AI-powered condensation of long outputs. Trigger: Claude CLI output is too long for comfortable voice delivery.

### Future Consideration (v2+)

Features to defer until the core model is proven.

- [ ] **Proactive status announcements** -- AI volunteers updates during conversation pauses. Defer: complex timing logic, risk of interrupting user.
- [ ] **Session persistence across reconnects** -- Disk-backed registry survives WebSocket drops. Defer: adds persistence layer complexity, v1 sessions are short enough that reconnect is rarely needed.
- [ ] **Voice-driven task composition** -- Chain task outputs. Defer: requires dependency graph management, rare use case.
- [ ] **Git worktree isolation** -- True parallel branches for concurrent same-repo tasks. Defer: heavy infrastructure for an edge case in v1.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Async task spawning | HIGH | MEDIUM | P1 |
| Task registry | HIGH | LOW | P1 |
| Completion/failure notification | HIGH | LOW | P1 |
| Status query tool | HIGH | LOW | P1 |
| Task cancellation | HIGH | LOW | P1 |
| Task result retrieval | HIGH | LOW | P1 |
| Non-blocking conversation | HIGH | MEDIUM | P1 |
| Named context switching | MEDIUM | MEDIUM | P2 |
| Audio notification sounds | MEDIUM | LOW | P2 |
| Ambient task awareness | MEDIUM | LOW | P2 |
| Context isolation (directories) | MEDIUM | MEDIUM | P2 |
| Smart result summarization | MEDIUM | LOW | P2 |
| Proactive announcements | MEDIUM | HIGH | P3 |
| Session persistence / reconnect | LOW | MEDIUM | P3 |
| Task composition / chaining | LOW | HIGH | P3 |
| Git worktree isolation | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch -- the async orchestration loop does not work without these
- P2: Should have -- significantly improves usability, add as soon as core is stable
- P3: Nice to have -- defer until usage patterns emerge

## Competitor Feature Analysis

| Feature | SystemPrompt Code Orchestrator | Claude Code Agent Teams | Serenade / Talon Voice | Conversation Mode (existing) | Our Live Mode Approach |
|---------|-------------------------------|------------------------|----------------------|------------------------------|----------------------|
| Voice input | Yes (mobile app, experimental v0.01) | No (terminal-only) | Yes (speech-to-code, not task orchestration) | Yes (PTT + Whisper) | Yes (PTT + OpenAI Realtime API, low-latency) |
| Async task management | Yes (MCP-based agent manager) | Yes (shared task list, teammate spawning) | No (synchronous commands) | No (blocks on Claude CLI) | Yes (async subprocess with registry) |
| Multiple concurrent tasks | Yes (multiple agent sessions) | Yes (multiple teammates, file-lock coordination) | No | No | Yes (process pool with named tasks) |
| Context isolation | Partial (Docker containers) | Yes (each teammate has own context window) | N/A | N/A | Yes (separate working directories, optionally git worktrees) |
| Task status monitoring | Via MCP protocol | Shared task list (pending/in-progress/completed) | N/A | N/A | Voice-queryable status + audio cues |
| Error handling | MCP error responses | Teammates can stop on errors; lead can spawn replacements | Voice feedback for parse errors | Desktop notification | Audio cue + spoken error summary |
| Task cancellation | Kill agent session | Graceful shutdown request to teammates | N/A | N/A | Process termination with cleanup |
| Inter-task communication | Via MCP server | Direct messaging between teammates | N/A | N/A | Not in v1 (tasks are independent) |
| Voice-to-voice conversation | Experimental (mobile) | No | No (speech-to-text only) | TTS responses (not real-time) | Yes (OpenAI Realtime API, sub-second latency) |
| Platform | Desktop + mobile (via tunnel) | Terminal (any OS) | Desktop (Windows/Mac/Linux) | Linux only | Linux only |

### Key Competitive Insights

1. **No one does voice + async task orchestration well yet.** SystemPrompt is closest but is experimental (v0.01) and mobile-focused. Claude Code Agent Teams is powerful but terminal-only with no voice interface. This is genuinely novel territory.

2. **Claude Code Agent Teams sets the bar for task coordination features** -- shared task lists, named tasks, dependency management, inter-agent messaging. We should study their patterns but adapt for voice (no file browsing, no split panes -- everything must be speakable).

3. **The voice advantage is real-time awareness.** Terminal users have to actively check task status. Voice users get proactive notifications. This is the core differentiator over Agent Teams.

4. **Serenade/Talon solve voice-to-code, not voice-to-orchestration.** They translate speech into code edits. We translate speech into task management commands. Different problem entirely.

## Sources

- [Claude Code Agent Teams documentation](https://code.claude.com/docs/en/agent-teams) -- HIGH confidence, official Anthropic docs
- [Claude Code background tasks patterns](https://apidog.com/blog/claude-code-background-tasks/) -- MEDIUM confidence, third-party analysis
- [SystemPrompt Code Orchestrator](https://github.com/systempromptio/systemprompt-code-orchestrator) -- MEDIUM confidence, GitHub project docs
- [Pipecat voice agent framework](https://github.com/pipecat-ai/pipecat) -- MEDIUM confidence, framework for voice agent patterns
- [VUI Design Principles](https://www.parallelhq.com/blog/voice-user-interface-vui-design-principles) -- MEDIUM confidence, general VUI best practices
- [Voice AI stack 2026](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents) -- MEDIUM confidence, ecosystem overview
- [Serenade voice coding](https://serenade.ai/) -- HIGH confidence, official product
- [Google VUI Design](https://design.google/library/speaking-the-same-language-vui) -- HIGH confidence, Google design guidelines
- Existing codebase analysis (`openai_realtime.py`, `push-to-talk.py`) -- HIGH confidence, direct code review

---
*Feature research for: Voice-controlled async AI task orchestrator (push-to-talk live mode)*
*Researched: 2026-02-13*
