# Pitfalls Research

**Domain:** Voice-controlled async task orchestrator (OpenAI Realtime API + Claude CLI subprocesses)
**Researched:** 2026-02-13
**Confidence:** HIGH (verified against existing codebase, official docs, community reports)

## Critical Pitfalls

### Pitfall 1: Blocking the Asyncio Event Loop with Synchronous subprocess.run()

**What goes wrong:**
The current `openai_realtime.py` calls `execute_tool()` synchronously inside `handle_events()`, which runs on the asyncio event loop. `execute_tool()` uses `subprocess.run()` with `timeout=30` for `run_command` and `timeout=120` for `ask_claude`. While that subprocess blocks, the entire asyncio event loop freezes -- no WebSocket pings are answered, no audio chunks are sent or received, and the user hears dead silence. For Claude CLI tasks that take 30-120 seconds, this means the WebSocket connection will almost certainly be killed by OpenAI's keepalive timeout (pings every 20 seconds, connection drops on missed pong).

**Why it happens:**
The existing code was built for single synchronous tool calls that return quickly. The new async task orchestrator needs Claude CLI processes that run for minutes. Calling `subprocess.run()` from an async coroutine blocks the event loop thread because `subprocess.run()` is synchronous and does not yield control back to the event loop.

**How to avoid:**
Never call `subprocess.run()` from within an async event handler. Two approaches:
1. Use `asyncio.create_subprocess_exec()` for subprocess management from async code, reading stdout/stderr with `await process.communicate()` or incremental `await process.stdout.readline()`.
2. Use `loop.run_in_executor()` to offload `subprocess.run()` to a thread pool, keeping the event loop free.

For async Claude CLI tasks specifically, use `asyncio.create_subprocess_exec()` with `Popen`-style management -- spawn the process, store the `Process` object, and poll/await it without blocking. Return an immediate acknowledgment to the Realtime API ("Task started, I'll let you know when it's done") and deliver results later via `conversation.item.create`.

**Warning signs:**
- WebSocket disconnects with error code 1011 ("keepalive ping timeout") during tool execution
- Audio playback freezes or stutters when tools are running
- The Realtime session goes silent for the duration of any tool call
- `response.done` events stop arriving during tool execution

**Phase to address:**
Phase 1 -- This is the foundation. The entire async task system depends on non-blocking subprocess management. Must be solved before any Claude CLI integration works.

---

### Pitfall 2: Fire-and-Forget asyncio Tasks Getting Garbage Collected

**What goes wrong:**
When spawning background Claude CLI tasks with `asyncio.create_task()`, Python's event loop holds only weak references to tasks. If you do not store a strong reference to the task object, the garbage collector can destroy the task mid-execution -- silently cancelling the Claude CLI process without any error or callback. The task just vanishes.

**Why it happens:**
This is a well-documented Python asyncio design decision. The event loop uses `WeakSet` for tasks to avoid memory leaks from abandoned tasks. Developers naturally write `asyncio.create_task(run_claude_task(prompt))` without storing the return value, assuming the event loop will keep it alive. It will not.

**How to avoid:**
Maintain a strong reference set for all background tasks:
```python
_background_tasks: set[asyncio.Task] = set()

def spawn_task(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
```
Alternatively, use `asyncio.TaskGroup` (Python 3.11+) which handles task lifetime automatically. However, `TaskGroup` waits for all tasks on exit, so it is better suited for bounded groups rather than the open-ended task spawning this project needs. The set-based approach is more appropriate here.

**Warning signs:**
- Tasks that "sometimes complete and sometimes don't" with no pattern
- Claude CLI processes that start (visible in `ps`) but whose results never arrive
- No error messages, no exceptions -- just silence
- Symptoms appear under memory pressure or when many tasks are active

**Phase to address:**
Phase 1 -- Task lifecycle management must include proper reference retention from the start. Retrofitting this is error-prone because every task spawn point needs auditing.

---

### Pitfall 3: OpenAI Realtime API Session Expiry and Reconnection State Loss

**What goes wrong:**
OpenAI Realtime API sessions have a maximum duration of 15-30 minutes (reports vary; the limit has been extended over time). After this, the server sends `session_expired` and closes the WebSocket. Additionally, connections can drop at any time due to keepalive ping timeout (server pings every 20 seconds), network interruption, or server-side issues. When the session drops, all conversation context -- including the AI's awareness of running tasks, task names, and conversation history -- is lost. The user says "how's that refactoring going?" and the AI has no idea what they're talking about.

**Why it happens:**
The Realtime API is stateful on the server side during a session, but that state is ephemeral. There is no session resume mechanism for Realtime sessions. When the WebSocket closes, the conversation context is gone. The current codebase does not have reconnection logic at all.

**How to avoid:**
1. Maintain a local task registry (dict of task_id -> TaskState) that survives WebSocket reconnection. On reconnect, inject a system message summarizing active/completed tasks via `conversation.item.create`.
2. Implement WebSocket reconnection with exponential backoff. After reconnecting, rebuild the AI's context by creating conversation items that describe the current task state.
3. Track session start time and proactively reconnect before the 15-minute expiry (e.g., at 14 minutes, gracefully disconnect and reconnect with context restoration).
4. Handle the ping/pong keepalive correctly -- the `websockets` library handles this automatically by default (its `ping_interval=20` in the current code matches OpenAI's expectations), but verify this is actually working under load.

**Warning signs:**
- Sessions dying after 15 minutes of use
- Random disconnects mid-conversation with error 1011
- After reconnection, AI responds as if no prior conversation happened
- Users report "it forgot what I asked it to do"

**Phase to address:**
Phase 2 -- Basic task management can work within a single session first (Phase 1), but reconnection with state recovery should come in Phase 2 before the feature is relied upon for real work.

---

### Pitfall 4: The "Double Update" -- Concurrent State Mutation in Task Registry

**What goes wrong:**
Multiple async handlers mutate the task registry simultaneously. Example scenario: Task A completes and its callback updates the registry to mark it "done" while simultaneously the user asks about Task A and the Realtime API tool handler reads the registry showing "running." The tool returns stale state to the AI, which tells the user the task is still running. Then Task B completes and overwrites Task A's completion data in a shared dict because both callbacks fired in the same event loop tick.

**Why it happens:**
Even in single-threaded asyncio, race conditions exist between coroutines. When a coroutine `await`s (e.g., awaiting a WebSocket send), other coroutines run. If two coroutines read-modify-write shared state with an `await` in between the read and write, the state can be inconsistent. Additionally, if any task management code runs in a thread (via `run_in_executor` or the existing `threading.Thread` pattern), true thread-safety races occur.

**How to avoid:**
1. Use `asyncio.Lock` for task registry access if staying in pure async code.
2. If mixing threads and asyncio (as the existing codebase does), use `threading.Lock` for the task registry and `loop.call_soon_threadsafe()` to schedule state updates back on the event loop thread.
3. Prefer an immutable state pattern: task callbacks produce "events" (TaskStarted, TaskCompleted, TaskFailed) that are processed sequentially by a single consumer coroutine, rather than multiple writers directly mutating a shared dict.
4. Use dataclasses with frozen=True for task snapshots provided to the AI, ensuring it always sees a consistent view.

**Warning signs:**
- AI reports task status that contradicts reality
- Task results attributed to the wrong task
- "Task not found" errors for tasks that definitely exist
- Intermittent issues that only appear when multiple tasks run concurrently

**Phase to address:**
Phase 1 -- The task registry design must be concurrent-safe from day one. This is an architectural decision, not something to bolt on later.

---

### Pitfall 5: Claude CLI Process Zombie Accumulation

**What goes wrong:**
When spawning Claude CLI as subprocesses, failing to `wait()` on completed processes creates zombie entries in the process table. Over a working session with many tasks, zombie processes accumulate. Each zombie holds a PID slot and process table entry. Eventually, the system may hit the PID limit or the user sees hundreds of `[claude] <defunct>` processes in `ps`. More insidiously, if the parent Python process crashes or is killed without cleanup, all child Claude CLI processes become orphans that continue running (consuming API tokens and writing to disk) with no one monitoring them.

**Why it happens:**
The existing codebase already shows this pattern -- `subprocess.run()` handles cleanup automatically, but switching to async `Popen`-style management means the developer must explicitly `await process.wait()` or register completion callbacks. Orphaned processes happen because the systemd service (push-to-talk.service) may restart the Python process on crash, but the old Claude CLI children are now adopted by init and keep running.

**How to avoid:**
1. Always register a done callback or await the process to collect exit status. Use the set-based pattern from Pitfall 2 for process tracking.
2. Store PIDs in a file (`~/.local/share/push-to-talk/active-tasks.json`) so that on startup, the service can check for and kill orphaned Claude CLI processes from a previous session.
3. Use process groups (`os.setpgrp` / `os.killpg`) so that killing the parent also kills all child Claude CLI processes.
4. Set `start_new_session=True` on `Popen` only if you genuinely want the process to outlive the parent (you do not want this here).
5. Register an `atexit` handler and signal handler (SIGTERM from systemd) that kills all tracked Claude CLI processes.

**Warning signs:**
- `ps aux | grep claude` shows many defunct or orphaned processes
- System memory/CPU usage creeps up over time
- Anthropic API usage is higher than expected (orphaned Claude processes keep working)
- After service restart, previously running tasks show mysterious additional commits or file changes

**Phase to address:**
Phase 1 -- Process lifecycle management is foundational. The cleanup infrastructure (PID tracking, signal handlers, atexit) must be in place before any Claude CLI processes are spawned.

---

### Pitfall 6: Realtime API Function Call Response Ordering Bugs

**What goes wrong:**
When the Realtime API makes a function call and the tool takes time to execute, the model can behave unpredictably. Known issues include: (a) the model generating a response before receiving the function call output (hallucinating the result), (b) the model not responding at all after receiving `function_call_output` until the user speaks again (documented bug with `gpt-4o-realtime-preview-2025-06-03`), and (c) when multiple function calls are made in sequence, results getting associated with the wrong call.

**Why it happens:**
The Realtime API is streaming and event-driven. The model does not block waiting for function outputs in the same way a synchronous chat API does. If the `function_call_output` item and `response.create` trigger are not sent in the correct sequence, or if there are timing issues, the model's behavior becomes undefined. This is compounded by the current code's pattern of executing tools synchronously (blocking the event handler from processing other events while the tool runs).

**How to avoid:**
1. For async tasks that take more than a few seconds, return an immediate acknowledgment as the function result: `{"status": "started", "task_id": "abc123", "message": "I've started that task. I'll let you know when it's done."}`. This satisfies the function call contract immediately.
2. When the async task completes, inject results via `conversation.item.create` with a system/user message describing the result, then trigger `response.create` to have the AI speak it.
3. Never leave a function call hanging without a response -- the model may hallucinate a response or go silent.
4. Pin to a specific model version and test thoroughly when updating, as behavior varies between Realtime API model versions.
5. After sending `function_call_output`, always send `response.create` to trigger the model to respond (the current code does this correctly -- preserve this pattern).

**Warning signs:**
- AI announces task results that are wrong or haven't happened yet
- AI goes silent after a tool call, requiring user to speak to "wake" it
- Results from Task A being spoken as if they belong to Task B
- Different behavior between Realtime API model versions

**Phase to address:**
Phase 1 -- The immediate-acknowledgment pattern for async tools is the core UX mechanism. Must be designed correctly from the start.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Blocking subprocess.run() for Claude CLI in async handler | Simple, works for quick calls | WebSocket dies for any call > 20s, entire session freezes | Never for this project -- all Claude CLI calls are long-running |
| Single shared CLAUDE_SESSION_DIR for all tasks | No directory management needed | Tasks share Claude's conversation context, causing confusion and context bleed between unrelated tasks | Never -- context isolation is a core requirement |
| No reconnection logic | Fewer edge cases to handle | Sessions limited to 15 minutes max, any network blip kills the session permanently | Acceptable only in Phase 1 prototype, must be added in Phase 2 |
| Storing task state only in memory | Simple, fast | All task awareness lost on crash/restart, orphaned processes invisible | Acceptable in Phase 1 if PID file is still written for orphan cleanup |
| Using threading.Thread for task execution (existing pattern) | Fits current architecture | Mixing threading with asyncio creates subtle race conditions, thread-safety of task registry must be managed manually | Acceptable if task registry uses proper locking, but prefer pure asyncio |
| Polling for task completion | Easy to implement | Wastes CPU, increases latency between task completion and user notification | Acceptable as a first pass; replace with callback/event-based notification in Phase 2 |

## Integration Gotchas

Common mistakes when connecting to external services in this specific stack.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenAI Realtime API (WebSocket) | Not handling ping/pong keepalive, leading to disconnects after idle periods | The `websockets` library handles pong responses automatically when `ping_interval` is set. Verify it matches server expectations (20s). Do not set `ping_interval=None` thinking it reduces overhead. |
| OpenAI Realtime API (Function Calls) | Leaving function calls unanswered while waiting for a long-running task | Always return an immediate result from the function (even just "started"), then deliver actual results later via `conversation.item.create` |
| OpenAI Realtime API (Audio) | Sending audio while AI is speaking, causing echo/feedback loops | Mute mic during AI speech (current code does this correctly). Add cooldown period after speech ends before unmuting (current code uses 1.5s delay -- validate this is sufficient). |
| Claude CLI (Subprocess) | Using `subprocess.run()` which blocks until complete | Use `asyncio.create_subprocess_exec()` or `subprocess.Popen()` with non-blocking output reading |
| Claude CLI (Sessions) | Running all tasks in the same `cwd` with shared `.claude` session state | Give each task its own working directory (e.g., `~/.local/share/push-to-talk/tasks/{task_id}/`). Use `--resume {session_id}` only to continue the same logical task, never across different tasks. |
| Claude CLI (Output) | Capturing only stdout and missing stderr, or truncating large outputs | Capture both stdout and stderr. Claude CLI may produce large outputs for coding tasks -- stream output incrementally rather than buffering entire result, and summarize before speaking. |
| Claude CLI (Permissions) | Using `--permission-mode bypassPermissions` allowing unconstrained file system access from voice commands | Use `--permission-mode acceptEdits` to allow edits within the project but not arbitrary system commands. The current code already does this correctly for some modes. |
| PipeWire/PulseAudio (Mic Control) | Using `pactl` for mic muting which may conflict with PipeWire's native controls | Current approach works; if migrating to PipeWire-native tools later, test mute/unmute latency carefully |

## Performance Traps

Patterns that work at small scale but fail as tasks accumulate.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Accumulating conversation context in Realtime API | Token count grows, responses slow down, costs increase per interaction | Periodically summarize and trim conversation history. For task results, inject only summaries, not raw Claude CLI output (which can be thousands of lines). | After ~20 task interactions or ~30 minutes of conversation |
| Unbounded stdout/stderr buffering from Claude CLI | Memory usage spikes, OOM for tasks that produce large outputs (e.g., running tests, large diffs) | Stream output line-by-line with a max buffer (e.g., keep last 500 lines). Summarize before injecting into Realtime conversation. | When a single Claude CLI task produces >10MB of output |
| Polling all tasks in a status check loop | CPU usage scales linearly with task count, and each poll adds latency | Use `asyncio.wait()` with `return_when=FIRST_COMPLETED` or completion callbacks instead of polling loops | With 5+ concurrent tasks being actively polled |
| Creating a new WebSocket connection for each Realtime session without cleanup | File descriptor leaks from unclosed connections, especially on error paths | Always close WebSocket in a `finally` block. The current code does this in `disconnect()`, but ensure all error paths lead there. | After 10+ sessions without service restart |
| Loading full task output into Realtime API context | Token costs explode, API may reject messages exceeding token limits | Summarize Claude CLI output before injecting: "Task completed: refactored 3 files, added 45 lines, removed 12 lines" not the full diff | First time a coding task produces a large diff |

## Security Mistakes

Domain-specific security issues for a voice-controlled task orchestrator.

| Mistake | Risk | Prevention |
|---------|------|------------|
| AI executing shell commands based on voice transcription without validation | Misheard "delete the test" could become "delete the rest" -- voice transcription errors become destructive commands | Claude CLI's `--permission-mode acceptEdits` provides a guardrail. Never use `bypassPermissions` for task orchestration. Add a confirmation step for destructive operations (AI repeats back the task before starting). |
| Passing raw voice transcription as Claude CLI prompt without sanitization | Prompt injection via spoken words (unlikely but possible if someone nearby speaks adversarial prompts) | This is low risk in a desktop app used by one person, but worth noting: the prompt goes through Whisper transcription which naturally sanitizes most injection attempts. No immediate action needed. |
| Task output containing secrets being spoken aloud | Claude CLI working on a project might read/output API keys, passwords, etc. which the Realtime API would then speak | Filter task output summaries for common secret patterns before injecting into the voice conversation. Never pipe raw Claude CLI output to TTS. |
| Claude CLI processes inheriting the full parent environment (including OPENAI_API_KEY, etc.) | Child processes have access to all parent environment variables | Use explicit `env` dict with `subprocess` to pass only necessary variables. Or accept this risk since Claude CLI needs similar permissions anyway. |

## UX Pitfalls

Common user experience mistakes in voice-controlled task orchestration.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Silent processing -- no feedback while task is being spawned or while waiting for results | User doesn't know if their command was heard, repeats it, spawns duplicate tasks | Immediate voice acknowledgment: "Starting that refactoring task now. I'll let you know when it's done." Then periodic status if task runs long. |
| Verbose task status updates interrupting flow | User is trying to give a new command but the AI keeps talking about previous task updates | Batch status updates. Don't interrupt the user. Queue task completion notifications and deliver them during natural pauses or when user asks. |
| Ambiguous task references | User says "how's that task going?" and there are 3 active tasks -- AI guesses wrong | Assign memorable names to tasks (not UUIDs). When ambiguous, ask: "Which task -- the refactoring or the test suite?" Use the user's own words as task names. |
| No way to cancel a running task | User realizes they gave wrong instructions but the Claude CLI process keeps running, potentially making unwanted changes | Implement cancel via voice: "Cancel the refactoring task." Kill the Claude CLI process group and report what (if anything) was already changed. |
| Context overload from many concurrent tasks | AI tries to track 10 tasks and the conversation becomes confusing for both AI and user | Limit concurrent tasks (suggest 3-5 max). When limit reached, say "You've got 5 tasks running. Want me to wait for one to finish first?" |
| Latency between "task done" and user notification | Task finishes but user isn't told for 30+ seconds because notification waits for next interaction | If user is idle (not speaking), proactively notify: "Hey, that test suite task just finished -- all 42 tests passed." Use audio cue (short tone) before speaking to get attention. |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Async tool execution:** Tool returns result to Realtime API -- but verify the WebSocket was alive the entire time the tool ran. If connection dropped during tool execution, the result has nowhere to go.
- [ ] **Task context isolation:** Each task gets its own directory -- but verify Claude CLI doesn't share session state across directories via `~/.claude/` global config or session cache.
- [ ] **Task completion notification:** AI says "task is done" -- but verify stdout was fully consumed. Claude CLI may still be writing final output after exit code is available.
- [ ] **Mic mute/unmute during tool calls:** Audio doesn't echo -- but verify mute state is restored after error paths (e.g., tool throws exception, mic stays muted permanently).
- [ ] **WebSocket reconnection:** Connection re-establishes -- but verify conversation context was rebuilt and the AI knows about all active/completed tasks from before the disconnect.
- [ ] **Process cleanup on shutdown:** Service stops cleanly -- but verify no Claude CLI processes survive after `systemctl --user stop push-to-talk`. Check with `pgrep -f claude` after shutdown.
- [ ] **Task output summarization:** AI speaks a summary -- but verify the summary accurately reflects the full output. A task that "completed" might have completed with errors that the summary omitted.
- [ ] **Hold-to-talk during task notification:** AI is proactively reporting a task result when user presses PTT -- verify the notification is interrupted and user's input takes priority.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Event loop blocked by sync subprocess | LOW | Kill stuck tool call, reconnect WebSocket, retry with async pattern |
| Task garbage collected | MEDIUM | Check PID file for orphaned processes. If Claude CLI is still running, attach to its output. If dead, restart the task. User must be informed that a task was lost. |
| WebSocket session expired | LOW | Reconnect, rebuild context from local task registry, inform user briefly ("I just reconnected, but I remember what we were working on") |
| Task state race condition | HIGH | Audit all task state for consistency. May need to kill and restart affected tasks if state is corrupted. Implement event-sourced state to prevent recurrence. |
| Zombie process accumulation | LOW | Run `pkill -f 'claude.*push-to-talk'` or similar. Add PID file cleanup on service start. |
| Function call response ordering bug | MEDIUM | Re-inject correct task results into conversation. May require user to re-ask about the task. Pin model version to avoid regression. |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Blocking event loop (Pitfall 1) | Phase 1: Core async infrastructure | Tool call completes while audio continues playing; WebSocket stays connected during 60s+ Claude CLI execution |
| Task GC (Pitfall 2) | Phase 1: Task lifecycle | Run 10 concurrent tasks, verify all complete and deliver results; no tasks silently vanish |
| Session expiry (Pitfall 3) | Phase 2: Resilience | Run session for 20 minutes; verify automatic reconnection and context recovery; task awareness survives disconnect |
| State race conditions (Pitfall 4) | Phase 1: Task registry design | Spawn 5 tasks simultaneously, complete them in random order; verify all results are correctly attributed and status is consistent |
| Zombie processes (Pitfall 5) | Phase 1: Process lifecycle | Kill the Python service mid-session with running tasks; verify all Claude CLI processes are cleaned up; restart service and verify no orphans from previous session |
| Function call ordering (Pitfall 6) | Phase 1: Tool integration | Spawn an async task via voice; verify immediate acknowledgment; verify result delivery after completion; verify no hallucinated results during wait |

## Sources

- [Python asyncio subprocess documentation](https://docs.python.org/3/library/asyncio-subprocess.html) -- HIGH confidence
- [Python asyncio development guidelines](https://docs.python.org/3/library/asyncio-dev.html) -- HIGH confidence
- [asyncio.create_task garbage collection issue (CPython #91887)](https://github.com/python/cpython/issues/91887) -- HIGH confidence
- [CPython docs: task disappearing bug and reference retention](https://docs.python.org/3/library/asyncio-task.html) -- HIGH confidence
- [OpenAI Realtime API WebSocket docs](https://platform.openai.com/docs/guides/realtime-websocket) -- HIGH confidence
- [OpenAI community: long function calls and Realtime API](https://community.openai.com/t/long-function-calls-and-realtime-api/1119021) -- MEDIUM confidence
- [OpenAI community: no response after function calling bug](https://community.openai.com/t/realtime-api-no-response-after-function-calling-until-next-user-turn-gpt-4o-realtime-preview-2025-06-03/1297639) -- MEDIUM confidence
- [OpenAI community: WebSocket disconnects with keepalive ping timeout](https://community.openai.com/t/realtime-api-websocket-disconnects-randomly-in-nodejs/1044456) -- MEDIUM confidence
- [Gladia: concurrent pipelines for real-time voice AI](https://www.gladia.io/blog/concurrent-pipelines-for-voice-ai) -- MEDIUM confidence
- [SignalWire: the Double Update problem](https://signalwire.com/blogs/developers/the-double-update) -- MEDIUM confidence
- [Armin Ronacher on asyncio.create_task footgun (2025)](https://x.com/mitsuhiko/status/1920384040005173320) -- MEDIUM confidence
- [Claude CLI session management: --continue and --resume flags](https://claudelog.com/faqs/what-is-resume-flag-in-claude-code/) -- MEDIUM confidence
- [Python subprocess zombie process prevention](https://dnmtechs.com/killing-or-avoiding-zombie-processes-in-python-3-with-subprocess-module/) -- MEDIUM confidence
- Existing codebase analysis of `/home/ethan/code/push-to-talk/openai_realtime.py` and `/home/ethan/code/push-to-talk/push-to-talk.py` -- HIGH confidence (direct code inspection)

---
*Pitfalls research for: voice-controlled async task orchestrator (PTT + OpenAI Realtime + Claude CLI)*
*Researched: 2026-02-13*
