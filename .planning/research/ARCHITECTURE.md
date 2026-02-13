# Architecture Patterns

**Domain:** Voice-controlled async task orchestrator (Live Mode)
**Researched:** 2026-02-13
**Overall Confidence:** HIGH (existing codebase well-understood, API capabilities verified)

## Recommended Architecture

The core challenge: a persistent OpenAI Realtime WebSocket session needs to spawn, monitor, and collect results from multiple long-running Claude CLI subprocesses -- all without blocking voice conversation. The user keeps talking; tasks keep running; results flow back when ready.

### Architecture Overview

```
                       +--------------------+
                       |   PushToTalk       |
                       | (hotkey routing)   |
                       +--------+-----------+
                                |
                   hotkey combo triggers live mode
                                |
                       +--------v-----------+
                       |   LiveSession      |
                       | (session lifecycle)|
                       +--------+-----------+
                                |
              +-----------------+------------------+
              |                                    |
    +---------v----------+              +----------v---------+
    | RealtimeSession    |              |   TaskManager      |
    | (voice layer)      |<------------>| (orchestration)    |
    | - WebSocket I/O    | tool calls   | - spawn tasks      |
    | - audio streaming  | & results    | - track state      |
    | - function calling |              | - collect output   |
    +--------------------+              +----+----------+----+
                                             |          |
                                    +--------v--+ +----v--------+
                                    | ClaudeTask | | ClaudeTask  |
                                    | (isolated) | | (isolated)  |
                                    | - subprocess| - subprocess |
                                    | - own cwd  | | - own cwd   |
                                    | - output   | | - output    |
                                    +------------+ +-------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With | Thread Model |
|-----------|---------------|-------------------|--------------|
| **PushToTalk** | Hotkey detection, mode routing, session lifecycle | LiveSession (start/stop) | Main thread (pynput listener) |
| **LiveSession** | Owns a Realtime session + TaskManager pair, manages session lifecycle | PushToTalk (lifecycle), RealtimeSession (voice), TaskManager (tasks) | Spawns asyncio event loop in background thread |
| **RealtimeSession** | WebSocket connection, audio I/O, function call dispatch | LiveSession (events), TaskManager (via tool handlers) | asyncio tasks within event loop |
| **TaskManager** | Spawn/track/collect Claude CLI processes, maintain task registry | RealtimeSession (tool call interface), ClaudeTask instances (subprocess management) | asyncio-compatible, runs tasks as asyncio subprocesses |
| **ClaudeTask** | Single Claude CLI subprocess with isolated context | TaskManager (lifecycle), filesystem (working dir, output) | asyncio subprocess (non-blocking) |
| **Indicator** | Status display, settings UI | PushToTalk (via status file) | Separate process (GTK main loop) |

## Component Design Details

### LiveSession -- The Glue Layer

LiveSession is the new component that ties the voice layer to the task layer. It replaces the current pattern where `start_realtime_session()` creates a bare RealtimeSession. Instead, LiveSession owns both the RealtimeSession and a TaskManager, and configures the RealtimeSession's tools to route through the TaskManager.

```python
class LiveSession:
    """Manages a live voice session with async task orchestration."""

    def __init__(self, api_key, on_status=None):
        self.session_id = str(uuid.uuid4())
        self.task_manager = TaskManager()
        self.realtime = RealtimeSession(
            api_key=api_key,
            on_status=on_status,
            tools=self._build_tools(),
            tool_handler=self._handle_tool_call,
        )
        self.active = False

    def _build_tools(self):
        """Define tools available to the Realtime AI."""
        return [
            # Task spawning
            {"name": "start_task", "description": "Spawn a Claude CLI task...", ...},
            # Task monitoring
            {"name": "check_tasks", "description": "Get status of all tasks...", ...},
            {"name": "get_task_result", "description": "Get the output of a completed task...", ...},
            # Task cancellation
            {"name": "cancel_task", "description": "Cancel a running task...", ...},
            # Lightweight tools (run inline, not as tasks)
            {"name": "run_command", ...},
            {"name": "read_file", ...},
            {"name": "remember", ...},
            {"name": "recall", ...},
        ]

    async def _handle_tool_call(self, name, arguments, call_id):
        """Route tool calls to appropriate handler."""
        if name == "start_task":
            # Spawn async -- return immediately with task_id
            task_id = await self.task_manager.spawn(arguments)
            return {"task_id": task_id, "status": "started"}
        elif name == "check_tasks":
            return self.task_manager.get_all_status()
        elif name == "get_task_result":
            return self.task_manager.get_result(arguments["task_id"])
        elif name == "cancel_task":
            return await self.task_manager.cancel(arguments["task_id"])
        else:
            # Inline tools (run_command, read_file, etc.)
            return execute_tool(name, arguments)
```

**Why this boundary:** LiveSession isolates the "what tools exist" question from both the voice layer (RealtimeSession doesn't need to know about tasks) and the task layer (TaskManager doesn't need to know about WebSockets). This makes each component testable and replaceable independently.

### TaskManager -- The Registry

TaskManager tracks all spawned ClaudeTask instances. It assigns IDs, monitors completion, and provides query interfaces. It does NOT own the asyncio event loop -- it participates in whatever loop LiveSession provides.

```python
class TaskManager:
    """Registry and orchestrator for async Claude CLI tasks."""

    def __init__(self):
        self.tasks: dict[str, ClaudeTask] = {}
        self._task_counter = 0

    async def spawn(self, args) -> str:
        """Spawn a new Claude CLI task. Returns task_id immediately."""
        self._task_counter += 1
        task_id = f"task-{self._task_counter}"
        task = ClaudeTask(
            task_id=task_id,
            prompt=args["prompt"],
            working_dir=args.get("working_dir"),
            description=args.get("description", f"Task {self._task_counter}"),
        )
        self.tasks[task_id] = task
        await task.start()  # Launches subprocess, returns immediately
        return task_id

    def get_all_status(self) -> dict:
        """Snapshot of all task states."""
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "status": t.status,  # "running", "completed", "failed", "cancelled"
                    "elapsed": t.elapsed_seconds(),
                    "summary": t.summary if t.status == "completed" else None,
                }
                for t in self.tasks.values()
            ]
        }

    def get_result(self, task_id) -> dict:
        """Get full output of a completed task."""
        task = self.tasks.get(task_id)
        if not task:
            return {"error": f"Unknown task: {task_id}"}
        if task.status == "running":
            return {"status": "running", "elapsed": task.elapsed_seconds()}
        return {
            "status": task.status,
            "output": task.output[:3000],  # Truncate for Realtime API context
            "elapsed": task.elapsed_seconds(),
        }

    async def cancel(self, task_id) -> dict:
        """Cancel a running task."""
        task = self.tasks.get(task_id)
        if task and task.status == "running":
            await task.cancel()
            return {"status": "cancelled"}
        return {"error": "Task not running"}
```

**Why a registry, not a queue:** Tasks are independent -- there's no ordering dependency between them. The user might say "start a task to fix the login bug" then "start another task to add tests for auth" and these run concurrently. A queue implies serial execution; a registry allows parallel execution with individual status tracking.

### ClaudeTask -- The Isolated Worker

Each ClaudeTask is a single Claude CLI subprocess with its own working directory and output buffer. Context isolation is achieved through filesystem separation: each task gets a unique temporary directory (or a user-specified project directory) and its own Claude session.

```python
class ClaudeTask:
    """A single Claude CLI subprocess with isolated context."""

    CLAUDE_CLI = Path.home() / ".local" / "bin" / "claude"

    def __init__(self, task_id, prompt, working_dir=None, description=""):
        self.task_id = task_id
        self.prompt = prompt
        self.description = description
        self.status = "pending"  # pending -> running -> completed/failed/cancelled
        self.output = ""
        self.start_time = None
        self.end_time = None
        self.process = None

        # Context isolation: each task gets its own working directory
        if working_dir:
            self.working_dir = Path(working_dir).expanduser().resolve()
        else:
            self.working_dir = Path.home()

    async def start(self):
        """Launch Claude CLI as async subprocess."""
        self.status = "running"
        self.start_time = time.time()

        self.process = await asyncio.create_subprocess_exec(
            str(self.CLAUDE_CLI),
            '-p', self.prompt,
            '--permission-mode', 'bypassPermissions',
            '--output-format', 'text',
            '--max-turns', '25',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.working_dir),
        )

        # Monitor completion in background (non-blocking)
        asyncio.create_task(self._monitor())

    async def _monitor(self):
        """Wait for process completion and capture output."""
        try:
            stdout, stderr = await asyncio.wait_for(
                self.process.communicate(),
                timeout=300,  # 5 minute max per task
            )
            self.output = stdout.decode('utf-8', errors='replace')
            self.status = "completed" if self.process.returncode == 0 else "failed"
            if self.process.returncode != 0:
                self.output += f"\n[stderr]: {stderr.decode('utf-8', errors='replace')}"
        except asyncio.TimeoutError:
            self.process.kill()
            self.output = "[Task timed out after 300 seconds]"
            self.status = "failed"
        except Exception as e:
            self.output = f"[Error: {e}]"
            self.status = "failed"
        finally:
            self.end_time = time.time()

    async def cancel(self):
        """Kill the subprocess."""
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
        self.status = "cancelled"
        self.end_time = time.time()

    def elapsed_seconds(self):
        end = self.end_time or time.time()
        return round(end - (self.start_time or end), 1)

    @property
    def summary(self):
        """First 200 chars of output as quick summary."""
        if self.output:
            return self.output[:200].strip()
        return None
```

**Context isolation approach:** Each ClaudeTask runs `claude -p` (non-interactive print mode) with its own `cwd`. This means:
- Each task has its own implicit Claude session (no `--continue` flag)
- File operations are scoped to that task's working directory
- No conversation history bleeds between tasks
- The `--permission-mode bypassPermissions` flag allows autonomous operation

**Why `-p` (print mode) and not `-c` (continue):** Tasks are independent work units, not conversational turns. Print mode runs a single prompt to completion and exits. This maps perfectly to "go do X and come back with the result." Using `-c` would create session entanglement between tasks, which is the opposite of context isolation.

### RealtimeSession Modifications

The existing RealtimeSession needs minimal changes. The key modification: extract the tool execution loop out of the event handler and make it pluggable, so LiveSession can inject its own tool_handler.

Current pattern (hardcoded tool execution):
```python
# In handle_events():
if item_type == "function_call":
    result = execute_tool(name, arguments)  # Synchronous, blocking
    await self.ws.send(...)  # Send result immediately
```

New pattern (pluggable, async-aware):
```python
# In handle_events():
if item_type == "function_call":
    # Delegate to handler (which may be async for task spawning)
    result = await self.tool_handler(name, arguments, call_id)
    if result is not None:
        # Immediate result -- send back now
        await self.ws.send(...)
    # If result is None, handler will send result later (async task)
```

**Critical insight from OpenAI Realtime API GA:** The GA model (`gpt-realtime`) supports async function calling natively. This means we can return a "task started" result immediately for `start_task` calls, and the model will gracefully handle the user asking about task status later -- it won't hallucinate results. The existing code uses the preview model (`gpt-4o-realtime-preview-2024-12-17`); upgrading to `gpt-realtime` is recommended for this feature.

### Async Task Completion Notification

When a task completes, the TaskManager can optionally inject a notification into the Realtime conversation:

```python
# In TaskManager, called from ClaudeTask._monitor() on completion:
async def _on_task_complete(self, task):
    """Notify the Realtime session that a task finished."""
    if self.on_task_complete:
        await self.on_task_complete(task)

# In LiveSession:
async def _notify_task_complete(self, task):
    """Inject task completion into Realtime conversation."""
    await self.realtime.ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "system",
            "content": [{
                "type": "input_text",
                "text": f"[Task '{task.description}' completed: {task.summary}]"
            }]
        }
    }))
    # Optionally trigger a response so the AI acknowledges
    await self.realtime.ws.send(json.dumps({
        "type": "response.create"
    }))
```

**Design choice:** Proactive notification vs. poll-only. Recommending proactive notification because the whole point of voice UX is hands-free -- the user shouldn't have to keep asking "is it done yet." The AI can naturally say "Hey, that login bug fix just finished -- want me to read you the summary?"

## Data Flow

### Happy Path: User Requests a Task

```
1. User speaks: "Hey, fix the login validation bug in the auth module"
2. Realtime API detects speech, transcribes, generates function_call for start_task
3. LiveSession._handle_tool_call() receives start_task with prompt + working_dir
4. TaskManager.spawn() creates ClaudeTask with isolated cwd, launches subprocess
5. Returns immediately: {"task_id": "task-1", "status": "started"}
6. Realtime AI speaks: "On it -- I've started task 1 to fix the login validation"
7. User can keep talking, ask questions, start more tasks
8. [30 seconds later] ClaudeTask._monitor() detects process completion
9. TaskManager._on_task_complete() fires notification callback
10. LiveSession injects system message into conversation
11. Realtime AI speaks: "Task 1 just finished -- the login validation is fixed"
12. User: "What did it change?" -> AI calls get_task_result -> reads output
```

### Concurrent Tasks

```
User: "Fix the login bug"          -> task-1 starts (cwd: ~/code/myapp)
User: "Also add tests for auth"    -> task-2 starts (cwd: ~/code/myapp)
User: "What's the status?"         -> AI calls check_tasks
                                    -> Returns both tasks' status
task-2 completes                   -> AI: "Tests are done, login fix still running"
task-1 completes                   -> AI: "Login fix is done too, both tasks complete"
```

### Error Handling Flow

```
Task fails (Claude CLI exits non-zero or times out):
1. ClaudeTask._monitor() captures stderr, sets status="failed"
2. Notification injected: "[Task 'fix login' failed: ...]"
3. AI speaks: "That task ran into an issue -- here's what happened..."
4. User can ask for details or retry
```

## Integration with Existing Architecture

### How Live Mode Fits Into the Mode System

The existing codebase has four AI modes: `claude`, `realtime`, `interview`, `conversation`. Live mode replaces the current `realtime` mode and adds task orchestration. The current "dictation live" mode gets renamed to "dictate."

Config changes:
```python
# dictation_mode options: "dictate" (was "live"), "prompt", "stream"
# ai_mode options: "claude", "live" (was "realtime"), "interview", "conversation"
```

### Hotkey Integration

Live mode uses the same PTT + AI key combo as the current realtime mode. The toggle behavior (press to start, press again to stop) carries over. The only difference: instead of creating a bare RealtimeSession, `PushToTalk.start_realtime_session()` creates a LiveSession which internally creates both.

```python
# In PushToTalk.on_press(), the ai_mode == 'live' branch:
if ai_mode == 'live':
    if self.live_session:
        self.stop_live_session()  # Toggle off
    else:
        self.start_live_session()  # Toggle on
```

### Status Indicator Updates

Live mode adds task-aware status to the indicator. The status file protocol stays the same (single string written to `status` file), but we add new states:

```python
# Existing: idle, recording, processing, success, error, listening, speaking
# New: task_running (indicates background tasks active while in live mode)
```

The indicator can show a compound state: "listening" (main dot color) + "task_running" (small secondary indicator). Implementation detail deferred to UI phase.

## Patterns to Follow

### Pattern 1: Async Subprocess via asyncio.create_subprocess_exec

**What:** Use asyncio's native subprocess support instead of threading + subprocess.run.
**When:** Always, for ClaudeTask execution.
**Why:** The Realtime session already runs in an asyncio event loop. Using asyncio subprocesses means tasks can be monitored without extra threads, and cancellation is clean (asyncio.Task cancellation propagates).

```python
process = await asyncio.create_subprocess_exec(
    str(CLAUDE_CLI), '-p', prompt,
    '--permission-mode', 'bypassPermissions',
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=str(working_dir),
)
stdout, stderr = await process.communicate()
```

### Pattern 2: Pluggable Tool Handler

**What:** Decouple tool execution from event handling in RealtimeSession.
**When:** Any time tools need to be configurable per-session.
**Why:** LiveSession needs different tool behavior than the current bare realtime mode. Making the tool handler a callback means RealtimeSession stays generic.

```python
class RealtimeSession:
    def __init__(self, api_key, on_status=None, tools=None, tool_handler=None):
        self.tools = tools or TOOLS
        self.tool_handler = tool_handler or execute_tool
```

### Pattern 3: Fire-and-Acknowledge for Long Tasks

**What:** Return an acknowledgment immediately, deliver the actual result later via conversation injection.
**When:** Any tool call that takes > 2 seconds.
**Why:** The Realtime API supports async function calling in GA, but the user experience is better when the AI acknowledges immediately ("I'm on it") rather than going silent for 30 seconds.

### Pattern 4: Context Isolation via Working Directory

**What:** Each ClaudeTask runs with its own `cwd`, no shared Claude session state.
**When:** Always for task spawning.
**Why:** Prevents cross-contamination. Task A editing files shouldn't affect Task B's view of the filesystem. Using `-p` (print mode) ensures no session continuity between tasks.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Shared Claude Session Across Tasks

**What:** Using `--continue` or `--session-id` to share a Claude session between multiple tasks.
**Why bad:** Tasks would see each other's conversation history, tool calls, and file edits. A task modifying a file would confuse another task reading the same file. Debug nightmare.
**Instead:** Each task is a fresh `claude -p` invocation with its own working directory.

### Anti-Pattern 2: Blocking the Asyncio Event Loop

**What:** Using `subprocess.run()` (synchronous) inside the async event handler.
**Why bad:** The entire Realtime session freezes -- no audio streaming, no event processing, no mic control. The user hears silence and thinks it crashed.
**Instead:** Use `asyncio.create_subprocess_exec()` and `await process.communicate()` with timeouts.

### Anti-Pattern 3: Polling for Task Status in a Loop

**What:** Having the Realtime AI repeatedly call `check_tasks` on a timer to detect completion.
**Why bad:** Wastes API tokens, creates annoying repeated interruptions ("Still running... still running..."), and the poll interval creates latency.
**Instead:** Use the proactive notification pattern -- TaskManager pushes completion events into the conversation.

### Anti-Pattern 4: Storing Task Output in Memory Indefinitely

**What:** Keeping all task output in the TaskManager dict forever during a session.
**Why bad:** Long sessions with many tasks accumulate output. Each `get_task_result` sends potentially 3000 chars to the Realtime API, consuming tokens.
**Instead:** Keep summaries (first 200 chars) in the registry, full output on disk. Load on demand.

### Anti-Pattern 5: Thread-per-Task Model

**What:** Spawning a Python thread for each Claude CLI subprocess (like the current conversation/interview modes do with `threading.Thread`).
**Why bad:** Mixes threading with asyncio, creates synchronization headaches, and makes it harder to cancel tasks cleanly. The existing codebase already suffers from thread-based complexity.
**Instead:** Stay fully in asyncio for the live mode path. The Realtime session already uses asyncio; keep tasks in the same event loop.

## Scalability Considerations

| Concern | At 1-2 tasks | At 5-10 tasks | At 20+ tasks |
|---------|--------------|---------------|-------------|
| Memory | Negligible (~1MB per task output) | ~10MB buffered output | Implement disk-based output storage |
| CPU | Fine (subprocess, not in-process) | Fine (OS handles scheduling) | May want to cap concurrent tasks |
| API cost | ~$0.10-0.50 per task | ~$1-5 per session | Add cost tracking, warn user |
| Realtime context | No issue | Context window starts filling | Implement task result summarization |
| Subprocess count | Fine | OS limit not a concern | Add max_concurrent_tasks config (default: 5) |

## Suggested Build Order (Dependencies)

This is the critical section for roadmap phase structure.

### Phase 1: RealtimeSession Refactor (foundation)

**Must come first.** The existing RealtimeSession has hardcoded tool execution. Before adding task management, the tool handler must be pluggable. Also upgrade from preview model to GA model.

Deliverables:
- Pluggable `tool_handler` callback on RealtimeSession
- Configurable `tools` list on RealtimeSession
- Upgrade WebSocket URL to `gpt-realtime` GA model
- Existing functionality preserved (current tools still work)

Dependencies: None (pure refactor of existing code)

### Phase 2: TaskManager + ClaudeTask (core capability)

**Depends on Phase 1 for the pluggable tool handler.** This is the new capability -- the ability to spawn and track Claude CLI subprocesses asynchronously.

Deliverables:
- TaskManager class (spawn, track, query, cancel)
- ClaudeTask class (async subprocess, output capture, timeout)
- Unit-testable in isolation (no Realtime API needed)

Dependencies: Phase 1 (pluggable handler interface)

### Phase 3: LiveSession Integration (assembly)

**Depends on Phase 1 + Phase 2.** Wire TaskManager into RealtimeSession via LiveSession. Define the task-oriented tools. Implement proactive completion notifications.

Deliverables:
- LiveSession class (owns RealtimeSession + TaskManager)
- Tool definitions (start_task, check_tasks, get_task_result, cancel_task)
- Tool handler routing
- Proactive task completion notification
- Integration into PushToTalk mode routing

Dependencies: Phase 1 (refactored RealtimeSession), Phase 2 (TaskManager)

### Phase 4: Mode Rename + UI (user-facing)

**Depends on Phase 3 for the new mode to exist.** Rename "live" to "dictate" and "realtime" to "live" across codebase and UI. Update indicator, settings, config.

Deliverables:
- Config migration (dictation_mode: "live" -> "dictate", ai_mode: "realtime" -> "live")
- Indicator UI updates (new mode names, task status display)
- Settings tab for live mode configuration
- Documentation updates

Dependencies: Phase 3 (live mode exists to be named)

## Sources

- OpenAI Realtime API documentation: https://platform.openai.com/docs/guides/realtime
- OpenAI Realtime GA announcement (gpt-realtime model): https://openai.com/index/introducing-gpt-realtime/
- OpenAI developer notes on async function calling: https://developers.openai.com/blog/realtime-api/
- Claude Code CLI reference: https://code.claude.com/docs/en/cli-reference
- Python asyncio subprocess docs: https://docs.python.org/3/library/asyncio-subprocess.html
- Existing codebase: `/home/ethan/code/push-to-talk/openai_realtime.py`, `/home/ethan/code/push-to-talk/push-to-talk.py`
- Existing architecture analysis: `/home/ethan/code/push-to-talk/.planning/codebase/ARCHITECTURE.md`

---

*Architecture research: 2026-02-13*
