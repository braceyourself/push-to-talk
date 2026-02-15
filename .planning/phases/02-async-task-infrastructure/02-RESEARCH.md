# Phase 2: Async Task Infrastructure - Research

**Researched:** 2026-02-15
**Domain:** Python asyncio subprocess management, Claude CLI programmatic invocation
**Confidence:** HIGH

## Summary

This phase builds a TaskManager singleton and ClaudeTask dataclass for spawning, tracking, and cancelling Claude CLI subprocesses without blocking the asyncio event loop. The codebase already uses asyncio extensively in `live_session.py` and `openai_realtime.py` (for WebSocket streaming and audio recording), and already spawns Claude CLI via `subprocess.run()` in multiple places -- but always synchronously and blocking. The task is to replace blocking `subprocess.run()` calls with `asyncio.create_subprocess_exec()` and wrap them in a management layer.

The existing architecture runs its asyncio event loop in a dedicated daemon thread (see `start_live_session()` at line 899 of `push-to-talk.py`). The TaskManager must live in this same event loop. Since it is a singleton that persists across sessions, it should be created once and its event loop reference retained.

**Primary recommendation:** Use `asyncio.create_subprocess_exec()` directly (not the Claude Agent SDK) for maximum control over process lifecycle, output streaming, and cleanup. The TaskManager is a pure-Python class with no external dependencies beyond the stdlib `asyncio` module.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `asyncio` (stdlib) | Python 3.12 | Async subprocess creation, task management | Already used throughout codebase; provides `create_subprocess_exec`, `create_task`, `TaskGroup` |
| `collections.deque` (stdlib) | Python 3.12 | Ring buffer for output capping | Built-in, O(1) append/popleft, `maxlen` parameter handles auto-eviction |
| `dataclasses` (stdlib) | Python 3.12 | ClaudeTask data structure | Clean, typed, lightweight; already the Python standard for data containers |
| `enum` (stdlib) | Python 3.12 | TaskStatus enumeration | Type-safe status tracking |
| `pathlib` (stdlib) | Python 3.12 | Path handling for working directories | Already used throughout codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `signal` (stdlib) | Python 3.12 | Process group management | When killing Claude CLI + its child processes |
| `os` (stdlib) | Python 3.12 | `os.killpg()` for process tree cleanup | Forceful termination of process groups |
| `json` (stdlib) | Python 3.12 | Parsing stream-json output from Claude CLI | If using `--output-format stream-json` |
| `time` (stdlib) | Python 3.12 | Timestamps for task metadata | Task creation/completion timing |
| `typing` (stdlib) | Python 3.12 | Type hints for callbacks and generics | API clarity |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw `asyncio.create_subprocess_exec` | `claude-agent-sdk` Python package | SDK wraps CLI subprocess anyway; adds dependency, bundles its own CLI binary, requires ANTHROPIC_API_KEY (not OpenAI-style file); raw subprocess gives us full control over process lifecycle, output streaming, and cleanup. The existing codebase already invokes Claude CLI directly. |
| `collections.deque` ring buffer | Custom ring buffer class | Unnecessary complexity; `deque(maxlen=N)` handles auto-eviction natively |
| `dataclasses.dataclass` | `TypedDict` or plain dict | Dataclass gives attribute access, defaults, and type checking |

**Installation:**
```bash
# No new dependencies needed -- everything is Python stdlib
# Claude CLI already installed at ~/.local/bin/claude (version 2.1.42)
```

## Architecture Patterns

### Recommended Project Structure
```
push-to-talk/
├── task_manager.py          # NEW: TaskManager singleton + ClaudeTask dataclass
├── push-to-talk.py          # Existing: main app (will import task_manager in Phase 3)
├── live_session.py          # Existing: LiveSession (will use TaskManager in Phase 3)
├── openai_realtime.py       # Existing: RealtimeSession (execute_tool becomes async in Phase 3)
└── ...
```

### Pattern 1: TaskManager Singleton
**What:** A single TaskManager instance per application, created once and shared across sessions.
**When to use:** Always -- the user decided TaskManager is a singleton that persists across sessions.

```python
# Source: User decision in CONTEXT.md + asyncio best practices
import asyncio
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import Callable, Optional, Any
from pathlib import Path
import time

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ClaudeTask:
    id: int
    name: str
    prompt: str
    project_dir: Path
    status: TaskStatus = TaskStatus.PENDING
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    _asyncio_task: Optional[asyncio.Task] = field(default=None, repr=False)
    output_lines: deque = field(default_factory=lambda: deque(maxlen=1000))
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    return_code: Optional[int] = None

class TaskManager:
    _instance: Optional['TaskManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._tasks: dict[int, ClaudeTask] = {}
        self._next_id = 1
        self._active_tasks: set[asyncio.Task] = set()  # Strong references
        self._project_locks: dict[str, int] = {}  # project_dir -> task_id
        self._callbacks: dict[str, list[Callable]] = {
            'on_task_complete': [],
            'on_task_failed': [],
            'on_output_line': [],
        }
```

### Pattern 2: Async Subprocess with Line-by-Line Streaming
**What:** Spawn Claude CLI as an async subprocess and read stdout line-by-line without blocking.
**When to use:** Every task spawn.

```python
# Source: Python 3.12 asyncio docs + Claude CLI --output-format docs
async def _run_task(self, task: ClaudeTask) -> None:
    """Internal coroutine that runs a Claude CLI subprocess."""
    try:
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        cmd = [
            str(Path.home() / '.local' / 'bin' / 'claude'),
            '-p', task.prompt,
            '--no-session-persistence',
            '--permission-mode', 'bypassPermissions',
            '--output-format', 'text',
        ]

        task.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            cwd=str(task.project_dir),
            start_new_session=True,  # Create new process group for clean kill
        )

        # Stream output line-by-line
        while True:
            line = await task.process.stdout.readline()
            if not line:
                break
            decoded = line.decode('utf-8', errors='replace').rstrip('\n')
            task.output_lines.append(decoded)
            await self._fire_callbacks('on_output_line', task, decoded)

        return_code = await task.process.wait()
        task.return_code = return_code
        task.completed_at = time.time()

        if return_code == 0:
            task.status = TaskStatus.COMPLETED
            await self._fire_callbacks('on_task_complete', task)
        else:
            task.status = TaskStatus.FAILED
            await self._fire_callbacks('on_task_failed', task)

    except asyncio.CancelledError:
        task.status = TaskStatus.CANCELLED
        await self._terminate_process(task)
        raise
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        task.output_lines.append(f"[Internal error: {e}]")
        await self._fire_callbacks('on_task_failed', task)
```

### Pattern 3: Strong Reference Management for asyncio Tasks
**What:** Keep strong references to asyncio.Task objects to prevent garbage collection mid-execution.
**When to use:** Every time `asyncio.create_task()` is called.

```python
# Source: Python 3.12 asyncio docs - "The event loop only keeps weak references to tasks"
async def spawn_task(self, name: str, prompt: str, project_dir: Path) -> ClaudeTask:
    """Spawn a new Claude CLI task."""
    # Enforce one-per-project-directory
    dir_key = str(project_dir.resolve())
    if dir_key in self._project_locks:
        existing_id = self._project_locks[dir_key]
        existing = self._tasks.get(existing_id)
        if existing and existing.status == TaskStatus.RUNNING:
            raise ValueError(f"Task {existing_id} already running in {project_dir}")

    task = ClaudeTask(
        id=self._next_id,
        name=name,
        prompt=prompt,
        project_dir=project_dir,
    )
    self._next_id += 1
    self._tasks[task.id] = task
    self._project_locks[dir_key] = task.id

    # Create asyncio task with strong reference
    asyncio_task = asyncio.create_task(
        self._run_task(task),
        name=f"claude-task-{task.id}"
    )
    task._asyncio_task = asyncio_task
    self._active_tasks.add(asyncio_task)
    asyncio_task.add_done_callback(self._active_tasks.discard)

    return task
```

### Pattern 4: Graceful Process Termination with Escalation
**What:** Terminate process gracefully (SIGTERM to process group), escalate to SIGKILL after timeout.
**When to use:** Task cancellation or cleanup.

```python
# Source: Python asyncio subprocess docs + official Claude SDK disconnect() pattern
import os
import signal

async def _terminate_process(self, task: ClaudeTask) -> None:
    """Terminate a task's subprocess gracefully, escalating to SIGKILL."""
    if task.process is None or task.process.returncode is not None:
        return

    try:
        # Send SIGTERM to the entire process group (Claude + its children)
        pgid = os.getpgid(task.process.pid)
        os.killpg(pgid, signal.SIGTERM)

        # Wait up to 5 seconds for graceful shutdown
        try:
            await asyncio.wait_for(task.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Force kill the process group
            os.killpg(pgid, signal.SIGKILL)
            await task.process.wait()
    except ProcessLookupError:
        pass  # Already dead
```

### Pattern 5: Callback/Event Hook System
**What:** Simple async callback registration for task lifecycle events.
**When to use:** Phase 3 will register handlers for voice notifications.

```python
# Source: Standard observer pattern adapted for asyncio
def on(self, event: str, callback: Callable) -> None:
    """Register a callback for an event."""
    if event not in self._callbacks:
        raise ValueError(f"Unknown event: {event}")
    self._callbacks[event].append(callback)

async def _fire_callbacks(self, event: str, *args) -> None:
    """Fire all registered callbacks for an event."""
    for callback in self._callbacks.get(event, []):
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"TaskManager callback error ({event}): {e}", flush=True)
```

### Anti-Patterns to Avoid
- **Using `subprocess.run()` for Claude CLI tasks:** This blocks the entire asyncio event loop. Always use `asyncio.create_subprocess_exec()`.
- **Using `process.communicate()` for long-running tasks:** This buffers all output in memory. Use `readline()` loop for streaming.
- **Forgetting `start_new_session=True`:** Without this, `process.terminate()` only kills the top-level Claude process, leaving child processes (Node.js, tool subprocesses) as zombies.
- **Reading stdout AND stderr as separate pipes:** Can cause ordering issues and deadlocks. Merge with `stderr=asyncio.subprocess.STDOUT`.
- **Using `asyncio.TaskGroup` for fire-and-forget tasks:** TaskGroup waits for all tasks on exit and cancels remaining on exception -- wrong for long-lived background tasks. Use `create_task()` with strong reference set instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ring buffer for output | Custom circular buffer class | `collections.deque(maxlen=N)` | O(1) operations, auto-eviction, thread-safe for single append/pop in CPython |
| Process output streaming | Custom polling loop with `read(N)` | `await process.stdout.readline()` | Handles buffering, EOF detection, backpressure correctly |
| Task ID generation | UUID or random string | Auto-incrementing integer | User decided on integer IDs; simpler to reference by voice ("task 3") |
| Process group management | Manual PID tracking | `start_new_session=True` + `os.killpg()` | OS handles the process group boundary correctly |
| Async event dispatch | Full pub/sub framework (pyventus, etc.) | Simple callback list with `async def _fire_callbacks()` | Only 3 event types needed; a framework is overkill |

**Key insight:** This phase's entire implementation uses Python standard library only. No new dependencies are needed. The complexity is in correct async lifecycle management, not in library selection.

## Common Pitfalls

### Pitfall 1: Zombie Processes from Missing `await process.wait()`
**What goes wrong:** Claude CLI process exits but is never reaped, accumulating zombie entries in the process table.
**Why it happens:** If the output reading loop exits early (exception, cancellation) without calling `process.wait()`, the OS keeps the process entry.
**How to avoid:** Always call `await process.wait()` in a `finally` block after the output loop, even on cancellation.
**Warning signs:** `ps aux | grep defunct` shows zombie processes after task completion.

### Pitfall 2: Orphaned Child Processes
**What goes wrong:** Claude CLI spawns Node.js subprocesses (for tool execution). Killing only the top-level Claude process leaves these children running.
**Why it happens:** `process.terminate()` only sends SIGTERM to the PID, not its children.
**How to avoid:** Use `start_new_session=True` when creating the subprocess, then `os.killpg(pgid, signal.SIGTERM)` to kill the entire group.
**Warning signs:** Node.js or other processes accumulating in `ps aux` after task cancellation.

### Pitfall 3: Event Loop Thread Mismatch
**What goes wrong:** TaskManager methods called from the wrong thread (e.g., the main GTK thread) fail silently or raise RuntimeError.
**Why it happens:** The asyncio event loop runs in a daemon thread (`start_live_session()` creates it). Calling `asyncio.create_task()` from outside that thread doesn't work.
**How to avoid:** Use `asyncio.run_coroutine_threadsafe(coro, loop)` when calling TaskManager from outside the event loop thread, or ensure all calls go through the event loop.
**Warning signs:** `RuntimeError: no running event loop` or tasks silently never starting.

### Pitfall 4: Garbage Collection of Fire-and-Forget Tasks
**What goes wrong:** An `asyncio.create_task()` call creates a background task, but no strong reference is kept. The task gets garbage collected before completion.
**Why it happens:** The event loop only holds weak references to tasks (documented CPython behavior).
**How to avoid:** Maintain a `set()` of active tasks, use `add_done_callback(set.discard)` to clean up after completion.
**Warning signs:** Tasks randomly "disappear" -- logs show creation but no completion.

### Pitfall 5: Output Buffer Memory Bloat
**What goes wrong:** A verbose Claude CLI task generates thousands of output lines, consuming excessive memory.
**Why it happens:** Storing all output lines without a cap.
**How to avoid:** Use `deque(maxlen=N)` as a ring buffer. The user decided on this approach. A maxlen of 1000 lines is reasonable (approximately 200KB at 200 chars/line).
**Warning signs:** Memory usage grows linearly with task runtime for verbose tasks.

### Pitfall 6: Deadlock from Separate stdout/stderr Pipes
**What goes wrong:** Reading from stdout blocks while the subprocess is blocked writing to stderr (or vice versa), causing a deadlock.
**Why it happens:** OS pipe buffers (typically 64KB) fill up. If you read stdout first and the process writes to stderr, the stderr buffer fills and the process blocks.
**How to avoid:** Merge stderr into stdout with `stderr=asyncio.subprocess.STDOUT`. The user left stdout/stderr handling as Claude's discretion; merging is the safest approach.
**Warning signs:** Task appears to hang indefinitely despite the subprocess still being alive.

### Pitfall 7: Race Condition on One-Per-Project Enforcement
**What goes wrong:** Two tasks are spawned for the same project directory near-simultaneously, bypassing the one-per-project check.
**Why it happens:** The check-then-act pattern is not atomic.
**How to avoid:** Since asyncio is single-threaded within the event loop, this is only a concern if `spawn_task()` is called from outside the loop. Ensure all spawn calls go through the event loop, making them inherently serialized.
**Warning signs:** Two Claude CLI processes running in the same project directory.

## Code Examples

### Complete Task Spawn and Cancellation
```python
# Source: Synthesized from asyncio docs + Claude SDK patterns
async def cancel_task(self, task_id: int) -> bool:
    """Cancel a running task by ID."""
    task = self._tasks.get(task_id)
    if not task or task.status != TaskStatus.RUNNING:
        return False

    task.status = TaskStatus.CANCELLED
    task.completed_at = time.time()

    # Cancel the asyncio task (triggers CancelledError in _run_task)
    if task._asyncio_task and not task._asyncio_task.done():
        task._asyncio_task.cancel()

    # Also terminate the process directly for immediate effect
    await self._terminate_process(task)

    # Release project lock
    dir_key = str(task.project_dir.resolve())
    if self._project_locks.get(dir_key) == task_id:
        del self._project_locks[dir_key]

    return True
```

### Task Query Methods
```python
# Source: Standard Python patterns
def get_task(self, task_id: int) -> Optional[ClaudeTask]:
    """Get a task by ID."""
    return self._tasks.get(task_id)

def find_task_by_name(self, name: str) -> Optional[ClaudeTask]:
    """Find a task by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for task in self._tasks.values():
        if name_lower in task.name.lower():
            return task
    return None

def get_all_tasks(self) -> list[ClaudeTask]:
    """Get all tasks, newest first."""
    return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

def get_running_tasks(self) -> list[ClaudeTask]:
    """Get all currently running tasks."""
    return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]

def get_task_output(self, task_id: int) -> Optional[str]:
    """Get the current output of a task as a string."""
    task = self._tasks.get(task_id)
    if not task:
        return None
    return '\n'.join(task.output_lines)
```

### Output Persistence on Completion
```python
# Source: User decision -- persist final output to disk
async def _persist_output(self, task: ClaudeTask) -> None:
    """Write task output to a file in the project directory."""
    output_dir = task.project_dir / '.claude-tasks'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'task-{task.id}-output.md'

    content = f"# Task {task.id}: {task.name}\n\n"
    content += f"**Status:** {task.status.value}\n"
    content += f"**Project:** {task.project_dir}\n"
    content += f"**Created:** {time.ctime(task.created_at)}\n"
    if task.completed_at:
        duration = task.completed_at - (task.started_at or task.created_at)
        content += f"**Duration:** {duration:.1f}s\n"
    content += f"\n## Output\n\n```\n"
    content += '\n'.join(task.output_lines)
    content += "\n```\n"

    # Write async-safely (small file, acceptable to block briefly)
    output_file.write_text(content)
```

### Claude CLI Invocation Flags
```python
# Source: claude --help output (v2.1.42) + user CONTEXT.md decisions
CLAUDE_CLI = Path.home() / '.local' / 'bin' / 'claude'

def _build_claude_command(self, task: ClaudeTask) -> list[str]:
    """Build the Claude CLI command for a task."""
    cmd = [
        str(CLAUDE_CLI),
        '-p', task.prompt,                    # Non-interactive print mode
        '--no-session-persistence',            # Don't save session to disk
        '--permission-mode', 'bypassPermissions',  # No permission prompts
        '--output-format', 'text',             # Plain text output for streaming
    ]
    return cmd
```

### Cleanup After Retrieval
```python
# Source: User decision -- cleanup after Phase 3 has read results
def cleanup_task(self, task_id: int) -> bool:
    """Remove a completed/failed task from tracking. Called after results retrieved."""
    task = self._tasks.get(task_id)
    if not task:
        return False
    if task.status == TaskStatus.RUNNING:
        return False  # Can't cleanup running task

    # Release project lock
    dir_key = str(task.project_dir.resolve())
    if self._project_locks.get(dir_key) == task_id:
        del self._project_locks[dir_key]

    # Remove from tracking (process already reaped, asyncio task already done)
    del self._tasks[task_id]
    return True
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `claude-code-sdk` PyPI package | `claude-agent-sdk` PyPI package | Sept 2025 | Old package deprecated; new SDK wraps CLI subprocess with async streaming |
| `subprocess.run()` for Claude CLI | `asyncio.create_subprocess_exec()` | N/A (our migration) | Enables non-blocking task execution |
| Claude CLI `-c` for session continuity | `--no-session-persistence` for isolation | Available in v2.1.42 | Each task runs fully isolated |
| `--output-format json` for structured output | `--output-format stream-json` for real-time streaming | Available in v2.1.42 | Can parse events as they arrive (but text mode is simpler for our use case) |

**Decision: Raw subprocess vs. Agent SDK:**

The `claude-agent-sdk` (v0.1.36) provides a clean async `query()` API that internally spawns Claude CLI as a subprocess with `--output-format stream-json`. However, for Phase 2 we should use raw `asyncio.create_subprocess_exec()` because:

1. The existing codebase already invokes `claude` directly -- consistency matters.
2. The Agent SDK bundles its own CLI binary and requires `ANTHROPIC_API_KEY` env var, while the existing setup uses a file-based key at `~/.config/openai/api_key`.
3. Raw subprocess gives full control over process group management (`start_new_session=True`, `os.killpg()`), which the SDK abstracts away.
4. Adding a dependency that bundles its own copy of Claude CLI is unnecessary when it's already installed at `~/.local/bin/claude`.
5. The SDK uses `anyio` rather than raw `asyncio`, adding another dependency.

If the project later needs structured event streaming (tool use tracking, thinking blocks), the `--output-format stream-json` flag can be used directly without the SDK.

## Open Questions

1. **Ring buffer size for output cap**
   - What we know: User decided on ring buffer approach with `deque(maxlen=N)`.
   - What's unclear: Optimal maxlen value. 1000 lines is approximately 200KB at 200 chars/line average.
   - Recommendation: Start with 1000 lines. This is easily tunable and can be made configurable later. The output is also persisted to disk on completion, so the ring buffer only needs to hold enough for Phase 3's progress summaries.

2. **Stdout/stderr merge vs. separate**
   - What we know: User left this to Claude's discretion.
   - What's unclear: Whether Phase 3 needs to distinguish error output from normal output.
   - Recommendation: **Merge stderr into stdout** (`stderr=asyncio.subprocess.STDOUT`). This avoids deadlock risks and ordering issues. Claude CLI's stderr contains progress/debug info that isn't useful for task result summaries. If needed later, a `[stderr]` prefix could be added via a more complex two-reader approach.

3. **Event loop lifecycle and TaskManager initialization**
   - What we know: The live session runs its asyncio loop in a daemon thread. TaskManager is a singleton.
   - What's unclear: The exact moment TaskManager should be initialized and how it obtains the event loop reference.
   - Recommendation: TaskManager stores an event loop reference set during its first use. When called from outside the loop thread, use `asyncio.run_coroutine_threadsafe(coro, loop)`. The LiveSession can set the loop reference when it initializes.

4. **Task output format: text vs. stream-json**
   - What we know: Claude CLI supports `--output-format text` (simple) and `--output-format stream-json` (structured events).
   - What's unclear: Whether structured event data (tool use, thinking) adds value in Phase 2.
   - Recommendation: Use `text` for Phase 2. Plain text output is simpler to stream line-by-line and sufficient for Phase 3's voice summaries. If structured data becomes needed, the flag can be changed to `stream-json` and a JSON parser added to the line reader.

## Sources

### Primary (HIGH confidence)
- [Python 3.14 asyncio subprocess docs](https://docs.python.org/3/library/asyncio-subprocess.html) - Process API, PIPE constants, wait/communicate, zombie warnings
- [Python 3.14 asyncio task docs](https://docs.python.org/3/library/asyncio-task.html) - create_task, strong references, background_tasks set pattern, TaskGroup
- [Claude Code CLI help output](claude --help, v2.1.42) - All flags: `-p`, `--no-session-persistence`, `--permission-mode`, `--output-format`, `--add-dir`
- [Claude Code headless mode docs](https://code.claude.com/docs/en/headless) - `-p` usage, `--output-format stream-json`, session management, continuation
- [Claude Agent SDK subprocess_cli.py](https://github.com/anthropics/claude-code-sdk-python/) - Official SDK's subprocess management patterns: command building, output streaming, graceful termination with 5s escalation, 1MB buffer limit, 10MB stderr cap

### Secondary (MEDIUM confidence)
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - SDK capabilities, comparison to CLI
- [Python deque as ring buffer](https://realpython.com/python-deque/) - maxlen behavior, thread safety properties
- [Sling Academy: asyncio stop/kill child process](https://www.slingacademy.com/article/python-asyncio-how-to-stop-kill-a-child-process/) - SIGTERM/SIGKILL patterns

### Tertiary (LOW confidence)
- WebSearch results on process group handling with `start_new_session=True` + `os.killpg()` - multiple sources agree but not verified against official docs for asyncio specifically

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib, verified against Python 3.12 docs
- Architecture: HIGH - Patterns verified from official asyncio docs and Claude SDK source code
- Pitfalls: HIGH - Zombie process, GC, deadlock issues are well-documented in Python asyncio docs
- Claude CLI flags: HIGH - Verified directly from `claude --help` output on the installed v2.1.42

**Research date:** 2026-02-15
**Valid until:** 2026-03-15 (stable -- stdlib APIs and Claude CLI flags unlikely to change)
