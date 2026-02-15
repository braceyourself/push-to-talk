"""Async Task Manager for Claude CLI Subprocess Management

Provides TaskManager singleton, ClaudeTask dataclass, and TaskStatus enum
for spawning, tracking, querying, and cancelling Claude CLI subprocesses
without blocking the asyncio event loop.

TaskManager is a singleton that persists across live sessions. Each task
runs Claude CLI in a target project directory with output streamed line-by-line
into a capped ring buffer. One task per project directory is enforced.

Phase 3 wires this into voice control via callback hooks.
"""

import asyncio
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class TaskStatus(Enum):
    """Status of a Claude CLI task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ClaudeTask:
    """Represents a single Claude CLI task with its state and output."""
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


# Claude CLI binary path
CLAUDE_CLI = Path.home() / '.local' / 'bin' / 'claude'


class TaskManager:
    """Singleton manager for async Claude CLI task lifecycle.

    Spawns Claude CLI as async subprocesses, tracks their state,
    streams output line-by-line, and provides clean cancellation
    with SIGTERM/SIGKILL escalation to the process group.

    Usage:
        tm = TaskManager()
        task = await tm.spawn_task("refactor auth", "Refactor the auth module", Path("/project"))
        # task runs in background, query with tm.get_task(task.id)
    """

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
        self._next_id: int = 1
        self._active_tasks: set[asyncio.Task] = set()
        self._project_locks: dict[str, int] = {}
        self._callbacks: dict[str, list[Callable]] = {
            'on_task_complete': [],
            'on_task_failed': [],
            'on_output_line': [],
        }
        print("TaskManager initialized", flush=True)

    def _build_claude_command(self, task: ClaudeTask) -> list[str]:
        """Build the Claude CLI command for a task."""
        cmd = [
            str(CLAUDE_CLI),
            '-p', task.prompt,
            '--no-session-persistence',
            '--permission-mode', 'bypassPermissions',
            '--output-format', 'text',
        ]
        return cmd

    async def spawn_task(self, name: str, prompt: str, project_dir: Path) -> ClaudeTask:
        """Spawn a new Claude CLI task in the given project directory.

        Args:
            name: Human-friendly task name (e.g. "refactor auth module")
            prompt: The prompt to pass to Claude CLI via -p
            project_dir: Directory where Claude CLI runs (cwd)

        Returns:
            ClaudeTask with status PENDING (transitions to RUNNING shortly)

        Raises:
            ValueError: If a task is already running in project_dir
        """
        dir_key = str(project_dir.resolve())

        # Enforce one task per project directory
        if dir_key in self._project_locks:
            existing_id = self._project_locks[dir_key]
            existing = self._tasks.get(existing_id)
            if existing and existing.status == TaskStatus.RUNNING:
                raise ValueError(
                    f"Task {existing_id} ('{existing.name}') already running in {project_dir}"
                )

        task = ClaudeTask(
            id=self._next_id,
            name=name,
            prompt=prompt,
            project_dir=project_dir,
        )
        self._next_id += 1
        self._tasks[task.id] = task
        self._project_locks[dir_key] = task.id

        # Create asyncio task with strong reference to prevent GC
        asyncio_task = asyncio.create_task(
            self._run_task(task),
            name=f"claude-task-{task.id}"
        )
        task._asyncio_task = asyncio_task
        self._active_tasks.add(asyncio_task)
        asyncio_task.add_done_callback(self._active_tasks.discard)

        print(f"TaskManager: spawned task {task.id} '{task.name}' in {project_dir}", flush=True)
        return task

    async def _run_task(self, task: ClaudeTask) -> None:
        """Internal coroutine that runs a Claude CLI subprocess.

        Streams output line-by-line, fires callbacks, and handles
        completion, failure, and cancellation.
        """
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

            cmd = self._build_claude_command(task)

            task.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(task.project_dir),
                start_new_session=True,
            )

            print(f"TaskManager: task {task.id} process started (pid {task.process.pid})", flush=True)

            # Stream output line-by-line
            while True:
                line = await task.process.stdout.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace').rstrip('\n')
                task.output_lines.append(decoded)
                await self._fire_callbacks('on_output_line', task, decoded)

            # Reap the process
            return_code = await task.process.wait()
            task.return_code = return_code
            task.completed_at = time.time()

            if return_code == 0:
                task.status = TaskStatus.COMPLETED
                duration = task.completed_at - (task.started_at or task.created_at)
                print(f"TaskManager: task {task.id} completed in {duration:.1f}s", flush=True)
                await self._fire_callbacks('on_task_complete', task)
            else:
                task.status = TaskStatus.FAILED
                print(f"TaskManager: task {task.id} failed with exit code {return_code}", flush=True)
                await self._fire_callbacks('on_task_failed', task)

            # Persist output to disk
            await self._persist_output(task)

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            print(f"TaskManager: task {task.id} cancelled", flush=True)
            await self._terminate_process(task)
            raise

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.output_lines.append(f"[Internal error: {e}]")
            print(f"TaskManager: task {task.id} internal error: {e}", flush=True)
            await self._fire_callbacks('on_task_failed', task)

        finally:
            # Release project lock
            dir_key = str(task.project_dir.resolve())
            if self._project_locks.get(dir_key) == task.id:
                del self._project_locks[dir_key]

            # Ensure process is reaped (prevent zombies)
            if task.process is not None and task.process.returncode is None:
                try:
                    await task.process.wait()
                except Exception:
                    pass

    async def cancel_task(self, task_id: int) -> bool:
        """Cancel a running task by ID.

        Sends SIGTERM to the process group, escalates to SIGKILL after 5s.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was cancelled, False if not found or not running
        """
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()

        # Cancel the asyncio task
        if task._asyncio_task and not task._asyncio_task.done():
            task._asyncio_task.cancel()

        # Terminate the process directly for immediate effect
        await self._terminate_process(task)

        # Release project lock
        dir_key = str(task.project_dir.resolve())
        if self._project_locks.get(dir_key) == task_id:
            del self._project_locks[dir_key]

        print(f"TaskManager: task {task_id} cancelled", flush=True)
        return True

    async def _terminate_process(self, task: ClaudeTask) -> None:
        """Terminate a task's subprocess gracefully, escalating to SIGKILL.

        Sends SIGTERM to the entire process group (Claude + its children),
        waits up to 5 seconds, then sends SIGKILL if still alive.
        """
        if task.process is None or task.process.returncode is not None:
            return

        try:
            pgid = os.getpgid(task.process.pid)
            os.killpg(pgid, signal.SIGTERM)

            try:
                await asyncio.wait_for(task.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill the process group
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await task.process.wait()
        except ProcessLookupError:
            pass  # Already dead

    # -- Query methods --

    def get_task(self, task_id: int) -> Optional[ClaudeTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def find_task_by_name(self, name: str) -> Optional[ClaudeTask]:
        """Find a task by name (case-insensitive partial match).

        Returns the first match found.
        """
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
        """Get the current output of a task as a string.

        Returns None if task not found, otherwise joins output_lines.
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
        return '\n'.join(task.output_lines)

    # -- Cleanup --

    def cleanup_task(self, task_id: int) -> bool:
        """Remove a completed/failed/cancelled task from tracking.

        Called after Phase 3 has read and reported results.
        Returns False if task not found or still running.
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status == TaskStatus.RUNNING:
            return False

        # Release project lock
        dir_key = str(task.project_dir.resolve())
        if self._project_locks.get(dir_key) == task_id:
            del self._project_locks[dir_key]

        # Remove from tracking
        del self._tasks[task_id]
        return True

    # -- Event system --

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for an event.

        Valid events: on_task_complete, on_task_failed, on_output_line

        Args:
            event: Event name to listen for
            callback: Function called when event fires. May be sync or async.
                      Receives the ClaudeTask as first arg, plus event-specific args.

        Raises:
            ValueError: If event name is not recognized
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}. Valid events: {list(self._callbacks.keys())}")
        self._callbacks[event].append(callback)

    async def _fire_callbacks(self, event: str, *args) -> None:
        """Fire all registered callbacks for an event.

        Supports both sync and async callbacks. Exceptions in callbacks
        are caught and logged -- a bad callback never crashes the task.
        """
        for callback in self._callbacks.get(event, []):
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"TaskManager callback error ({event}): {e}", flush=True)

    # -- Output persistence --

    async def _persist_output(self, task: ClaudeTask) -> None:
        """Write task output to a markdown file in the project directory.

        Creates {project_dir}/.claude-tasks/task-{id}-output.md with
        task metadata and full output.
        """
        try:
            output_dir = task.project_dir / '.claude-tasks'
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f'task-{task.id}-output.md'

            content = f"# Task {task.id}: {task.name}\n\n"
            content += f"**Status:** {task.status.value}\n"
            content += f"**Project:** {task.project_dir}\n"
            content += f"**Created:** {time.ctime(task.created_at)}\n"
            if task.started_at:
                content += f"**Started:** {time.ctime(task.started_at)}\n"
            if task.completed_at:
                duration = task.completed_at - (task.started_at or task.created_at)
                content += f"**Duration:** {duration:.1f}s\n"
            if task.return_code is not None:
                content += f"**Exit Code:** {task.return_code}\n"
            content += f"\n## Output\n\n```\n"
            content += '\n'.join(task.output_lines)
            content += "\n```\n"

            output_file.write_text(content)
            print(f"TaskManager: persisted output for task {task.id} to {output_file}", flush=True)
        except Exception as e:
            print(f"TaskManager: failed to persist output for task {task.id}: {e}", flush=True)
