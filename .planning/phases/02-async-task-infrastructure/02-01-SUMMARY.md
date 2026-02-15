---
phase: 02-async-task-infrastructure
plan: 01
subsystem: task-management
tags: asyncio subprocess singleton lifecycle deque ring-buffer process-group

requires:
  - phase: 01-mode-rename-and-live-voice-session
    provides: live session architecture and asyncio event loop
provides:
  - TaskManager singleton for async Claude CLI subprocess management
  - ClaudeTask dataclass with status tracking and output buffering
  - Callback system for task lifecycle events (complete, failed, output_line)
affects:
  - phase: 03-voice-controlled-task-orchestration

tech-stack:
  added: []
  patterns:
    - "Singleton with __new__ + _initialized guard"
    - "asyncio.create_task with strong reference set (_active_tasks)"
    - "Line-by-line subprocess output streaming via readline()"
    - "Process group termination with SIGTERM/SIGKILL escalation"
    - "deque(maxlen=1000) ring buffer for output capping"

key-files:
  created:
    - task_manager.py
    - test_task_manager.py
  modified: []

key-decisions:
  - "All stdlib, no new dependencies -- asyncio, collections.deque, dataclasses, enum, pathlib, signal, os"
  - "stderr merged into stdout (STDOUT redirect) to avoid deadlock and simplify streaming"
  - "Ring buffer maxlen=1000 (~200KB cap) with disk persistence on completion"
  - "Monkey-patch _build_claude_command in tests to avoid real Claude CLI calls"

duration: 3min
completed: 2026-02-15
---

# Phase 2 Plan 1: TaskManager and ClaudeTask Summary

**TaskManager singleton with async subprocess lifecycle, one-per-project enforcement, SIGTERM/SIGKILL escalation, and deque ring buffer output streaming**

## Performance

- **Duration:** 3 minutes
- **Started:** 2026-02-15T13:39:47Z
- **Completed:** 2026-02-15T13:42:34Z
- **Tasks:** 2/2
- **Files created:** 2

## Accomplishments

- Built TaskManager singleton class (404 lines) with full async subprocess lifecycle management
- Implemented ClaudeTask dataclass with 12 fields including deque ring buffer for output capping
- One-per-project enforcement prevents concurrent Claude CLI tasks in the same directory
- SIGTERM/SIGKILL escalation via os.killpg ensures clean process group termination
- Strong reference set (_active_tasks) with add_done_callback prevents asyncio task GC
- Callback system supports on_task_complete, on_task_failed, on_output_line events
- Output persistence writes task results to .claude-tasks/task-{id}-output.md
- 10 integration tests (373 lines) prove full lifecycle with zero zombie processes

## Task Commits

1. **Task 1: Create task_manager.py with TaskManager, ClaudeTask, and full async lifecycle** - `9701521` (feat)
2. **Task 2: Create integration test proving full task lifecycle** - `3eecee0` (test)

## Files Created/Modified

### Created
- `task_manager.py` -- TaskStatus enum, ClaudeTask dataclass, TaskManager singleton (404 lines)
- `test_task_manager.py` -- 10 integration tests with monkey-patched commands (373 lines)

### Modified
None.

## Decisions Made

1. **All stdlib, zero new dependencies** -- asyncio, collections.deque, dataclasses, enum, pathlib, signal, os cover all requirements
2. **stderr merged into stdout** -- `stderr=asyncio.subprocess.STDOUT` avoids deadlock risks from separate pipes
3. **Ring buffer maxlen=1000** -- ~200KB cap; output also persisted to disk on completion for full history
4. **Test isolation via monkey-patch** -- `_build_claude_command` replaced with test commands to avoid real CLI calls and API costs
5. **Singleton reset between tests** -- `TaskManager._instance = None` allows clean state per test

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

Phase 3 (Voice-Controlled Task Orchestration) can now:
- Import `TaskManager` and `ClaudeTask` from `task_manager.py`
- Call `spawn_task()` from within the asyncio event loop used by LiveSession
- Register callbacks via `tm.on('on_task_complete', handler)` for voice notifications
- Query task state via `get_task()`, `find_task_by_name()`, `get_running_tasks()`
- Cancel tasks via `cancel_task()` and clean up via `cleanup_task()`

No blockers. TaskManager is a standalone module with no dependencies on existing code.
