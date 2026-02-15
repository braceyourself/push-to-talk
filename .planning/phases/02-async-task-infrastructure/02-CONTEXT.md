# Phase 2: Async Task Infrastructure - Context

**Gathered:** 2026-02-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Build TaskManager and ClaudeTask classes for non-blocking Claude CLI subprocess management. TaskManager spawns, tracks, queries, and cancels isolated Claude CLI subprocesses without blocking the asyncio event loop. Phase 3 wires this into voice control.

</domain>

<decisions>
## Implementation Decisions

### Task output capture
- Stream output in real-time (line-by-line as subprocess produces it)
- Phase 3 can peek at partial output mid-task for progress summaries
- Cap retained output with a ring buffer (last N lines) to prevent memory bloat on verbose tasks
- Persist final output to disk in the task's working directory when task completes
- Reference: GSD plugin's approach of storing artifacts as markdown files in the filesystem

### Task lifecycle & limits
- No limit on concurrent tasks (but one task per project directory — see working directory)
- No timeout — tasks run until completion or manual cancellation
- System (Phase 3) can cancel tasks based on output analysis, not dumb timers
- On failure (non-zero exit or crash): mark status as "failed", retain output for inspection, no auto-retry
- Cleanup after retrieval: process reaped and memory freed after Phase 3 has read and reported results

### Working directory setup
- Tasks run directly in the target project directory (no clones or worktrees)
- One task at a time per project directory to prevent conflicts
- Pass conversation summary as context alongside the project dir so Claude knows WHY it was asked
- Artifacts (output files, logs) left in place after task completion — not cleaned up

### Task identity & naming
- Auto-incrementing integer ID assigned by system
- Human-friendly name attached by Phase 3 (derived from voice request)
- Both ID and name available for referencing tasks
- Metadata tracked per task: id, name, status, process handle, captured output, timestamps (created/started/completed), project directory
- TaskManager is a singleton (one per app, not per session) — tasks persist across sessions until cleaned up

### Async event system
- TaskManager exposes callback hooks for key events (e.g. on_task_complete, on_output_line)
- Phase 3 registers handlers for notifications and voice reporting
- Clean separation between infrastructure (Phase 2) and orchestration (Phase 3)

### Claude's Discretion
- Stdout/stderr handling (merge vs separate)
- Ring buffer size for output cap
- Exact callback/event API design
- Internal data structures for task tracking

</decisions>

<specifics>
## Specific Ideas

- "GSD plugin does this well by storing plans in the filesystem as md files. We could leverage this." — task output should persist as files, similar to GSD's `.planning/` artifacts
- GSD has milestones for large tasks and quick mode for small ones — Phase 3 could dispatch different task "weights" (noted for Phase 3)

</specifics>

<deferred>
## Deferred Ideas

- Task dispatch modes (milestone-style vs quick-style based on complexity) — Phase 3 concern
- Git worktree isolation for concurrent tasks in same project — revisit if one-per-project becomes limiting

</deferred>

---

*Phase: 02-async-task-infrastructure*
*Context gathered: 2026-02-13*
