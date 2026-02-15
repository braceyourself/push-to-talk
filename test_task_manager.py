#!/usr/bin/env python3
"""Integration tests for TaskManager async subprocess lifecycle.

Runs 10 tests proving spawn, track, cancel, cleanup, callbacks,
and output persistence. Uses simple shell commands instead of
Claude CLI to test the async machinery without API calls.

Run: python3 test_task_manager.py
Exit code: 0 if all pass, 1 if any fail.
"""

import asyncio
import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from task_manager import TaskManager, ClaudeTask, TaskStatus


def reset_task_manager():
    """Reset TaskManager singleton between tests."""
    TaskManager._instance = None


def patch_command(tm, cmd):
    """Monkey-patch TaskManager to use a test command instead of Claude CLI."""
    tm._build_claude_command = lambda task: cmd


# -- Test functions --

async def test_spawn_and_complete():
    """Spawn a task, wait for completion, verify final state."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    cmd = [
        sys.executable, '-c',
        'import time\n'
        'for i in range(5):\n'
        '    print(f"line {i}", flush=True)\n'
        '    time.sleep(0.05)\n'
        'print("done", flush=True)\n'
    ]
    patch_command(tm, cmd)

    task = await tm.spawn_task("test task", "test prompt", tmpdir)
    assert task.status in (TaskStatus.PENDING, TaskStatus.RUNNING), \
        f"Expected PENDING or RUNNING, got {task.status}"
    assert task.id == 1

    # Wait for completion
    while task.status == TaskStatus.PENDING or task.status == TaskStatus.RUNNING:
        await asyncio.sleep(0.1)

    assert task.status == TaskStatus.COMPLETED, f"Expected COMPLETED, got {task.status}"
    assert task.return_code == 0, f"Expected return_code 0, got {task.return_code}"
    assert task.started_at is not None, "started_at should be set"
    assert task.completed_at is not None, "completed_at should be set"
    assert len(task.output_lines) == 6, f"Expected 6 output lines, got {len(task.output_lines)}"
    assert "done" in task.output_lines[-1], f"Last line should contain 'done', got '{task.output_lines[-1]}'"

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_singleton():
    """Verify TaskManager returns the same instance."""
    tm1 = TaskManager()
    tm2 = TaskManager()
    assert tm1 is tm2, "TaskManager should be a singleton"


async def test_one_per_project():
    """Spawn in dir A, try again in A (fail), spawn in B (succeed)."""
    tm = TaskManager()
    dir_a = Path(tempfile.mkdtemp())
    dir_b = Path(tempfile.mkdtemp())

    # Long-running command so task stays RUNNING
    cmd = [sys.executable, '-c', 'import time; time.sleep(30)']
    patch_command(tm, cmd)

    task_a = await tm.spawn_task("task in A", "prompt", dir_a)

    # Wait for it to start running
    while task_a.status == TaskStatus.PENDING:
        await asyncio.sleep(0.05)

    # Try to spawn another in dir A -- should raise
    try:
        await tm.spawn_task("another in A", "prompt", dir_a)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already running" in str(e).lower(), f"Expected 'already running' in error, got: {e}"

    # Spawn in dir B -- should succeed
    task_b = await tm.spawn_task("task in B", "prompt", dir_b)
    assert task_b.id != task_a.id, "Tasks should have different IDs"

    # Cleanup
    await tm.cancel_task(task_a.id)
    await tm.cancel_task(task_b.id)
    await asyncio.sleep(0.5)

    shutil.rmtree(dir_a, ignore_errors=True)
    shutil.rmtree(dir_b, ignore_errors=True)


async def test_cancel_task():
    """Spawn a long-running task, cancel it, verify cleanup."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    cmd = [sys.executable, '-c', 'import time; time.sleep(30)']
    patch_command(tm, cmd)

    task = await tm.spawn_task("long task", "prompt", tmpdir)

    # Wait for process to start
    while task.status == TaskStatus.PENDING:
        await asyncio.sleep(0.05)

    assert task.status == TaskStatus.RUNNING

    pid = task.process.pid

    # Cancel
    result = await tm.cancel_task(task.id)
    assert result is True, "cancel_task should return True"

    # Allow time for process cleanup
    await asyncio.sleep(0.5)

    assert task.status == TaskStatus.CANCELLED, f"Expected CANCELLED, got {task.status}"

    # Verify process is dead (not zombie)
    try:
        os.kill(pid, 0)
        assert False, f"Process {pid} should be dead"
    except ProcessLookupError:
        pass  # Expected -- process is gone

    # Verify project lock is released
    dir_key = str(tmpdir.resolve())
    assert dir_key not in tm._project_locks, "Project lock should be released after cancel"

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_callbacks():
    """Register on_task_complete callback, verify it fires."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    completed_tasks = []

    def on_complete(task):
        completed_tasks.append(task)

    tm.on('on_task_complete', on_complete)

    cmd = [sys.executable, '-c', 'print("hello", flush=True)']
    patch_command(tm, cmd)

    task = await tm.spawn_task("callback test", "prompt", tmpdir)

    # Wait for completion
    while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        await asyncio.sleep(0.1)

    assert len(completed_tasks) == 1, f"Expected 1 callback call, got {len(completed_tasks)}"
    assert completed_tasks[0].id == task.id, "Callback should receive the completed task"

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_output_streaming():
    """Register on_output_line callback, verify lines arrive."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    collected_lines = []

    def on_line(task, line):
        collected_lines.append(line)

    tm.on('on_output_line', on_line)

    cmd = [
        sys.executable, '-c',
        'for i in range(5): print(f"stream-{i}", flush=True)\n'
    ]
    patch_command(tm, cmd)

    task = await tm.spawn_task("stream test", "prompt", tmpdir)

    while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        await asyncio.sleep(0.1)

    assert len(collected_lines) == 5, f"Expected 5 lines, got {len(collected_lines)}"
    assert collected_lines[0] == "stream-0", f"First line should be 'stream-0', got '{collected_lines[0]}'"
    assert collected_lines[4] == "stream-4", f"Last line should be 'stream-4', got '{collected_lines[4]}'"

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_query_methods():
    """Test get_task, find_task_by_name, get_all_tasks, get_running_tasks."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    cmd = [sys.executable, '-c', 'import time; time.sleep(2)']
    patch_command(tm, cmd)

    task = await tm.spawn_task("query target", "prompt", tmpdir)

    # Wait for running
    while task.status == TaskStatus.PENDING:
        await asyncio.sleep(0.05)

    # get_task
    found = tm.get_task(task.id)
    assert found is not None, "get_task should find the task"
    assert found.id == task.id

    # find_task_by_name (partial, case-insensitive)
    found = tm.find_task_by_name("query")
    assert found is not None, "find_task_by_name should find partial match"
    assert found.id == task.id

    found = tm.find_task_by_name("QUERY TARGET")
    assert found is not None, "find_task_by_name should be case-insensitive"

    # get_all_tasks
    all_tasks = tm.get_all_tasks()
    assert len(all_tasks) >= 1, "get_all_tasks should return at least 1 task"
    assert any(t.id == task.id for t in all_tasks)

    # get_running_tasks
    running = tm.get_running_tasks()
    assert any(t.id == task.id for t in running), "get_running_tasks should include our task"

    # get_task_output
    output = tm.get_task_output(task.id)
    assert output is not None, "get_task_output should return something"

    # Cleanup
    await tm.cancel_task(task.id)
    await asyncio.sleep(0.5)

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_cleanup():
    """Spawn task, wait for completion, cleanup, verify removed."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    cmd = [sys.executable, '-c', 'print("cleanup test", flush=True)']
    patch_command(tm, cmd)

    task = await tm.spawn_task("cleanup target", "prompt", tmpdir)

    while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        await asyncio.sleep(0.1)

    task_id = task.id

    result = tm.cleanup_task(task_id)
    assert result is True, "cleanup_task should return True for completed task"

    assert tm.get_task(task_id) is None, "get_task should return None after cleanup"

    # Verify project lock released
    dir_key = str(tmpdir.resolve())
    assert dir_key not in tm._project_locks, "Project lock should be released after cleanup"

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_output_persistence():
    """Spawn task, wait for completion, verify output file exists."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    cmd = [sys.executable, '-c', 'print("persisted output", flush=True)']
    patch_command(tm, cmd)

    task = await tm.spawn_task("persist test", "prompt", tmpdir)

    while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        await asyncio.sleep(0.1)

    output_file = tmpdir / '.claude-tasks' / f'task-{task.id}-output.md'
    assert output_file.exists(), f"Output file should exist at {output_file}"

    content = output_file.read_text()
    assert "persisted output" in content, "Output file should contain task output"
    assert "persist test" in content, "Output file should contain task name"
    assert "completed" in content.lower(), "Output file should show completed status"

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_failed_task():
    """Spawn a task that exits non-zero, verify FAILED status."""
    tm = TaskManager()
    tmpdir = Path(tempfile.mkdtemp())

    cmd = [sys.executable, '-c', 'import sys; print("error output", flush=True); sys.exit(1)']
    patch_command(tm, cmd)

    task = await tm.spawn_task("fail test", "prompt", tmpdir)

    while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        await asyncio.sleep(0.1)

    assert task.status == TaskStatus.FAILED, f"Expected FAILED, got {task.status}"
    assert task.return_code == 1, f"Expected return_code 1, got {task.return_code}"
    assert len(task.output_lines) > 0, "Failed task should still have output"

    shutil.rmtree(tmpdir, ignore_errors=True)


# -- Runner --

async def run_tests():
    """Run all tests sequentially, resetting singleton between each."""
    tests = [
        ("test_spawn_and_complete", test_spawn_and_complete),
        ("test_singleton", test_singleton),
        ("test_one_per_project", test_one_per_project),
        ("test_cancel_task", test_cancel_task),
        ("test_callbacks", test_callbacks),
        ("test_output_streaming", test_output_streaming),
        ("test_query_methods", test_query_methods),
        ("test_cleanup", test_cleanup),
        ("test_output_persistence", test_output_persistence),
        ("test_failed_task", test_failed_task),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        reset_task_manager()
        try:
            await test_fn()
            print(f"{name}: PASS", flush=True)
            passed += 1
        except Exception as e:
            print(f"{name}: FAIL -- {e}", flush=True)
            failed += 1
            failures.append((name, str(e)))

    print(f"\n{passed}/{len(tests)} tests passed", flush=True)
    if failures:
        print("\nFailures:", flush=True)
        for name, err in failures:
            print(f"  {name}: {err}", flush=True)

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
