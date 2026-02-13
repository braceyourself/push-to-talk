# Stack Research: Async Task Orchestrator for Voice-Controlled Claude CLI

**Domain:** Voice-controlled async task orchestration (Python desktop app)
**Researched:** 2026-02-13
**Confidence:** HIGH

## Executive Summary

The existing push-to-talk app already runs an asyncio event loop in a dedicated thread for the OpenAI Realtime WebSocket session. The task orchestrator must live inside that same event loop, using `asyncio.create_subprocess_exec()` to spawn Claude CLI processes and `asyncio.Queue` to report status back to the Realtime session's tool-calling layer. No new frameworks are needed. The entire orchestrator can be built with Python 3.12 stdlib (`asyncio`, `dataclasses`, `enum`, `uuid`, `json`) plus one small library (`janus`) for the thread-bridge queue.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python `asyncio` (stdlib) | 3.12 (installed) | Async event loop, subprocess management, task coordination | Already the foundation of RealtimeSession; create_subprocess_exec provides non-blocking process spawning with stdout/stderr streaming. Avoids adding concurrency frameworks on top of what's already running. |
| Python `dataclasses` (stdlib) | 3.12 | Task state modeling (TaskRecord, TaskResult) | Zero-dependency, slots=True for memory efficiency, field(default_factory=...) for auto-generated IDs. Already used pattern in the codebase (InterviewSession, ConversationSession follow similar patterns). |
| Python `enum.StrEnum` (stdlib) | 3.12 | Task lifecycle states (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED) | StrEnum (added 3.11) lets states be compared as strings for easy serialization to JSON tool responses. Cleaner than raw strings, no third-party state machine library needed for this simple lifecycle. |
| Python `uuid` (stdlib) | 3.12 | Unique task identifiers | uuid4() provides collision-free task IDs without external state. Tasks need stable references for voice interactions ("check on the first task", "what's task X doing"). |
| Claude CLI | 2.1.41 (installed) | Background task execution engine | Already integrated. Key flags for orchestration: `-p` (print/non-interactive), `--output-format stream-json` (NDJSON progress), `--permission-mode`, `--max-turns`, `--max-budget-usd`, `--session-id`, `--add-dir`. |
| OpenAI Realtime API | gpt-4o-realtime-preview / gpt-realtime (GA) | Voice interface with function calling | Already integrated. The `gpt-realtime` GA model adds async function calling -- the model can continue talking while tool calls are pending. This is the key enabler for "spawn a task and keep talking". |

**Confidence: HIGH** -- All core technologies are stdlib or already installed. Verified against Python 3.12 docs and existing codebase.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `janus` | 2.0.0 | Thread-safe asyncio-aware queue | Bridge between pynput keyboard thread (sync) and asyncio event loop. Needed because hotkey events arrive in a threading callback but task orchestrator lives in asyncio. The existing codebase already faces this boundary (see `request_interrupt()` pattern in openai_realtime.py line 506). janus solves it cleanly instead of flag-polling. |
| `websockets` | 10.4 (installed) / 16.0 (latest) | WebSocket client for Realtime API | Already installed at 10.4. Works fine. Upgrading to 16.0 is optional; only needed if hitting bugs. Latest version requires Python >= 3.10. |
| `openai` | 2.17.0 (installed) | TTS API calls | Already installed. Not directly used by the orchestrator but used by the TTS layer that speaks task results. |

**Confidence: HIGH for janus** -- Well-maintained by aio-libs (same org as aiohttp), production/stable status, 2.0.0 released Dec 2024. Verified via PyPI and GitHub. The codebase already has the thread-to-async bridging problem (flag-based workaround in openai_realtime.py), and janus is the standard solution.

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `journalctl --user -u push-to-talk -f` | Log monitoring | Existing pattern. All orchestrator events should log via `print(..., flush=True)` for systemd journal capture. |
| Claude CLI `--output-format stream-json` | Task progress streaming | Parse NDJSON lines from Claude's stdout for real-time progress. Each line is `{"type": "message"|"tool_use"|"tool_result"|"result", ...}`. Use `asyncio.StreamReader.readline()` on the subprocess stdout pipe. |
| `--verbose` flag on Claude CLI | Debug task execution | Shows full turn-by-turn output. Use during development, disable in production. |

## Installation

```bash
# Only new dependency
pip install janus

# Optional: upgrade websockets (not required, 10.4 works)
# pip install --upgrade websockets
```

The requirements.txt should add:
```
janus>=2.0.0
```

## Architecture Decision: Why NOT Use External Frameworks

The task orchestrator is architecturally simple: spawn processes, track state, report results. It does NOT need:

- **Celery/Dramatiq/TaskIQ** -- These are distributed task queues for multi-machine deployments. This is a single-user desktop app. Adding a message broker (Redis/RabbitMQ) for local subprocess management is massive overengineering.
- **python-statemachine/transitions** -- The task lifecycle is linear (PENDING -> RUNNING -> COMPLETED/FAILED). A simple StrEnum with explicit transitions in the TaskManager methods is clearer than a state machine DSL for 5 states.
- **anyio/trio** -- These are alternative async runtimes. The app already uses asyncio. Introducing another runtime creates compatibility headaches with the existing RealtimeSession.
- **multiprocessing** -- Claude CLI is a separate process (Node.js). We need subprocess management, not Python multiprocessing. `asyncio.create_subprocess_exec()` is the right tool.
- **concurrent.futures** -- `ProcessPoolExecutor` and `ThreadPoolExecutor` are for CPU-bound Python work. We're spawning external processes and reading their output. asyncio subprocess APIs are more appropriate and integrate natively with the event loop.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `asyncio.create_subprocess_exec()` | `subprocess.Popen` + threading | Never for this use case. The asyncio loop already exists. Popen would require a separate thread per task for non-blocking reads, recreating what asyncio gives for free. |
| `janus` queue | `asyncio.get_event_loop().call_soon_threadsafe()` | If you only need to schedule a callback from a thread into asyncio (one-way). janus is better because we need bidirectional communication: thread puts events, asyncio reads them AND asyncio puts status updates, thread reads them. |
| `janus` queue | Flag-polling (current `_interrupt_requested` pattern) | Only for simple boolean signals. Breaks down for structured data like "spawn task X with prompt Y". The existing flag pattern in RealtimeSession works for a single interrupt signal but does not scale to task commands. |
| `dataclasses` for state | SQLite / TinyDB persistence | Only if task state must survive process restarts. PROJECT.md explicitly says "Persistent task state across live mode sessions" is out of scope. In-memory dataclasses are sufficient. |
| `StrEnum` for states | `python-statemachine` library | Only if lifecycle has complex branching, guards, or callbacks. Our lifecycle is linear with one branch point (COMPLETED vs FAILED). StrEnum is simpler and has zero dependencies. |
| `--output-format stream-json` | `--output-format json` (wait for completion) | If you don't need progress updates during execution. Stream-json gives real-time NDJSON lines so the orchestrator can report "Claude is editing files" while the task runs. Regular json only returns after Claude finishes entirely. |
| `gpt-realtime` (GA model) | `gpt-4o-realtime-preview` (current) | The preview model works and is already in use. The GA model adds native async function calling (model talks while tool executes). Worth upgrading to when implementing the orchestrator, but the preview model can be made to work with application-level async patterns. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `subprocess.Popen` for Claude CLI tasks | Blocking. Would require one thread per task for non-blocking stdout reads. The asyncio loop is right there. | `asyncio.create_subprocess_exec()` |
| `threading.Thread` per Claude task | Unnecessary complexity. The asyncio event loop can manage hundreds of concurrent subprocess coroutines without threads. | `asyncio.create_task()` wrapping a subprocess coroutine |
| `os.system()` or `subprocess.run()` | Fully blocking. Freezes the event loop or requires a thread. | `asyncio.create_subprocess_exec()` |
| Celery / Redis / RabbitMQ | Enterprise distributed task queue. This is a single-machine desktop app managing ~3-10 concurrent Claude processes. | In-memory `dict[str, TaskRecord]` + asyncio |
| `asyncio.TaskGroup` (Python 3.11+) | TaskGroup enforces structured concurrency: ALL tasks must complete before the context manager exits. We need tasks to start/complete independently. A cancelled task should not cancel all others. | Individual `asyncio.create_task()` calls tracked in a dict |
| Global mutable state for task tracking | Hard to reason about, race-condition prone if accessed from multiple threads | A single `TaskManager` class that owns all state, accessed only from the asyncio thread |
| `pickle` for task serialization | Security risk, unnecessary. Task state is simple enough for JSON. | `dataclasses.asdict()` + `json.dumps()` |

## Stack Patterns

**Pattern: Async subprocess with streaming output**
```python
async def run_claude_task(task_id: str, prompt: str, cwd: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        str(CLAUDE_CLI), '-p', prompt,
        '--output-format', 'stream-json',
        '--permission-mode', 'acceptEdits',
        '--max-turns', '50',
        '--no-session-persistence',
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    output_lines = []
    async for line in proc.stdout:
        data = json.loads(line)
        if data.get('type') == 'result':
            output_lines.append(data.get('result', ''))
        # Could emit progress events here

    await proc.wait()
    return '\n'.join(output_lines)
```

**Pattern: Thread-to-asyncio bridge with janus**
```python
import janus

# In asyncio setup:
command_queue: janus.Queue[dict] = janus.Queue()

# From pynput thread (sync side):
command_queue.sync_q.put({"action": "spawn", "prompt": "Fix the tests"})

# In asyncio consumer (async side):
async def process_commands():
    while True:
        cmd = await command_queue.async_q.get()
        if cmd["action"] == "spawn":
            await task_manager.spawn(cmd["prompt"])
        command_queue.async_q.task_done()
```

**Pattern: Task state as dataclass**
```python
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4
from datetime import datetime

class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskRecord:
    task_id: str = field(default_factory=lambda: str(uuid4())[:8])
    prompt: str = ""
    state: TaskState = TaskState.PENDING
    cwd: str = ""
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    pid: int = 0
    _process: asyncio.subprocess.Process | None = field(
        default=None, repr=False, compare=False
    )
```

**Pattern: OpenAI Realtime tool returning async acknowledgment**
```python
# When Realtime API calls spawn_task tool:
async def handle_spawn_task(arguments: dict) -> str:
    task = await task_manager.spawn(
        prompt=arguments["prompt"],
        cwd=arguments.get("directory", str(Path.home())),
    )
    # Return immediately -- don't wait for Claude to finish
    return json.dumps({
        "task_id": task.task_id,
        "status": "spawned",
        "message": f"Task '{task.task_id}' started. Ask me about its status anytime."
    })
```

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| Python 3.12 | asyncio.create_subprocess_exec, StrEnum, dataclasses(slots=True), TaskGroup | All features available. StrEnum requires 3.11+. slots=True requires 3.10+. |
| janus 2.0.0 | Python >= 3.9 | No compatibility issues with 3.12. |
| websockets 10.4 | Python 3.7+ | Installed version works fine. 16.0 requires >= 3.10. |
| Claude CLI 2.1.41 | `--output-format stream-json`, `--session-id`, `--max-budget-usd` | All flags verified against current CLI reference docs. |
| OpenAI Realtime API | `gpt-4o-realtime-preview-2024-12-17` (current) / `gpt-realtime` (GA) | Both support function calling. GA model adds native async function calling. |

## Key Integration Points with Existing Codebase

The orchestrator must integrate at these specific points:

1. **RealtimeSession (openai_realtime.py)** -- Add new tool definitions (spawn_task, check_task, list_tasks, cancel_task, get_result). The `execute_tool()` function at line 155 dispatches tool calls; new tools go here.

2. **Event loop thread (push-to-talk.py line 789-800)** -- The `run_session()` function creates a new asyncio event loop in a thread. The TaskManager must be initialized in this same loop so its async methods work.

3. **Tool execution (openai_realtime.py line 382)** -- Currently `execute_tool()` is synchronous. For async tool handlers (spawning tasks), either:
   - Make `execute_tool()` async (preferred, since it's called from an async context), or
   - Use `asyncio.get_event_loop().run_until_complete()` within it (hack, avoid).

4. **Context isolation** -- Each Claude CLI invocation gets its own `--add-dir` and `cwd`. No shared session directory between tasks. The `--no-session-persistence` flag prevents session files from accumulating.

## Claude CLI Flags for Task Orchestration

| Flag | Value | Purpose |
|------|-------|---------|
| `-p` | `"<task prompt>"` | Non-interactive print mode. Required for headless subprocess invocation. |
| `--output-format` | `stream-json` | NDJSON progress stream. Enables real-time status reporting. |
| `--permission-mode` | `acceptEdits` or `bypassPermissions` | Controls what Claude can do. `acceptEdits` is safer default; `bypassPermissions` for trusted tasks. |
| `--max-turns` | `50` (configurable) | Prevent runaway tasks. Safety limit on how many tool calls Claude makes. |
| `--max-budget-usd` | `5.00` (configurable) | Cost safety limit per task. |
| `--no-session-persistence` | (flag) | Don't save session to disk. Tasks are ephemeral. |
| `--add-dir` | `<project directory>` | Give Claude access to additional directories beyond cwd. |
| `--model` | `sonnet` (default) | Use Sonnet for most tasks (fast, capable). Can be overridden per task. |
| `--append-system-prompt` | `"<context>"` | Inject task-specific context without replacing default behavior. |

## Sources

- [Python 3.12 asyncio subprocess docs](https://docs.python.org/3.12/library/asyncio-subprocess.html) -- Verified create_subprocess_exec API, Process class methods (HIGH confidence)
- [Python 3.12 dataclasses docs](https://docs.python.org/3/library/dataclasses.html) -- Verified frozen, slots, field(default_factory=...) (HIGH confidence)
- [Python 3.12 enum docs (StrEnum)](https://docs.python.org/3/library/enum.html) -- Verified StrEnum availability in 3.11+ (HIGH confidence)
- [janus 2.0.0 on PyPI](https://pypi.org/project/janus/) -- Verified version, API, thread-safe design (HIGH confidence)
- [janus GitHub](https://github.com/aio-libs/janus) -- Verified sync_q/async_q pattern, aclose() requirement (HIGH confidence)
- [Claude CLI reference](https://code.claude.com/docs/en/cli-reference) -- Verified all flags: -p, --output-format, --permission-mode, --max-turns, --max-budget-usd, --session-id, --no-session-persistence, --add-dir (HIGH confidence)
- [websockets 16.0 on PyPI](https://pypi.org/project/websockets/) -- Verified latest version, Python >= 3.10 requirement (HIGH confidence)
- [OpenAI Realtime API docs](https://platform.openai.com/docs/guides/realtime) -- gpt-realtime GA model with async function calling (MEDIUM confidence -- docs returned 403, relied on search results and community posts)
- [OpenAI community: long function calls](https://community.openai.com/t/long-function-calls-and-realtime-api/1119021) -- Async function calling patterns, workarounds (MEDIUM confidence)
- [OpenAI community: async tool calling](https://community.openai.com/t/disabling-asynchronous-tool-calling-with-gpt-realtime/1360261) -- gpt-realtime async behavior confirmed (MEDIUM confidence)
- Existing codebase analysis: `openai_realtime.py`, `push-to-talk.py` -- Direct code inspection (HIGH confidence)

---
*Stack research for: Voice-controlled async task orchestrator*
*Researched: 2026-02-13*
