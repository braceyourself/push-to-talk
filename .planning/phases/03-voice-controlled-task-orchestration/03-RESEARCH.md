# Phase 3: Voice-Controlled Task Orchestration - Research

**Researched:** 2026-02-15
**Domain:** OpenAI Realtime API tool calling, async task orchestration, voice UX for background process management
**Confidence:** HIGH

## Summary

This phase wires the Phase 2 TaskManager into the Phase 1 LiveSession so users can spawn, query, cancel, and receive results from Claude CLI tasks through natural voice conversation. The core work is: (1) defining Realtime API tool schemas for task management operations, (2) adding async tool execution handlers to LiveSession's event loop, (3) implementing a callback-driven notification system for task completion/failure that injects spoken summaries into the conversation at natural pauses, and (4) building ambient task awareness so the AI always knows what tasks are running.

The existing codebase has all the building blocks. `openai_realtime.py` already demonstrates the complete Realtime API tool calling flow: tools defined in `session.update`, function calls detected in `response.done` events, results returned via `conversation.item.create` with type `function_call_output`, and new responses triggered via `response.create`. The `live_session.py` already has the WebSocket connection, audio handling, and conversation state management. The `task_manager.py` already has spawn, cancel, query, output retrieval, and callback hooks (`on_task_complete`, `on_task_failed`, `on_output_line`).

The integration points are clear: LiveSession's `session.update` call (line 96-113 of `live_session.py`) needs tools added, its `handle_events()` method needs a `function_call` handler in the `response.done` branch, and TaskManager callbacks need to inject completion notifications into the WebSocket conversation.

**Primary recommendation:** Add 5-6 tool definitions to LiveSession (`spawn_task`, `get_task_status`, `cancel_task`, `get_task_result`, `list_tasks`), handle them asynchronously in the `response.done` event handler (following the proven pattern from `openai_realtime.py` lines 363-399), and use TaskManager's existing callback hooks to inject `conversation.item.create` system messages followed by `response.create` to deliver spoken task notifications.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `live_session.py` (existing) | Phase 1 | WebSocket voice session with Realtime API | Already running, has tool slot (`tools: []`), needs tools added |
| `task_manager.py` (existing) | Phase 2 | Async Claude CLI subprocess lifecycle | Already has spawn, cancel, query, callbacks; zero new deps |
| `openai_realtime.py` (existing) | Phase 1 | Reference implementation for Realtime API tool calling | Proven tool call pattern to replicate |
| `asyncio` (stdlib) | Python 3.12 | Event loop integration between LiveSession and TaskManager | Both already use asyncio; same event loop in daemon thread |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `json` (stdlib) | Python 3.12 | WebSocket message serialization, tool argument parsing | Every tool call and response |
| `pathlib` (stdlib) | Python 3.12 | Project directory resolution for task spawning | When user specifies project paths |
| `time` (stdlib) | Python 3.12 | Task duration tracking for status summaries | Human-readable elapsed time in status |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Tools in LiveSession `handle_events()` | Separate tool handler class | Unnecessary abstraction for 5-6 tools; keep co-located with WebSocket event handling |
| `response.done` for function call detection | `response.output_item.done` per-item event | Best practice is `response.output_item.done` for granular handling, but the existing codebase uses `response.done` successfully; either works; consistency with existing code wins |
| TaskManager callbacks for notifications | Polling TaskManager periodically | Callbacks are event-driven and already implemented; polling adds latency and complexity |
| `conversation.item.create` system message for notifications | `response.create` with per-response instructions | System message injection is the established pattern for context injection; per-response instructions would override the personality prompt |

**Installation:**
```bash
# No new dependencies. Everything is stdlib + existing modules.
```

## Architecture Patterns

### Integration Point Map
```
push-to-talk.py
  └── start_live_session()
        └── LiveSession(api_key, voice, on_status)  [live_session.py]
              ├── connect()
              │     └── session.update { tools: TASK_TOOLS, tool_choice: "auto" }
              ├── handle_events()
              │     └── response.done → function_call → execute_tool_async()
              │           ├── spawn_task → TaskManager().spawn_task()
              │           ├── get_task_status → TaskManager().get_all_tasks()
              │           ├── get_task_result → TaskManager().get_task_output()
              │           ├── cancel_task → TaskManager().cancel_task()
              │           └── list_tasks → TaskManager().get_all_tasks()
              └── _on_task_event(task)  [callback from TaskManager]
                    └── conversation.item.create + response.create
                          → AI speaks task completion summary

task_manager.py (unchanged)
  └── TaskManager singleton
        ├── spawn_task() → ClaudeTask
        ├── cancel_task() → bool
        ├── get_task() → ClaudeTask
        ├── get_all_tasks() → list[ClaudeTask]
        ├── get_task_output() → str
        └── Callbacks: on_task_complete, on_task_failed
```

### Pattern 1: Realtime API Tool Definitions for Task Management
**What:** JSON tool schemas passed in `session.update` that expose task management operations to the AI.
**When to use:** At session initialization and whenever tools need updating.
**Example:**
```python
# Source: openai_realtime.py TOOLS pattern + Realtime API docs
TASK_TOOLS = [
    {
        "type": "function",
        "name": "spawn_task",
        "description": (
            "Start a Claude CLI task running in the background. "
            "Use when the user asks you to do real work like refactoring, writing code, "
            "fixing bugs, or any task that benefits from Claude's deep capabilities. "
            "The task runs asynchronously - acknowledge it briefly and continue the conversation. "
            "Keep your acknowledgment to one sentence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short descriptive name for this task (2-4 words, e.g. 'auth refactor', 'fix tests')"
                },
                "prompt": {
                    "type": "string",
                    "description": "The detailed prompt/instructions for Claude CLI"
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to the project directory where Claude should work"
                }
            },
            "required": ["name", "prompt", "project_dir"]
        }
    },
    {
        "type": "function",
        "name": "list_tasks",
        "description": (
            "Get status of all tasks (running, completed, failed). "
            "Use when the user asks what tasks are doing, or wants a status update. "
            "Summarize conversationally - every word takes time to speak."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "get_task_result",
        "description": (
            "Get the output/result of a specific task. "
            "Use when the user asks what a task produced or wants details about a completed task. "
            "Summarize the output concisely - do not read it verbatim."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or task number"
                }
            },
            "required": ["task_identifier"]
        }
    },
    {
        "type": "function",
        "name": "cancel_task",
        "description": (
            "Cancel a running task immediately. No confirmation needed. "
            "Use when the user asks to stop or cancel a task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or task number"
                }
            },
            "required": ["task_identifier"]
        }
    },
]
```

### Pattern 2: Async Tool Execution in handle_events()
**What:** Function call handling that runs async operations (TaskManager calls) without blocking the WebSocket event loop.
**When to use:** In the `response.done` event handler when output items include `function_call` type.
**Example:**
```python
# Source: Adapted from openai_realtime.py lines 363-399
elif event_type == "response.done":
    response = data.get("response", {})
    output_items = response.get("output", [])

    for item in output_items:
        if item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = json.loads(item.get("arguments", "{}"))

            # Execute tool asynchronously
            result = await self._execute_tool(name, arguments)

            # Send result back to conversation
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result)
                }
            }))

            # Trigger AI to speak the result
            await self.ws.send(json.dumps({
                "type": "response.create"
            }))

    # ... existing response.done handling (transcript tracking, etc.)
```

### Pattern 3: Task Notification via Callback + Conversation Injection
**What:** When a task completes or fails, inject a system message into the Realtime conversation and trigger the AI to speak a summary.
**When to use:** Registered as TaskManager callbacks at session start.
**Example:**
```python
# Source: TaskManager callback hooks + Realtime API conversation.item.create
async def _on_task_complete(self, task):
    """Callback fired when a background task completes."""
    if not self.ws or not self.running:
        return

    # Build context for the AI
    output_preview = '\n'.join(list(task.output_lines)[-20:])  # Last 20 lines
    duration = (task.completed_at or 0) - (task.started_at or task.created_at)

    notification = (
        f"[Task notification] Task '{task.name}' (#{task.id}) has completed "
        f"successfully after {duration:.0f} seconds. "
        f"Output preview:\n{output_preview[:500]}"
    )

    # Inject as system message
    await self.ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",  # "user" role ensures AI responds
            "content": [{"type": "input_text", "text": notification}]
        }
    }))

    # Trigger AI to acknowledge (it will speak a brief summary)
    await self.ws.send(json.dumps({
        "type": "response.create"
    }))

async def _on_task_failed(self, task):
    """Callback fired when a background task fails."""
    if not self.ws or not self.running:
        return

    output_preview = '\n'.join(list(task.output_lines)[-10:])
    notification = (
        f"[Task notification] Task '{task.name}' (#{task.id}) has failed "
        f"with exit code {task.return_code}. "
        f"Last output:\n{output_preview[:300]}"
    )

    await self.ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": notification}]
        }
    }))

    await self.ws.send(json.dumps({
        "type": "response.create"
    }))
```

### Pattern 4: Task Identifier Resolution
**What:** Resolve user's natural language task reference (name, partial name, or number) to a TaskManager task.
**When to use:** In `get_task_result` and `cancel_task` tool handlers.
**Example:**
```python
def _resolve_task(self, identifier: str) -> Optional[ClaudeTask]:
    """Resolve a task identifier (name, partial name, or number) to a task."""
    tm = TaskManager()

    # Try as numeric ID first
    try:
        task_id = int(identifier)
        task = tm.get_task(task_id)
        if task:
            return task
    except ValueError:
        pass

    # Try name match
    return tm.find_task_by_name(identifier)
```

### Pattern 5: System Prompt with Task Awareness Context
**What:** Extend the personality prompt with task management context so the AI knows its role as both conversationalist and task orchestrator.
**When to use:** In `_build_personality()` or as an addendum to the system prompt.
**Example:**
```python
TASK_ORCHESTRATOR_PROMPT = """
## Task Management

You have access to tools that let you manage background Claude CLI tasks. You are a conversation partner first - task management is one of your capabilities, not your primary purpose.

When the user asks you to do work (refactor code, write tests, fix bugs, etc.), use the spawn_task tool to start it in the background. Acknowledge briefly - one sentence max - then continue the conversation.

When tasks complete or fail, you'll receive notifications. Mention them naturally when there's a pause in conversation. Keep status updates brief - every word you say takes time.

For task status queries, be word-efficient. "The auth refactor finished, and the test fix is still running" is better than listing every field.

When a task reference is ambiguous, ask to clarify. When cancelling, just do it - no confirmation needed.

You learn project directories as the user works. If they've spawned tasks in /home/user/project-a before, you know that project exists.
"""
```

### Anti-Patterns to Avoid
- **Blocking the WebSocket event loop with synchronous TaskManager calls:** TaskManager methods are mostly synchronous for queries (get_task, find_task_by_name), but `spawn_task()` and `cancel_task()` are async. Never use `subprocess.run()` or blocking calls inside `handle_events()`.
- **Sending verbose task output as speech:** Claude CLI can produce thousands of lines of output. Always summarize/truncate for voice delivery. The tool descriptions and system prompt must instruct the AI to summarize, not read verbatim.
- **Injecting notifications while the user is speaking:** Check `self.playing_audio` and `self.muted` state before injecting task completion notifications. If the user is mid-sentence, queue the notification.
- **Using `session.update` to change tools dynamically:** While the API supports updating tools mid-session, there is no need - all task tools are available from the start. Avoid the complexity of dynamic tool registration.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Task spawning/tracking | Custom process management | `TaskManager` from Phase 2 | Already handles spawn, cancel, output streaming, callbacks, cleanup, process group termination |
| Task identifier resolution | Complex NLP parsing | `TaskManager.find_task_by_name()` + try `int()` | Case-insensitive partial matching already implemented |
| WebSocket tool calling protocol | Custom message framing | Existing pattern from `openai_realtime.py` | Proven flow: `response.done` -> detect `function_call` -> `conversation.item.create` -> `response.create` |
| Conversation context management | Custom context window | `LiveSession.maybe_summarize()` | Already handles context summarization when token count exceeds threshold |
| Output ring buffer / persistence | Custom file management | `ClaudeTask.output_lines` + `TaskManager._persist_output()` | Already implemented: capped deque + disk write on completion |

**Key insight:** Phase 3 is primarily a wiring/integration phase. All the hard infrastructure problems (async subprocess lifecycle, voice session management, WebSocket protocol) were solved in Phases 1 and 2. The new code here is tool definitions (JSON schemas), tool handler routing (a dispatcher function), callback registration (a few lines connecting TaskManager events to WebSocket messages), and system prompt extension.

## Common Pitfalls

### Pitfall 1: Event Loop Thread Mismatch
**What goes wrong:** TaskManager callbacks fire from the LiveSession's asyncio event loop thread. If somehow called from the main thread (e.g., a hotkey handler), `await self.ws.send()` will fail.
**Why it happens:** `push-to-talk.py` runs hotkey handling on the main thread, while LiveSession runs in a daemon thread with its own asyncio event loop.
**How to avoid:** TaskManager is created and used only within the LiveSession's event loop. Register callbacks in the `connect()` or `run()` method of LiveSession (which runs in the correct loop). Never call TaskManager from the main thread without `asyncio.run_coroutine_threadsafe()`.
**Warning signs:** `RuntimeError: no running event loop`, or callbacks that silently fail.

### Pitfall 2: Notification Stomping Active Speech
**What goes wrong:** A task completion callback fires while the AI is mid-sentence answering a user question. The injected `conversation.item.create` + `response.create` interrupts or conflicts with the current response.
**Why it happens:** Task completions are asynchronous and unpredictable. The user might be mid-conversation when a task finishes.
**How to avoid:** Check if a response is currently being generated (`self.playing_audio` is True) before injecting notifications. If busy, queue the notification and deliver it after the current response completes (on the next `response.done` event or `response.audio.done`). A simple list `self._pending_notifications = []` checked at the end of each response cycle handles this.
**Warning signs:** Garbled audio, truncated responses, or the AI talking over itself.

### Pitfall 3: Tool Execution Blocking WebSocket Events
**What goes wrong:** `spawn_task()` is an async call that creates a subprocess. If the tool handler awaits it directly in the event handler, it should be fine since `spawn_task()` returns quickly (it creates an asyncio.Task and returns). But if the handler accidentally does something blocking (like reading a file synchronously), it stalls the entire WebSocket event processing.
**Why it happens:** Easy to accidentally add synchronous operations in the tool handler.
**How to avoid:** All tool handlers must be async. `spawn_task()` already returns immediately. Query methods (`get_task`, `get_all_tasks`, etc.) are synchronous but fast (in-memory dict lookups). `cancel_task()` is async (sends signals, waits for process). Keep all handlers non-blocking.
**Warning signs:** WebSocket messages pile up, audio stutters, delayed responses.

### Pitfall 4: Overly Verbose AI Responses for Task Status
**What goes wrong:** The AI reads back task output verbatim or gives lengthy status reports. In voice, every word takes real time. A 30-second status monologue destroys conversational flow.
**Why it happens:** Default AI behavior is to be thorough. Without explicit instruction to be brief, it will over-explain.
**How to avoid:** Tool descriptions must include "summarize concisely" and "every word takes time to speak". The system prompt must reinforce word-efficiency. The existing `voice-style.md` personality file already says "Concise responses. A few sentences max." -- the task orchestrator prompt must be consistent with this.
**Warning signs:** User gets frustrated, interrupts frequently, says "just the summary".

### Pitfall 5: Race Condition on Task Name Resolution
**What goes wrong:** User says "cancel the auth task" but two tasks match "auth". The AI needs to ask for clarification, but the tool handler has already picked one.
**Why it happens:** `find_task_by_name()` returns the first match. Multiple tasks can have overlapping names.
**How to avoid:** When `find_task_by_name()` matches but there are multiple possible matches, return a disambiguation result from the tool (listing matches with IDs) rather than auto-selecting. The AI then asks the user to clarify. Alternatively, return all matching tasks and let the AI choose.
**Warning signs:** Wrong task cancelled, user confusion about which task was affected.

### Pitfall 6: Notification After Session Disconnect
**What goes wrong:** A long-running task completes after the LiveSession has disconnected. The callback tries to send a WebSocket message to a closed connection.
**Why it happens:** TaskManager is a singleton that persists across sessions. Callbacks registered for one session may fire after that session ends.
**How to avoid:** Check `self.ws is not None and self.running` at the top of every callback. Deregister callbacks on session disconnect. Or, since TaskManager callbacks are stored as lists, replace the callback list on each new session connect rather than appending.
**Warning signs:** Unhandled exceptions in callback, `websockets.exceptions.ConnectionClosed`.

## Code Examples

### Complete Tool Handler Dispatcher
```python
# Source: Adapted from openai_realtime.py execute_tool() pattern
async def _execute_tool(self, name: str, arguments: dict) -> dict:
    """Execute a task management tool and return result dict."""
    tm = TaskManager()

    if name == "spawn_task":
        task_name = arguments.get("name", "unnamed task")
        prompt = arguments.get("prompt", "")
        project_dir = Path(arguments.get("project_dir", str(Path.home())))

        if not project_dir.is_dir():
            return {"error": f"Directory does not exist: {project_dir}"}

        try:
            task = await tm.spawn_task(task_name, prompt, project_dir)
            return {
                "success": True,
                "task_id": task.id,
                "task_name": task.name,
                "message": f"Task '{task.name}' started in {project_dir}"
            }
        except ValueError as e:
            return {"error": str(e)}

    elif name == "list_tasks":
        tasks = tm.get_all_tasks()
        if not tasks:
            return {"tasks": [], "message": "No tasks"}

        task_list = []
        for t in tasks:
            info = {
                "id": t.id,
                "name": t.name,
                "status": t.status.value,
            }
            if t.started_at:
                elapsed = (t.completed_at or time.time()) - t.started_at
                info["elapsed_seconds"] = round(elapsed)
            task_list.append(info)

        return {"tasks": task_list}

    elif name == "get_task_result":
        identifier = arguments.get("task_identifier", "")
        task = self._resolve_task(identifier)
        if not task:
            return {"error": f"No task found matching '{identifier}'"}

        output = tm.get_task_output(task.id)
        # Truncate for voice delivery (AI will summarize further)
        if output and len(output) > 2000:
            output = output[-2000:]  # Last 2000 chars most relevant

        return {
            "task_id": task.id,
            "task_name": task.name,
            "status": task.status.value,
            "output": output or "(no output)",
        }

    elif name == "cancel_task":
        identifier = arguments.get("task_identifier", "")
        task = self._resolve_task(identifier)
        if not task:
            return {"error": f"No task found matching '{identifier}'"}

        cancelled = await tm.cancel_task(task.id)
        return {
            "success": cancelled,
            "task_name": task.name,
            "message": f"Cancelled '{task.name}'" if cancelled else f"Task '{task.name}' is not running"
        }

    return {"error": f"Unknown tool: {name}"}
```

### Callback Registration in LiveSession
```python
# Source: TaskManager callback API + LiveSession lifecycle
async def connect(self):
    """Connect to OpenAI Realtime API and configure session."""
    # ... existing connection code ...

    # Register task completion callbacks
    tm = TaskManager()
    # Clear any stale callbacks from previous sessions
    tm._callbacks['on_task_complete'] = []
    tm._callbacks['on_task_failed'] = []
    tm.on('on_task_complete', self._on_task_complete)
    tm.on('on_task_failed', self._on_task_failed)
```

### Notification Queue for Conversational Timing
```python
# Source: Design pattern for non-interruptive notifications
class LiveSession:
    def __init__(self, ...):
        # ... existing init ...
        self._pending_notifications = []  # Queue for deferred task notifications

    async def _on_task_complete(self, task):
        """Queue a task completion notification."""
        notification = self._format_task_notification(task, "completed")
        if self.playing_audio:
            # AI is currently speaking - defer
            self._pending_notifications.append(notification)
        else:
            await self._deliver_notification(notification)

    async def _deliver_notification(self, notification_text):
        """Inject a notification into the conversation."""
        if not self.ws or not self.running:
            return
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": notification_text}]
            }
        }))
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def _flush_pending_notifications(self):
        """Deliver any queued notifications. Call after response.done."""
        while self._pending_notifications:
            notification = self._pending_notifications.pop(0)
            await self._deliver_notification(notification)
```

### Session.update with Tools
```python
# Source: Existing live_session.py connect() method + openai_realtime.py TOOLS pattern
await self.ws.send(json.dumps({
    "type": "session.update",
    "session": {
        "modalities": ["text", "audio"],
        "instructions": self.personality_prompt,
        "voice": self.voice,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": {
            "type": "semantic_vad",
            "eagerness": "medium",
            "interrupt_response": True
        },
        "tools": TASK_TOOLS,       # NEW: task management tools
        "tool_choice": "auto"       # CHANGED: from "none" to "auto"
    }
}))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tools: [], tool_choice: "none"` in LiveSession | `tools: TASK_TOOLS, tool_choice: "auto"` | Phase 3 | AI can now spawn/manage tasks |
| Synchronous `execute_tool()` in openai_realtime.py | Async `_execute_tool()` in LiveSession | Phase 3 | Non-blocking tool execution in the event loop |
| No task awareness in voice session | System prompt includes task orchestrator context | Phase 3 | AI proactively manages and reports on tasks |
| `response.done` without function call handling in LiveSession | `response.done` with function call dispatch | Phase 3 | LiveSession can process tool calls like RealtimeSession |

**Not deprecated but superseded:**
- `openai_realtime.py` RealtimeSession: Still functional for legacy "realtime" AI mode, but LiveSession becomes the primary voice mode with tool capabilities. The `RealtimeSession.execute_tool()` synchronous pattern is NOT used in LiveSession (we use async instead).

## Open Questions

1. **Notification timing precision**
   - What we know: We can check `self.playing_audio` to see if the AI is currently speaking. We can queue notifications.
   - What's unclear: The exact right moment to flush queued notifications. After `response.audio.done`? After `response.done`? After a short delay?
   - Recommendation: Flush pending notifications in the `response.done` handler, after processing function calls and transcript tracking. This ensures the AI finished speaking and any tool calls are handled before we inject new context.

2. **Multi-match task disambiguation**
   - What we know: `find_task_by_name()` returns the first partial match. User context decisions say "when reference is ambiguous, AI asks to clarify."
   - What's unclear: Whether to handle disambiguation in the tool handler (return multiple matches) or let the AI figure it out.
   - Recommendation: Return all matches when multiple tasks match the identifier. Let the AI's judgment (which has conversational context) decide how to ask for clarification.

3. **Project directory discovery**
   - What we know: User decided on "multi-context project awareness -- AI learns projects over time."
   - What's unclear: Where to store the project directory map. In TaskManager? In LiveSession? On disk?
   - Recommendation: Start simple -- the AI's conversational context naturally accumulates knowledge of project directories as tasks are spawned. No persistent storage needed for v1. The conversation history and summarization system preserves this context. If the session restarts, the user re-establishes context naturally.

## Sources

### Primary (HIGH confidence)
- `openai_realtime.py` (lines 34-133, 363-399) - Existing proven tool calling implementation
- `live_session.py` (lines 46-113, 211-312) - LiveSession architecture and session.update
- `task_manager.py` (lines 55-405) - Complete TaskManager API with callbacks
- OpenAI Realtime API guide: [Realtime conversations](https://platform.openai.com/docs/guides/realtime-conversations) - Tool calling flow
- OpenAI Realtime API reference: [Client events](https://platform.openai.com/docs/api-reference/realtime-client-events) - session.update, conversation.item.create, response.create
- Mamezou function calling guide: [Realtime API Function Calling](https://developer.mamezou-tech.com/en/blogs/2024/10/09/openai-realtime-api-function-calling/) - Verified tool call flow: define tools -> response.output_item.done -> conversation.item.create -> response.create

### Secondary (MEDIUM confidence)
- OpenAI Community thread on [conversation.item.create behavior](https://community.openai.com/t/realtime-api-questions-on-client-events-and-token-usage-conversation-item-create-session-update-response-create/991218) - System message roles and token implications
- OpenAI Community thread on [function call output injection](https://community.openai.com/t/injecting-function-call-output-in-realtime-api-beta/1260451) - Confirmed flow: function_call_output then response.create

### Tertiary (LOW confidence)
- Notification timing (queue flush timing) - Based on architectural reasoning, not API documentation. Should be validated empirically.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing code from Phases 1 and 2, no new dependencies
- Architecture: HIGH - Following proven patterns from openai_realtime.py, direct code analysis
- Tool calling protocol: HIGH - Verified against existing working implementation and official docs
- Notification timing: MEDIUM - Logical design based on event model, but timing edge cases need empirical testing
- Pitfalls: HIGH - Based on codebase analysis and asyncio expertise

**Research date:** 2026-02-15
**Valid until:** 90 days (stable domain; Realtime API WebSocket protocol is mature; TaskManager API is frozen)
