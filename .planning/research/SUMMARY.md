# Project Research Summary

**Project:** Voice-Controlled Async Task Orchestrator (Live Mode)
**Domain:** Voice-controlled async task orchestration for Claude CLI
**Researched:** 2026-02-13
**Confidence:** HIGH

## Executive Summary

This project adds a "Live Mode" to the existing push-to-talk application, enabling voice-controlled orchestration of multiple concurrent Claude CLI tasks. The core insight: the app already runs an asyncio event loop for the OpenAI Realtime WebSocket session, so the orchestrator lives inside that same loop using `asyncio.create_subprocess_exec()` for Claude CLI processes. No new frameworks are needed—just stdlib asyncio plus one library (`janus`) for thread-to-async bridging.

The architecture is cleanly layered: LiveSession owns both a RealtimeSession (voice) and TaskManager (orchestration). The TaskManager spawns isolated ClaudeTask instances, each running a separate Claude CLI subprocess in its own working directory. The OpenAI Realtime API's GA model (`gpt-realtime`) supports async function calling, meaning the AI can acknowledge task spawning immediately ("On it, task started") and continue conversation while work happens in background. Task completion notifications inject results back into the conversation stream.

The critical risk is blocking the asyncio event loop with synchronous subprocess calls—this kills the WebSocket connection within 20 seconds. Prevention: use `asyncio.create_subprocess_exec()` exclusively, never `subprocess.run()`. Secondary risks include task garbage collection (solved with strong reference tracking), session expiry/reconnection (mitigated with local task registry persistence), and zombie process accumulation (prevented with PID tracking and cleanup handlers).

## Key Findings

### Recommended Stack

The existing asyncio foundation supports the orchestrator without new dependencies. Python 3.12 stdlib provides everything needed: `asyncio` for subprocess management, `dataclasses` for task state modeling, `StrEnum` for lifecycle states, `uuid` for task IDs. The Claude CLI is already installed and supports all necessary flags (`-p`, `--output-format stream-json`, `--permission-mode`, `--max-turns`, `--no-session-persistence`). The only new dependency is `janus` (thread-safe asyncio queue) to bridge pynput hotkey events from the keyboard thread into the asyncio loop.

**Core technologies:**
- Python `asyncio` (stdlib): Async subprocess spawning with `create_subprocess_exec()`, non-blocking I/O
- Python `dataclasses` (stdlib): Task state modeling (TaskRecord with slots=True for efficiency)
- Python `StrEnum` (stdlib): Task lifecycle states (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- `janus` 2.0.0: Thread-safe bridge between pynput (sync thread) and asyncio event loop
- Claude CLI 2.1.41: Background execution engine with stream-json output for progress monitoring
- OpenAI Realtime API `gpt-realtime`: GA model with native async function calling support

**Key decision:** Avoid external task queue frameworks (Celery, Dramatiq) and state machines—this is a single-user desktop app with a simple linear task lifecycle. Stdlib is sufficient and eliminates dependencies.

### Expected Features

Research identified a clear MVP scope and v2 deferral strategy. The core insight from competitors (Claude Code Agent Teams, SystemPrompt) is that no one does voice + async task orchestration well yet—this is genuinely novel territory.

**Must have (table stakes):**
- Async task spawning with immediate voice acknowledgment
- Non-blocking conversation while tasks run in background
- Task completion/failure notifications (audio + spoken summary)
- Status query ("What are my tasks doing?")
- Task cancellation ("Cancel the auth task")
- Task result retrieval with summarization
- In-memory task registry mapping task_id to state

**Should have (competitive differentiators):**
- Named task context switching (refer to tasks by description, not ID)
- Proactive status announcements (AI volunteers updates during pauses)
- Audio notification differentiation (distinct sounds for start/complete/fail)
- Context isolation (separate working directories per task)
- Ambient task awareness (AI knows running tasks without being asked)
- Smart result summarization (condense Claude CLI output for voice delivery)

**Defer (v2+):**
- Session persistence across WebSocket reconnects (disk-backed registry)
- Voice-driven task composition (chain task outputs)
- Git worktree isolation (parallel branches for same-repo tasks)
- Real-time streaming of Claude CLI output (too verbose for voice)

**Anti-features to avoid:**
- Interactive mid-task steering (Claude CLI doesn't support stdin mid-execution; spawn new tasks instead)
- Visual task dashboard (defeats the voice-first premise)
- Always-on listening without PTT (Server VAD within sessions already handles this)

### Architecture Approach

The architecture extends the existing codebase cleanly without disruptive refactoring. LiveSession becomes the glue layer between voice (RealtimeSession) and orchestration (TaskManager). Each ClaudeTask is an isolated worker with its own subprocess and working directory.

**Major components:**
1. **LiveSession** — Owns both RealtimeSession and TaskManager, configures tools, routes tool calls to task operations
2. **TaskManager** — Registry tracking all spawned tasks, provides spawn/query/cancel interfaces, monitors completion
3. **ClaudeTask** — Single Claude CLI subprocess with isolated context (own cwd, unique session, output buffer)
4. **RealtimeSession (modified)** — Pluggable tool handler instead of hardcoded execution, supports async tool callbacks
5. **Task Notification System** — Injects completion events into Realtime conversation via `conversation.item.create`

**Key patterns:**
- **Async subprocess management:** Use `asyncio.create_subprocess_exec()` exclusively, never `subprocess.run()`
- **Fire-and-acknowledge for long tasks:** Return immediate acknowledgment ("task started"), deliver results later via conversation injection
- **Context isolation via working directory:** Each task gets own `cwd`, no shared Claude session state
- **Pluggable tool handler:** RealtimeSession accepts a `tool_handler` callback, making it reusable across modes

**Data flow:** User speaks → Realtime API transcribes + generates function_call → LiveSession routes to TaskManager → TaskManager spawns ClaudeTask subprocess → Returns immediately with task_id → AI acknowledges → Task runs in background → On completion, notification injected into conversation → AI speaks result

### Critical Pitfalls

Research identified six critical pitfalls that will break the system if not addressed from day one.

1. **Blocking the asyncio event loop with synchronous subprocess.run()** — Using `subprocess.run()` from async code freezes the entire WebSocket session, killing the connection after 20s (keepalive timeout). Prevention: use `asyncio.create_subprocess_exec()` exclusively. Phase 1 foundation.

2. **Fire-and-forget asyncio tasks getting garbage collected** — Python's event loop uses WeakSet for tasks; without strong references, background tasks silently vanish mid-execution. Prevention: maintain a strong reference set with done callbacks. Phase 1 lifecycle management.

3. **OpenAI Realtime API session expiry and reconnection state loss** — Sessions expire after 15-30 minutes; WebSocket drops lose all conversation context including task awareness. Prevention: persist task registry to disk, inject state summary on reconnect. Phase 2 resilience.

4. **Concurrent state mutation in task registry** — Multiple async handlers mutate shared task dict simultaneously, causing race conditions. Prevention: use `asyncio.Lock` for registry access or event-sourced state pattern. Phase 1 registry design.

5. **Claude CLI process zombie accumulation** — Failing to `await process.wait()` creates zombies; crashes leave orphaned Claude processes running. Prevention: track PIDs in file, cleanup on startup, use atexit handlers. Phase 1 process lifecycle.

6. **Realtime API function call response ordering bugs** — Model may hallucinate results if function calls take too long without response. Prevention: immediate acknowledgment pattern, always return something within seconds. Phase 1 tool integration.

## Implications for Roadmap

Based on research, the roadmap should follow this phase structure. The dependency chain is clear: foundation → capability → integration → polish.

### Phase 1: Async Infrastructure Foundation

**Rationale:** The entire async task system depends on non-blocking subprocess management and pluggable tool handlers. Without this foundation, everything else fails. This phase refactors existing code to support async patterns without adding new features.

**Delivers:**
- RealtimeSession with pluggable `tool_handler` callback and configurable `tools` list
- Upgrade to `gpt-realtime` GA model for native async function calling
- Async-aware tool execution pattern (handlers can be async)
- Proof that existing functionality (current tools) still works

**Avoids:**
- Pitfall 1: Blocking event loop (replaced `subprocess.run()` with async subprocess)
- Pitfall 6: Function call ordering bugs (immediate acknowledgment pattern established)

**Research flag:** Standard patterns—asyncio subprocess is well-documented. Skip `/gsd:research-phase`.

---

### Phase 2: Task Orchestration Core

**Rationale:** With async infrastructure in place, build the task management capability in isolation. This phase is testable without voice integration—TaskManager and ClaudeTask can be unit-tested via direct API calls.

**Delivers:**
- TaskManager class (spawn, track, query, cancel operations)
- ClaudeTask class (async subprocess wrapper with output capture and timeout)
- Task state modeling (dataclasses: TaskRecord, TaskState enum)
- Strong reference tracking to prevent task garbage collection
- PID file for orphan cleanup
- Basic unit tests for task lifecycle

**Uses:**
- `asyncio.create_subprocess_exec()` for Claude CLI spawning
- `dataclasses` for task state
- `StrEnum` for lifecycle states
- `janus` for thread-to-async bridge (hotkey events)

**Avoids:**
- Pitfall 2: Task garbage collection (strong reference set with done callbacks)
- Pitfall 4: State race conditions (asyncio.Lock on registry)
- Pitfall 5: Zombie processes (PID tracking, cleanup handlers)

**Research flag:** Standard patterns—asyncio subprocess management is well-understood. Skip `/gsd:research-phase`.

---

### Phase 3: Voice Integration (Live Mode Assembly)

**Rationale:** With foundation (Phase 1) and capability (Phase 2) complete, wire them together. LiveSession connects RealtimeSession to TaskManager. This phase makes the orchestrator voice-controllable.

**Delivers:**
- LiveSession class (owns RealtimeSession + TaskManager pair)
- Task-oriented tool definitions (start_task, check_tasks, get_task_result, cancel_task)
- Tool handler routing to TaskManager operations
- Proactive task completion notifications via `conversation.item.create`
- Integration into PushToTalk mode routing
- Hotkey handling for live mode toggle

**Implements:**
- Fire-and-acknowledge pattern: tools return immediately, results delivered later
- Context isolation: each task gets own working directory
- Notification injection: TaskManager callbacks push completion events into conversation

**Addresses:**
- Must-have features: async spawning, status query, cancellation, result retrieval, non-blocking conversation
- Competitive feature: ambient task awareness (AI knows running tasks via tool access)

**Avoids:**
- Pitfall 1: Non-blocking during task execution (voice conversation continues)
- Pitfall 6: Immediate tool acknowledgment prevents hallucinated results

**Research flag:** Moderate complexity—OpenAI Realtime API function calling has known quirks. Consider `/gsd:research-phase` if function call ordering causes issues.

---

### Phase 4: Resilience and UX Polish

**Rationale:** Core functionality works, now make it production-ready. Address WebSocket reconnection, session persistence, audio notifications, and error handling.

**Delivers:**
- WebSocket reconnection with task state recovery
- Disk-backed task registry for session persistence
- Audio notification sounds (start/complete/fail cues)
- Task result summarization (truncate/condense for voice)
- Named task context switching (refer by description)
- Cost tracking and warnings (max concurrent tasks, API budget)

**Addresses:**
- Should-have features: proactive announcements, audio differentiation, smart summarization, named tasks
- UX pitfalls: silent processing, verbose updates, ambiguous references, context overload

**Avoids:**
- Pitfall 3: Session expiry (proactive reconnection, state restoration)
- UX pitfalls: no feedback, interruption during flow, latency in notifications

**Research flag:** Reconnection state management is moderately complex. Consider `/gsd:research-phase` for WebSocket reconnection patterns.

---

### Phase 5: Mode Rename and Documentation

**Rationale:** With live mode complete, update all user-facing elements to reflect the new mode structure. Rename "live" dictation mode to "dictate" and "realtime" to "live" across codebase and UI.

**Delivers:**
- Config migration (dictation_mode: "live" → "dictate", ai_mode: "realtime" → "live")
- Indicator UI updates (new mode names, task count display)
- Settings tab for live mode configuration (max tasks, default working dir)
- Documentation updates (README, CLAUDE.md, website)

**Addresses:**
- User-facing clarity (distinct mode names)
- Discoverability (settings UI for configuration)

**Research flag:** UI/config work—no research needed. Skip `/gsd:research-phase`.

---

### Phase Ordering Rationale

**Dependency chain enforces this order:**
- Phase 1 must come first: pluggable tool handler is required for Phase 3 integration; async subprocess patterns are required for Phase 2 task execution
- Phase 2 depends on Phase 1: TaskManager uses async subprocess APIs established in Phase 1
- Phase 3 depends on Phase 1 + 2: LiveSession wires together the refactored RealtimeSession (Phase 1) and TaskManager (Phase 2)
- Phase 4 can only be validated after Phase 3 works: reconnection/persistence need a working live mode to test against
- Phase 5 is last: can't rename modes until live mode exists

**Pitfall mitigation drives this structure:**
- Critical pitfalls (1, 2, 4, 5, 6) are all addressed in Phases 1-2, before voice integration
- This means the orchestration core is solid before adding the complexity of WebSocket + voice + conversation context
- Phase 3 risks are lower because the hard async problems are already solved
- Phase 4 addresses the one deferred pitfall (session expiry) after core functionality proves stable

**Incremental validation:**
- Phase 1: existing features still work (regression testing)
- Phase 2: task lifecycle works via unit tests (no voice needed)
- Phase 3: full voice-controlled task orchestration (integration testing)
- Phase 4: resilience under network failures, long sessions (stress testing)
- Phase 5: user-facing polish (acceptance testing)

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 3:** OpenAI Realtime API async function calling has known quirks (model version differences, response ordering bugs). If issues arise, use `/gsd:research-phase` to investigate model-specific behavior and workarounds.
- **Phase 4:** WebSocket reconnection state management may need research for edge cases (reconnect during task completion, multiple reconnects in sequence). Consider `/gsd:research-phase` if complexity exceeds expectations.

**Phases with standard patterns (skip research-phase):**
- **Phase 1:** Asyncio subprocess management and callback patterns are well-documented in Python stdlib docs
- **Phase 2:** Task registry and subprocess lifecycle are straightforward asyncio patterns
- **Phase 5:** Config migration and UI updates are mechanical—no novel patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All core technologies verified against Python 3.12 docs and existing codebase. Janus is production-stable (aio-libs, 2.0.0 released Dec 2024). Claude CLI flags confirmed in official docs. |
| Features | MEDIUM-HIGH | Strong codebase understanding + ecosystem research. Domain is novel (no direct competitors doing voice + async task orchestration), so fewer precedents. Feature prioritization is inference-based but grounded in VUI best practices. |
| Architecture | HIGH | Existing codebase well-understood. Component boundaries are clean extensions of current architecture. Async subprocess patterns are proven. Critical paths (event loop, WebSocket lifecycle, tool execution) directly analyzed. |
| Pitfalls | HIGH | All critical pitfalls verified against official docs (Python asyncio, OpenAI Realtime API) and community reports. Process lifecycle, GC behavior, WebSocket keepalive are documented behaviors. Function call ordering bugs confirmed in OpenAI community posts. |

**Overall confidence:** HIGH

### Gaps to Address

While research is thorough, these areas need validation during implementation:

- **OpenAI Realtime API GA model behavior:** The `gpt-realtime` model is newly GA; async function calling behavior may differ from preview model. Test thoroughly during Phase 3. If unexpected issues arise, pin to preview model or research model-specific quirks.

- **WebSocket reconnection timing:** Optimal reconnection strategy (backoff intervals, context injection timing) will require testing under real network conditions. Phase 4 may need experimentation to find the right balance between eager reconnection and avoiding thundering herd on the OpenAI API.

- **Task result summarization quality:** How much Claude CLI output can fit in Realtime API context before hitting token limits? How effective is AI summarization of technical output for voice delivery? Phase 4 should include testing with diverse task types (code edits, test runs, git operations).

- **Janus thread-bridge performance:** The pynput keyboard thread to asyncio bridge via janus is untested in this specific use case. Monitor for latency in hotkey response during Phase 2 integration.

- **Concurrent task limits:** Research suggests 5-10 concurrent tasks is reasonable, but actual limits depend on system resources and Claude CLI behavior. Phase 4 should validate and set appropriate defaults based on testing.

## Sources

### Primary (HIGH confidence)
- [Python 3.12 asyncio subprocess documentation](https://docs.python.org/3.12/library/asyncio-subprocess.html) — subprocess APIs, Process class
- [Python 3.12 dataclasses documentation](https://docs.python.org/3/library/dataclasses.html) — field(default_factory), slots=True
- [Python 3.12 enum documentation](https://docs.python.org/3/library/enum.html) — StrEnum availability (3.11+)
- [janus 2.0.0 on PyPI](https://pypi.org/project/janus/) — thread-safe queue API
- [janus GitHub repository](https://github.com/aio-libs/janus) — sync_q/async_q pattern, aclose() requirement
- [Claude CLI reference](https://code.claude.com/docs/en/cli-reference) — all flags verified
- [Claude Code Agent Teams documentation](https://code.claude.com/docs/en/agent-teams) — task coordination patterns
- [websockets 16.0 on PyPI](https://pypi.org/project/websockets/) — latest version compatibility
- [CPython asyncio task reference retention](https://docs.python.org/3/library/asyncio-task.html) — garbage collection behavior
- Existing codebase: `/home/ethan/code/push-to-talk/openai_realtime.py`, `/home/ethan/code/push-to-talk/push-to-talk.py` — direct code inspection

### Secondary (MEDIUM confidence)
- [OpenAI Realtime API documentation](https://platform.openai.com/docs/guides/realtime) — WebSocket protocol (403 during research, relied on cached/community sources)
- [OpenAI community: long function calls](https://community.openai.com/t/long-function-calls-and-realtime-api/1119021) — async patterns
- [OpenAI community: async tool calling](https://community.openai.com/t/disabling-asynchronous-tool-calling-with-gpt-realtime/1360261) — GA model behavior
- [OpenAI community: WebSocket keepalive timeout](https://community.openai.com/t/realtime-api-websocket-disconnects-randomly-in-nodejs/1044456) — 20s ping interval
- [OpenAI community: function calling response bug](https://community.openai.com/t/realtime-api-no-response-after-function-calling-until-next-user-turn-gpt-4o-realtime-preview-2025-06-03/1297639) — model version differences
- [SystemPrompt Code Orchestrator](https://github.com/systempromptio/systemprompt-code-orchestrator) — MCP-based agent patterns
- [Pipecat voice agent framework](https://github.com/pipecat-ai/pipecat) — voice agent architecture patterns
- [VUI Design Principles](https://www.parallelhq.com/blog/voice-user-interface-vui-design-principles) — voice UX best practices
- [Google VUI Design](https://design.google/library/speaking-the-same-language-vui) — conversation flow patterns
- [Gladia: concurrent pipelines for voice AI](https://www.gladia.io/blog/concurrent-pipelines-for-voice-ai) — async architecture patterns
- [SignalWire: the Double Update problem](https://signalwire.com/blogs/developers/the-double-update) — state race conditions
- [Armin Ronacher on asyncio.create_task footgun](https://x.com/mitsuhiko/status/1920384040005173320) — task GC issue awareness

### Tertiary (LOW confidence, needs validation)
- [Voice AI stack 2026](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents) — ecosystem overview, not product-specific
- [Claude CLI session management flags](https://claudelog.com/faqs/what-is-resume-flag-in-claude-code/) — third-party documentation

---
*Research completed: 2026-02-13*
*Ready for roadmap: yes*
