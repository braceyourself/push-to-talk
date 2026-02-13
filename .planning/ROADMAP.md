# Roadmap: Push-to-Talk Live Mode

## Overview

This roadmap delivers a voice-controlled async task orchestrator in three phases. First, rename the existing "live" dictation mode to "dictate" and stand up a new live mode with basic real-time voice conversation. Second, build the async Claude CLI task management layer (TaskManager, ClaudeTask, process lifecycle). Third, wire task orchestration into the live voice session so users can spawn, monitor, and control Claude CLI tasks by voice.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Mode Rename and Live Voice Session** - Rename "live" to "dictate" and create a new live mode with real-time voice conversation via OpenAI Realtime API
- [ ] **Phase 2: Async Task Infrastructure** - Build TaskManager and ClaudeTask classes for non-blocking Claude CLI subprocess management
- [ ] **Phase 3: Voice-Controlled Task Orchestration** - Wire task management into the live session so users can spawn, query, cancel, and receive results from Claude CLI tasks by voice

## Phase Details

### Phase 1: Mode Rename and Live Voice Session
**Goal**: User can select the new "live" dictation mode and have a real-time voice conversation with AI, with the old live mode cleanly renamed to "dictate"
**Depends on**: Nothing (first phase)
**Requirements**: RENAME-01, RENAME-02, RENAME-03, RENAME-04, LIVE-01, LIVE-02, LIVE-03, LIVE-04
**Success Criteria** (what must be TRUE):
  1. User can select "Dictate" in the Settings combo box and it behaves exactly like the old "Live" mode did
  2. User can select "Live" in the Settings combo box and it opens an OpenAI Realtime voice session
  3. User can hold PTT to speak, release to send, and hear AI respond through speakers in live mode
  4. Conversation context persists across multiple PTT presses within a single live session
  5. Starting and stopping a live session cleanly initializes and tears down without errors
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Rename "live" dictation mode to "dictate" across codebase, config, and UI
- [x] 01-02-PLAN.md -- Implement new live mode with LiveSession, personality system, and overlay widget

### Phase 2: Async Task Infrastructure
**Goal**: A TaskManager can spawn, track, query, and cancel isolated Claude CLI subprocesses without blocking the asyncio event loop
**Depends on**: Phase 1
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06
**Success Criteria** (what must be TRUE):
  1. Claude CLI tasks spawn asynchronously via `asyncio.create_subprocess_exec()` and do not block the event loop
  2. TaskManager tracks each task's id, name, status, process handle, and captured output
  3. Each task runs in its own isolated working directory with no shared session state
  4. Completed and failed tasks are cleaned up (process reaped, strong references released)
  5. No zombie Claude CLI processes accumulate during normal operation or after failures
**Plans**: TBD

Plans:
- [ ] 02-01: TaskManager and ClaudeTask classes with async subprocess lifecycle

### Phase 3: Voice-Controlled Task Orchestration
**Goal**: User can manage Claude CLI tasks entirely by voice during a live session -- spawning work, checking status, retrieving results, and cancelling tasks through natural conversation
**Depends on**: Phase 1, Phase 2
**Requirements**: TASK-01, TASK-02, TASK-03, TASK-04, TASK-05, TASK-06, TASK-07, CTX-01, CTX-02, CTX-03
**Success Criteria** (what must be TRUE):
  1. User can say something like "ask Claude to refactor the auth module" and a task spawns in the background while conversation continues uninterrupted
  2. When a task completes or fails, the user hears a notification and a spoken summary of the outcome
  3. User can ask "what are my tasks doing?" and hear a status summary of all active and completed tasks
  4. User can cancel a running task by referring to it by name
  5. User can ask for a specific task's results and hear a spoken summary of its output
**Plans**: TBD

Plans:
- [ ] 03-01: Realtime API tool definitions and LiveSession tool handler routing
- [ ] 03-02: Task notifications, context naming, and ambient awareness integration

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Mode Rename and Live Voice Session | 2/2 | Complete | 2026-02-13 |
| 2. Async Task Infrastructure | 0/1 | Not started | - |
| 3. Voice-Controlled Task Orchestration | 0/2 | Not started | - |
