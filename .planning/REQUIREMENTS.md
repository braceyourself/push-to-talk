# Requirements: Push-to-Talk Live Mode

**Defined:** 2026-02-13
**Core Value:** Real-time voice-to-voice AI conversation that can delegate real work to Claude CLI and manage multiple async tasks with context isolation

## v1 Requirements

### Mode Rename

- [x] **RENAME-01**: Current "live" dictation mode renamed to "dictate" in all code references
- [x] **RENAME-02**: Settings UI updated — combo box shows "Dictate" instead of "Live"
- [x] **RENAME-03**: Voice commands updated — "dictate mode" activates dictate, "live mode" activates new live mode
- [x] **RENAME-04**: Config default value changed from "live" to "dictate" where appropriate

### Async Infrastructure

- [x] **INFRA-01**: RealtimeSession tool execution is async — `execute_tool()` uses `asyncio.create_subprocess_exec()` instead of `subprocess.run()`
- [x] **INFRA-02**: TaskManager class tracks spawned tasks in-memory (id, name, process, status, start_time, stdout, stderr)
- [x] **INFRA-03**: Each Claude CLI task runs in its own working directory (`~/.local/share/push-to-talk/tasks/{task_id}/`)
- [x] **INFRA-04**: Claude CLI invoked with `-p` (no `-c`), `--no-session-persistence` for full context isolation
- [x] **INFRA-05**: Strong references maintained for asyncio tasks to prevent garbage collection
- [x] **INFRA-06**: Process cleanup on task completion/failure (PID tracking, directory cleanup)

### Task Lifecycle

- [ ] **TASK-01**: User can spawn a Claude CLI task by voice — "ask Claude to refactor the auth module" triggers async subprocess
- [ ] **TASK-02**: Task spawning returns immediately — AI acknowledges "started" and conversation continues
- [ ] **TASK-03**: Task completion triggers audio notification + spoken summary
- [ ] **TASK-04**: Task failure triggers distinct audio notification + spoken error summary
- [ ] **TASK-05**: User can ask task status — "what are my tasks doing?" returns spoken summary of all tasks
- [ ] **TASK-06**: User can cancel a running task — "cancel the auth task" terminates the subprocess
- [ ] **TASK-07**: User can retrieve task results — "what did the database task produce?" speaks output summary

### Context Management

- [ ] **CTX-01**: Named context switching — AI assigns human-readable names to tasks, user refers to them by name
- [ ] **CTX-02**: Context isolation — concurrent tasks cannot see each other's file changes or session state
- [ ] **CTX-03**: Ambient task awareness — AI automatically knows what tasks are running without being asked

### Live Mode Session

- [x] **LIVE-01**: New "live" dictation mode activates OpenAI Realtime voice session
- [x] **LIVE-02**: Hold PTT to speak, release to send — AI responds through speakers
- [x] **LIVE-03**: Session memory — conversation persists across PTT presses within a session
- [x] **LIVE-04**: Session start/stop cleanly initializes and tears down task registry

## v1.1 Requirements: Voice UX Polish

### Filler System

- [ ] **FILL-01**: Remove Ollama smart filler generation entirely — no LLM-generated filler text
- [ ] **FILL-02**: Live session uses only non-verbal canned audio clips as fillers (breaths, hums, etc.)
- [ ] **FILL-03**: Clip factory subprocess generates new non-verbal clips via Piper TTS
- [ ] **FILL-04**: Clip pool has a configurable size cap with rotation (phase out old, generate new)
- [ ] **FILL-05**: Clip factory evaluates naturalness of generated clips before adding to pool

### STT Reliability

- [x] **STT-01**: Whisper no_speech_prob filtering rejects non-speech segments (committed)
- [ ] **STT-02**: False trigger rate reduced — throat clearing, coughs, ambient noise don't trigger transcription

### Speech Flow

- [x] **FLOW-01**: Tool-use text suppression — only final post-tool response is spoken (committed)
- [x] **FLOW-02**: Inter-tool narration is discarded via post_tool_buffer (committed)

### Barge-in

- [ ] **BARGE-01**: User can interrupt AI mid-speech by speaking
- [ ] **BARGE-02**: VAD detects speech during playback without muting the mic
- [ ] **BARGE-03**: Interruption cancels current TTS playback and queued audio
- [ ] **BARGE-04**: Interrupted speech is not sent as context (or marked as interrupted)

### Overlay

- [x] **OVL-01**: All status states (listening, thinking, tool_use, speaking, idle, muted) render correctly (committed)
- [x] **OVL-02**: Status history panel shows transitions with timestamps (committed)

## v2 Requirements

### Resilience

- **RESIL-01**: WebSocket reconnection with exponential backoff preserves task state
- **RESIL-02**: Disk-backed task registry survives session crashes
- **RESIL-03**: Proactive session renewal before API timeout

### Voice UX

- **VUX-01**: Proactive status announcements during conversation pauses
- **VUX-02**: Audio notification differentiation (distinct sounds for start/complete/fail)
- **VUX-03**: Smart result summarization — AI condenses long Claude CLI output for voice

### Advanced

- **ADV-01**: Voice-driven task composition — chain task outputs as input to new tasks
- **ADV-02**: Git worktree isolation for parallel same-repo tasks

## Out of Scope

| Feature | Reason |
|---------|--------|
| Interactive Claude CLI steering | Running sessions don't accept stdin mid-execution; spawn new tasks instead |
| Real-time streaming of CLI output to voice | Verbose tool-use output is overwhelming as audio; summarize on completion |
| Arbitrary shell commands | Dangerous with voice misrecognition; route through Claude's permission model |
| Always-on listening (no PTT) | Privacy, CPU drain, false positives; server VAD handles within-session listening |
| Visual task dashboard | Defeats voice-first purpose; use status indicator + voice queries |
| Persistent tasks across sessions | Scope creep into job scheduler; each session starts fresh |
| Multi-user task sharing | Single-user desktop tool |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RENAME-01 | Phase 1 | Complete |
| RENAME-02 | Phase 1 | Complete |
| RENAME-03 | Phase 1 | Complete |
| RENAME-04 | Phase 1 | Complete |
| INFRA-01 | Phase 2 | Complete |
| INFRA-02 | Phase 2 | Complete |
| INFRA-03 | Phase 2 | Complete |
| INFRA-04 | Phase 2 | Complete |
| INFRA-05 | Phase 2 | Complete |
| INFRA-06 | Phase 2 | Complete |
| TASK-01 | Phase 3 | Pending |
| TASK-02 | Phase 3 | Pending |
| TASK-03 | Phase 3 | Pending |
| TASK-04 | Phase 3 | Pending |
| TASK-05 | Phase 3 | Pending |
| TASK-06 | Phase 3 | Pending |
| TASK-07 | Phase 3 | Pending |
| CTX-01 | Phase 3 | Pending |
| CTX-02 | Phase 3 | Pending |
| CTX-03 | Phase 3 | Pending |
| LIVE-01 | Phase 1 | Complete |
| LIVE-02 | Phase 1 | Complete |
| LIVE-03 | Phase 1 | Complete |
| LIVE-04 | Phase 1 | Complete |

| FILL-01 | Phase 4 | Pending |
| FILL-02 | Phase 4 | Pending |
| FILL-03 | Phase 4 | Pending |
| FILL-04 | Phase 4 | Pending |
| FILL-05 | Phase 4 | Pending |
| STT-01 | Phase 6 | Complete (pre-work) |
| STT-02 | Phase 6 | Pending |
| FLOW-01 | Phase 6 | Complete (pre-work) |
| FLOW-02 | Phase 6 | Complete (pre-work) |
| BARGE-01 | Phase 5 | Pending |
| BARGE-02 | Phase 5 | Pending |
| BARGE-03 | Phase 5 | Pending |
| BARGE-04 | Phase 5 | Pending |
| OVL-01 | Phase 6 | Complete (pre-work) |
| OVL-02 | Phase 6 | Complete (pre-work) |

**Coverage:**
- v1.0 requirements: 24 total, all mapped
- v1.1 requirements: 15 total, all mapped to phases 4-6
- Unmapped: 0

---
*Requirements defined: 2026-02-13*
*Last updated: 2026-02-17 — v1.1 requirements added*
