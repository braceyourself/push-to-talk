# Push-to-Talk: Live Mode

## What This Is

A new "live" dictation mode for push-to-talk that provides real-time voice conversation with AI via the OpenAI Realtime API, with the ability to manage async Claude CLI tasks. You talk, the AI responds through speakers, and it can spawn, monitor, and report on Claude CLI processes running in the background — acting as a voice-controlled task orchestrator. The current "live" mode gets renamed to "dictate."

## Core Value

Real-time voice-to-voice AI conversation that can delegate real work to Claude CLI and manage multiple async tasks with clean context isolation.

## Requirements

### Validated

- ✓ OpenAI Realtime API WebSocket integration — existing (`openai_realtime.py`)
- ✓ Audio recording via PipeWire — existing
- ✓ TTS playback via aplay — existing
- ✓ Mic muting during AI speech — existing
- ✓ Function calling / tool execution — existing
- ✓ Dictation modes (live/prompt/stream) — existing
- ✓ Settings UI for mode selection — existing (`indicator.py`)

### Active

- [ ] Rename current "live" mode to "dictate" across codebase and UI
- [ ] New "live" mode that starts an OpenAI Realtime voice session
- [ ] Hold PTT to speak, release to send — AI responds through speakers
- [ ] Session memory — conversation persists across PTT presses within a session
- [ ] Claude CLI task spawning — voice commands spawn Claude CLI processes with prompts
- [ ] Async task management — Claude processes run in background, don't block conversation
- [ ] Task status awareness — AI knows what tasks are running, completed, or failed
- [ ] Context switching — user can refer to different tasks by name/description and AI tracks them
- [ ] Context isolation — each Claude CLI task runs in its own context, no bleed between tasks
- [ ] Task result summarization — AI reads Claude CLI output and speaks a summary

### Out of Scope

- Interactive Claude CLI sessions (steering a running Claude session with voice) — complexity too high for v1
- Shell commands beyond Claude CLI — keep focused on Claude as the work engine
- Persistent task state across live mode sessions — each session starts fresh
- Audio recording/saving of live mode sessions — not a podcast, just a working session

## Context

Push-to-talk is a Linux desktop tool (X11, PipeWire, systemd user service) that provides global hotkey-driven voice input. It already has multiple AI modes (Claude, Realtime, Interview, Conversation) and dictation modes (live, prompt, stream). The `openai_realtime.py` module already implements WebSocket streaming, mic control, and function calling against the Realtime API. The main work is: (1) renaming the current live mode, (2) elevating the Realtime session into a first-class dictation mode with task orchestration capabilities, and (3) building the async Claude CLI management layer with context isolation.

## Constraints

- **Platform**: Linux X11 with PipeWire — no cross-platform concerns
- **API**: OpenAI Realtime API (WebSocket) for voice, Claude CLI for task execution
- **Architecture**: Must integrate with existing hotkey/mode system in `push-to-talk.py` and `indicator.py`
- **Async**: Claude CLI tasks must be non-blocking — user must be able to keep talking while tasks run

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rename "live" → "dictate" | Free up "live" name for the more fitting real-time voice mode | — Pending |
| OpenAI Realtime as voice layer, Claude CLI as work layer | Realtime gives low-latency voice; Claude gives deep code capabilities | — Pending |
| Hold-to-talk (not toggle or always-listening) | Consistent with existing PTT UX, user is familiar with it | — Pending |
| Each Claude task gets isolated context | Prevents confusion when multiple tasks run concurrently | — Pending |

---
*Last updated: 2026-02-13 after initialization*
