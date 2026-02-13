# Architecture

**Analysis Date:** 2026-02-13

## Pattern Overview

**Overall:** Event-driven, modular microservice architecture with multiple specialized modes

**Key Characteristics:**
- Global hotkey listener triggers state machines for different AI/dictation modes
- Thread-based background processing for transcription and response generation
- Configuration-driven mode selection (Claude, Realtime, Interview, Conversation)
- Subprocess orchestration for audio recording, transcription, and TTS
- Session-based persistence for Interview and Conversation modes

## Layers

**Input/Hotkey Layer:**
- Purpose: Detect and respond to global keyboard events
- Location: `push-to-talk.py` - `PushToTalk.on_press()` / `PushToTalk.on_release()`
- Contains: Hotkey mapping, mode detection, state machine logic
- Depends on: pynput library for keyboard monitoring
- Used by: Application entry point; all other layers respond to key events

**Audio Recording & Processing Layer:**
- Purpose: Capture microphone input and convert to transcribable format
- Location: `push-to-talk.py` - `PushToTalk.start_recording()`, `PushToTalk.stream_mode_*`
- Contains: pw-record subprocess calls, audio format conversion via ffmpeg
- Depends on: PipeWire audio system, ffmpeg
- Used by: All transcription and AI modes

**Transcription Layer:**
- Purpose: Convert recorded audio to text using Whisper model
- Location: `push-to-talk.py` - Whisper model instantiation and `transcribe()` calls
- Contains: Model loading, vocabulary context building, language detection
- Depends on: OpenAI Whisper (loaded once at startup)
- Used by: Dictation mode, AI modes, Interview/Conversation modes

**Dictation Layer:**
- Purpose: Type transcribed text directly into focused application
- Location: `push-to-talk.py` - `PushToTalk.transcribe_and_type()`
- Contains: Voice command parsing, prompt dialog (zenity), xdotool typing
- Depends on: xdotool (X11 automation), vocabulary learning
- Used by: Direct user invocation via PTT key

**AI Assistant Layer:**
- Purpose: Process voice questions and generate spoken responses
- Locations:
  - Claude mode: `push-to-talk.py` - `PushToTalk.transcribe_and_ask_ai()`
  - Realtime mode: `openai_realtime.py` - `RealtimeSession` class
  - Interview mode: `push-to-talk.py` - `PushToTalk._interview_*()` methods
  - Conversation mode: `push-to-talk.py` - `PushToTalk._conversation_*()` methods
- Contains: Claude CLI invocation, OpenAI Realtime API, session management
- Depends on: Claude CLI, OpenAI APIs, subprocess orchestration
- Used by: AI key presses, auto-listen for follow-ups

**Text-to-Speech Layer:**
- Purpose: Convert AI responses or system messages to audio playback
- Location: `push-to-talk.py` - `PushToTalk.speak()`, `PushToTalk.speak_openai()`
- Contains: Backend selection (Piper local vs OpenAI cloud), audio output via aplay
- Depends on: Piper TTS models (local) or OpenAI API
- Used by: AI assistant responses, system notifications

**Session Management Layer:**
- Purpose: Maintain state and output for Interview and Conversation modes
- Locations: `push-to-talk.py` - `InterviewSession` and `ConversationSession` classes
- Contains: Session directories, transcript logging, metadata tracking, post-processing
- Depends on: Filesystem, Claude CLI, ffmpeg for audio stitching
- Used by: Interview/Conversation modes for persistence and output generation

**UI/Status Indicator Layer:**
- Purpose: Visual feedback and configuration management
- Location: `indicator.py` (separate process), `push-to-talk.py` - `set_status()`
- Contains: GTK UI, status dot drawing, settings windows, tray/floating modes
- Depends on: GTK 3, Cairo, AppIndicator3 (optional)
- Used by: Status file communication, user interaction

## Data Flow

**Dictation Flow (Live Mode):**

1. User holds PTT key (Right Ctrl)
2. `on_press()` → `start_recording()` starts `pw-record` subprocess
3. Status indicator shows "recording" (red dot)
4. User releases PTT key
5. `on_release()` → `stop_recording_with_mode()` stops recording
6. Audio file converted to 16kHz mono via ffmpeg
7. Whisper transcribes with vocabulary prompt
8. Text parsed for voice commands (clear, undo, paste, etc.)
9. If command matched: execute and return
10. If correction detected (add word/correction:): learn and return
11. If prompt mode: show zenity dialog for editing
12. xdotool types text into focused window
13. Status shows "success" (green dot) for 1.5s, then idle

**AI Assistant Flow (Claude + Whisper):**

1. User holds PTT + AI key (Right Ctrl + Right Shift)
2. `on_press()` → `start_recording()` (same as dictation)
3. User releases, `stop_recording_with_mode()` → `transcribe_and_ask_ai()`
4. Whisper transcribes user question
5. Claude CLI invoked with question in session directory
6. Response captured from stdout
7. TTS backend (Piper or OpenAI) speaks response
8. `auto_listen_for_followup()` records for 4 seconds
9. If audio detected and level > threshold: transcribe and recurse
10. Else: return to idle

**Interview Mode Flow:**

1. User activates AI key → `start_interview()`
2. Zenity prompt for topic (or use config)
3. `InterviewSession` created with session_dir at `~/Audio/push-to-talk/sessions/<timestamp>/`
4. Claude generates first question
5. TTS speaks question, saves audio to `001_interviewer.wav`
6. User presses PTT to record answer
7. Whisper transcribes, saved to `002_user.wav`
8. Detect wrap signal ("that's a wrap") or continue
9. Claude generates follow-up question using conversation history
10. On wrap: Claude generates closing statement
11. Post-processing: ffmpeg stitches audio, generates transcript and show notes
12. Output: `interview_<id>.wav`, `interview_<id>.mp3`, `transcript.md`, `show-notes.md`, `metadata.json`

**Conversation Mode Flow:**

1. User activates AI key → `start_conversation()`
2. Zenity prompt for project directory
3. `ConversationSession` created
4. TTS speaks "Ready. What would you like to know?"
5. Auto-listen for first question
6. Claude CLI invoked with `--permission-mode bypassPermissions` in project directory
7. Full tool access available (file read/write, command execution)
8. TTS speaks response (stripped of thinking blocks)
9. Auto-listen loop continues until user says "goodbye"
10. End notification shows turn count

**Realtime Mode Flow (OpenAI GPT-4o):**

1. User activates AI key → `start_realtime_session()`
2. `RealtimeSession.run()` establishes WebSocket to OpenAI API
3. Audio continuously streamed via `pw-record` (24kHz mono)
4. Mic muted during AI response to prevent echo
5. WebSocket receives audio_delta events (streaming TTS)
6. aplay subprocess plays audio in real-time
7. Tool calls (function_call events) detected and executed
8. Results sent back to API
9. Interrupt key (Esc) can stop current response
10. Toggle hotkey again to stop session, unmute mic

**State Management:**

- **Status File:** `push-to-talk/status` - read by indicator every 100ms
- **Config:** `push-to-talk/config.json` - loaded at startup, can be hot-reloaded
- **Vocabulary:** `push-to-talk/vocabulary.txt` - appended on corrections
- **Session Dirs:** `~/Audio/push-to-talk/sessions/<id>/` - persistent across sessions
- **Claude Session:** `push-to-talk/claude-session/` - persists for multi-turn conversations

## Key Abstractions

**VocabularyManager:**
- Purpose: Learn custom words to improve Whisper transcription accuracy
- Examples: `push-to-talk.py` line 478
- Pattern: Load from file, add via voice command, generate initial_prompt for Whisper

**InterviewSession:**
- Purpose: Manage podcast interviewer mode with persistent output
- Examples: `push-to-talk.py` line 535
- Pattern: State machine (starting→idle→recording→processing), transcript accumulation, post-processing pipeline

**ConversationSession:**
- Purpose: Manage voice conversation with Claude + full tool access
- Examples: `push-to-talk.py` line 628
- Pattern: Turn counting, end signal detection, system prompt generation

**RealtimeSession:**
- Purpose: WebSocket-based low-latency voice conversation with OpenAI
- Examples: `openai_realtime.py` line 223
- Pattern: Async event loop, dual audio I/O (record + play), tool execution, mic control

**PushToTalk:**
- Purpose: Main orchestrator — handles all hotkey input and mode routing
- Examples: `push-to-talk.py` line 662
- Pattern: Global state machine, thread spawning for background tasks, process management

## Entry Points

**`push-to-talk.py` (main service):**
- Location: `/home/ethan/code/push-to-talk/push-to-talk.py`
- Triggers: Systemd user service auto-start
- Responsibilities: Hotkey monitoring, mode orchestration, background processing

**`indicator.py` (status UI):**
- Location: `/home/ethan/code/push-to-talk/indicator.py`
- Triggers: Started by `push-to-talk.py`
- Responsibilities: Visual status indicator, settings GUI, configuration UI

**Systemd Service:**
- Location: `~/.config/systemd/user/push-to-talk.service`
- Triggers: User login (enabled with `--user-name`)
- Responsibilities: Daemonization, restart on crash

## Error Handling

**Strategy:** Local try-except with status fallback, desktop notifications for user-facing errors

**Patterns:**

- Transcription failures: Log error, set status to 'error', brief notification, return to idle
- API timeouts (Claude/OpenAI): 120s timeout for Claude CLI, 30s for tool execution
- Audio level detection: Default to silence on sox error (assume silence rather than crash)
- Missing dependencies: Check at startup, fall back to alternative mode (e.g., Piper if OpenAI fails)
- File I/O: Suppress exceptions in cleanup paths, check file existence before processing

Example at `push-to-talk.py` line 1436:
```python
except Exception as e:
    print(f"Error during AI transcription: {e}", flush=True)
    set_status('error')
finally:
    for f in [temp_file.name, temp_file.name.replace('.wav', '_16k.wav')]:
        try:
            os.unlink(f)
        except:
            pass
```

## Cross-Cutting Concerns

**Logging:** All operations logged to systemd journal via `print(..., flush=True)`
- View: `journalctl --user -u push-to-talk -f`

**Validation:** Hotkey conflict detection (PTT and AI keys must differ)
- Implemented at `push-to-talk.py` line 700-703

**Authentication:** OpenAI API key sourced from environment or `~/.config/openai/api_key`
- Never passed as CLI argument; uses subprocess env inheritance

**Audio Safety:** Mic physically muted during Realtime AI speech to prevent echo
- Controlled via pactl: `pactl set-source-mute @DEFAULT_SOURCE@ {0|1}`

**Thread Safety:** Model lock guards Whisper model access (single instance, concurrent requests possible)
- `self.model_lock = threading.Lock()` used in transcription methods

---

*Architecture analysis: 2026-02-13*
