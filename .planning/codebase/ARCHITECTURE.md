# Architecture

**Analysis Date:** 2026-02-03

## Pattern Overview

**Overall:** Event-driven service with plugin-like mode switching (Push-to-Talk dictation, AI conversation, Realtime voice interaction)

**Key Characteristics:**
- Keyboard-driven state machine (recording states triggered by hotkey combinations)
- Dual-mode audio processing: transcribe-then-type vs transcribe-then-ask-ai
- Pluggable TTS backend (local Piper or cloud OpenAI)
- Separate indicator/UI process communicating via status file
- Optional voice-to-voice AI using OpenAI Realtime API

## Layers

**Input Layer (Keyboard & Audio):**
- Purpose: Capture user input and system state changes
- Location: `push-to-talk.py` lines 855-941 (on_press/on_release), `openai_realtime.py` lines 440-468 (audio recording)
- Contains: Keyboard listener using pynput, audio recording via pw-record, WebSocket event handling
- Depends on: pynput library, PipeWire audio system
- Used by: State machine to trigger processing workflows

**Processing Layer (Speech-to-Text & AI):**
- Purpose: Convert audio to text, optionally delegate to AI, generate responses
- Location: `push-to-talk.py` lines 548-853 (transcribe_and_type, transcribe_and_ask_ai, handle_ai_response, auto_listen_for_followup)
- Contains: Whisper transcription (lines 578-584), Claude CLI invocation (lines 810-821), TTS response handling (lines 837-853)
- Depends on: Whisper model, Claude CLI, Piper/OpenAI TTS, ffmpeg for audio conversion
- Used by: State machine after recording completes

**UI/Output Layer:**
- Purpose: Visual feedback and user settings
- Location: `indicator.py` (status indicator), `openai_realtime.py` (audio playback)
- Contains: GTK3 floating dot or system tray status widget, settings window with tabs, status file polling
- Depends on: GTK3, AppIndicator3 (optional), cairo for drawing
- Used by: Main service communicates status via file; indicator runs in separate process

**AI Mode Layer (Optional Realtime):**
- Purpose: Low-latency voice-to-voice interaction using OpenAI Realtime API
- Location: `openai_realtime.py` (entire module)
- Contains: WebSocket connection to Realtime API, function calling for tool execution (run_command, read_file, write_file, ask_claude, remember, recall)
- Depends on: websockets library, OpenAI API key
- Used by: Main service when ai_mode='realtime' and keys are pressed

**Vocabulary & Learning Layer:**
- Purpose: Improve transcription accuracy by learning domain-specific words
- Location: `push-to-talk.py` lines 285-334 (VocabularyManager class)
- Contains: Vocabulary file I/O, prompt generation for Whisper, correction pattern detection
- Depends on: vocabulary.txt file
- Used by: Transcription pipelines include vocab in initial_prompt

## Data Flow

**Basic Dictation Flow:**
1. User holds PTT key (Right Ctrl) → `on_press()` triggers `start_recording()` (line 910-912)
2. Audio captured to temp WAV file via `pw-record` (lines 536-542)
3. User releases PTT key → `on_release()` stops recording (lines 924-938)
4. Background thread calls `transcribe_and_type()` (line 964)
   - Audio converted to 16kHz mono with ffmpeg (lines 563-567)
   - Whisper transcribes with vocabulary prompt (lines 578-584)
   - Correction patterns checked (lines 592-600)
   - Text typed via xdotool (line 606)

**AI Assistant Flow (Claude Mode):**
1. User holds PTT key + AI key (Right Ctrl + Right Shift) → `on_press()` sets ai_mode=True, calls `start_recording()` (line 900)
2. User releases keys → `on_release()` calls `stop_recording_with_mode()` which spawns `transcribe_and_ask_ai()` (line 962)
3. Audio transcribed (same as above)
4. Claude CLI invoked with question in persistent session (lines 810-821)
5. Response captured and passed to `handle_ai_response()` (line 825)
6. `speak()` plays response via TTS (Piper or OpenAI) (lines 491-505)
7. `auto_listen_for_followup()` records for 4 seconds (lines 637-719)
8. If audio detected, recursively calls `process_ai_question()` (line 707)

**AI Assistant Flow (Realtime Mode):**
1. User holds PTT key + AI key → `on_press()` detects both keys held (line 871)
2. Calls `start_realtime_session()` which spawns async event loop in daemon thread (lines 431-464)
3. `RealtimeSession.run()` establishes WebSocket, starts audio recording/sending (lines 469-490)
4. Background tasks: `record_and_send()` streams mic audio, `handle_events()` processes API responses (lines 318-468)
5. When API sends audio delta events, mic is muted to prevent echo (lines 344-346)
6. When user stops speaking, function calls are executed (lines 373-399)
7. User presses interrupt key → `request_interrupt()` sets flag, `handle_events()` cancels response (lines 506-508, 327-329)
8. User holds both keys again → toggles off realtime session (lines 872-875)

**State Management:**
- Recording state tracked in instance vars: `self.recording`, `self.ai_mode`, `self.ctrl_r_pressed`, `self.shift_r_pressed` (lines 337-349)
- Status communicated to indicator via `status` file: 'idle', 'recording', 'processing', 'success', 'error', 'listening', 'speaking' (lines 227-232, set_status)
- Indicator polls status file every 100ms (indicator.py line 925)
- Config persisted to JSON: `config.json` (lines 151-180)

## Key Abstractions

**PushToTalk Class:**
- Purpose: Main orchestrator for dictation and AI modes
- Examples: `push-to-talk.py` lines 336-969
- Pattern: Singleton service managing keyboard listener, Whisper model, config, threading

**VocabularyManager:**
- Purpose: Load/save custom words, generate Whisper prompt hints
- Examples: `push-to-talk.py` lines 285-334
- Pattern: Simple file-backed in-memory set with persistence

**RealtimeSession:**
- Purpose: Manage OpenAI Realtime API connection and event loop
- Examples: `openai_realtime.py` lines 223-527
- Pattern: Async context manager for WebSocket connection, thread-safe state (interrupt flag)

**StatusIndicator / TrayIndicator:**
- Purpose: Visual status display and settings UI
- Examples: `indicator.py` lines 854-1210 (floating), 1068-1190 (tray)
- Pattern: Gtk.Window with cairo drawing, polling for status changes, modal SettingsWindow

## Entry Points

**Main Service:**
- Location: `push-to-talk.py` lines 971-989 (main function)
- Triggers: Systemd user service `push-to-talk.service`
- Responsibilities: Initialize indicator, start keyboard listener, run event loop indefinitely

**Status Indicator Process:**
- Location: `indicator.py` lines 1193-1210 (main function)
- Triggers: Spawned by push-to-talk.py as subprocess (line 260)
- Responsibilities: Display status dot, handle clicks/drags, show settings window, poll status file

**OpenAI Realtime Session:**
- Location: `openai_realtime.py` lines 469-490 (RealtimeSession.run)
- Triggers: Called from PushToTalk.on_press when both modifier keys pressed (line 897)
- Responsibilities: Connect to API, stream audio bidirectionally, handle function calls, manage mic mute

## Error Handling

**Strategy:** Defensive with fallback modes

**Patterns:**
- Missing dependencies: Try/except on imports (lines 27-39, 15-19 in openai_realtime.py), fallback to simpler modes
- API key missing: Prompt user in terminal (lines 55-111 in push-to-talk.py)
- Recording too short: Check file size before transcribing (lines 554-559)
- Whisper model lock: Use threading.Lock to prevent concurrent access (line 342)
- TTS failure: Log error, continue (lines 213-215 in push-to-talk.py)
- Status file I/O: Bare except blocks (lines 231-232, 279-282)
- WebSocket connection errors: Catch ConnectionClosed, log and exit gracefully (openai_realtime.py line 435)

## Cross-Cutting Concerns

**Logging:** Print statements to stdout (captured by journalctl via systemd)

**Validation:** Config type checking (validate ptt_key != ai_key, lines 362-366); hotkey validation in indicator UI (lines 535-539)

**Authentication:** OpenAI API key loaded from env or files (lines 41-52, openai_realtime.py lines 511-523); saved locally with restricted permissions (indicator.py line 99)

**Audio Management:** PipeWire (pw-record), aplay for playback; mic muting during TTS/Realtime AI speaking (lines 344-346, 407, 424)

**Configuration:** JSON file at `~/.local/share/push-to-talk/config.json` with defaults, hot-reload on indicator settings changes

---

*Architecture analysis: 2026-02-03*
