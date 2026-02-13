# Codebase Structure

**Analysis Date:** 2026-02-13

## Directory Layout

```
push-to-talk/
├── push-to-talk.py          # Main service: hotkey listener, mode orchestrator
├── indicator.py             # Status UI: GTK indicator (floating dot or tray)
├── openai_realtime.py       # OpenAI Realtime API wrapper with tool support
├── vocabulary.txt           # Custom vocabulary words for Whisper
├── vocabulary.txt.example   # Template/documentation for vocabulary
├── config.json              # Configuration (created at first run)
├── status                   # Runtime status file (read by indicator)
├── requirements.txt         # Python dependencies
├── install.sh               # Installation script
├── uninstall.sh             # Uninstall script
├── update.sh                # Update script
├── sync.sh                  # Sync script (website integration)
├── push-to-talk.service     # Systemd user service
├── push-to-talk-update.service # Auto-update service
├── push-to-talk-update.timer    # Auto-update schedule
├── push-to-talk-watcher.service # File watch service
├── push-to-talk-watcher.sh      # File watch script
├── README.md                # User documentation
├── CLAUDE.md                # Project instructions (read by Interview context feature)
└── .planning/
    └── codebase/
        ├── ARCHITECTURE.md  # This analysis
        └── STRUCTURE.md     # This file
```

## Directory Purposes

**Project Root (`/home/ethan/code/push-to-talk/`):**
- Purpose: All code, config, and runtime files in single location
- Contains: Python modules, configuration, service files
- Key files: `push-to-talk.py`, `indicator.py`, `openai_realtime.py`

**User Config/Data Dirs (created at runtime):**
- `~/.config/openai/api_key` - OpenAI API key
- `~/.config/systemd/user/push-to-talk.service` - Systemd service definition
- `~/.local/share/push-to-talk/` - Session directories, venv
- `~/Audio/push-to-talk/` - Audio recordings (if save_audio enabled)
  - `sessions/<timestamp>/` - Interview/Conversation session output
- `.planning/codebase/` - Analysis documents (for GSD)

## Key File Locations

**Entry Points:**
- `push-to-talk.py`: Main service process (hotkey listener entry point)
- `indicator.py`: Status indicator UI process (spawned by main)
- `openai_realtime.py`: Standalone module for Realtime API (imported by main)

**Configuration:**
- `config.json`: User settings (modes, hotkeys, TTS backend, etc.)
- `vocabulary.txt`: Custom words for transcription accuracy
- `push-to-talk.service`: Systemd unit file
- `CLAUDE.md`: Project context (read by Interview mode for context_dirs)

**Core Logic:**
- `push-to-talk.py` classes:
  - `VocabularyManager` (line 478): Vocabulary learning
  - `InterviewSession` (line 535): Interview mode state
  - `ConversationSession` (line 628): Conversation mode state
  - `PushToTalk` (line 662): Main orchestrator
- `openai_realtime.py` classes:
  - `RealtimeSession` (line 223): OpenAI Realtime API wrapper

**Testing & Utilities:**
- `install.sh`: Sets up venv, installs dependencies
- `uninstall.sh`: Removes service and files
- `requirements.txt`: Python package dependencies

## Naming Conventions

**Files:**
- Module files: lowercase with hyphens (`push-to-talk.py`)
- Config/data files: lowercase with underscores (`vocabulary.txt`, `config.json`)
- Service files: lowercase with hyphens (`push-to-talk.service`)
- Documentation: UPPERCASE.md

**Classes:**
- Main services: `PushToTalk`, `StatusIndicator`, `RealtimeSession`
- Session managers: `InterviewSession`, `ConversationSession`
- Utilities: `VocabularyManager`
- UI windows: `SettingsWindow`, `StatusPopup`, `QuickControlWindow`

**Functions:**
- Utility functions: snake_case (`load_config`, `speak_openai`, `get_audio_level`)
- Handlers: `on_<event>` pattern (`on_press`, `on_release`, `on_draw`)
- Special methods: Leading underscore for internal methods (`_interview_generate_question`)

**Variables:**
- Configuration dicts: UPPERCASE constants (`MODIFIER_KEY_OPTIONS`, `COLORS`)
- File paths: UPPERCASE constants (`BASE_DIR`, `CONFIG_FILE`, `VOCAB_FILE`)
- Module state: lowercase (`indicator_process`, `realtime_session`)

**Status Values:**
- 'idle' - Waiting for input
- 'recording' - Currently recording audio
- 'processing' - Transcribing or calling AI
- 'success' - Operation succeeded (brief flash)
- 'error' - Error occurred
- 'listening' - AI Realtime listening for input (blue)
- 'speaking' - AI Realtime speaking (purple)

## Where to Add New Code

**New Dictation Feature (e.g., smart corrections):**
- Primary code: `push-to-talk.py` - add to `PushToTalk.transcribe_and_type()` method or create new helper
- Voice command detection: Add pattern to `CORRECTION_PATTERNS` (line 411)
- Tests: Manual test via PTT, check logs with `journalctl --user -u push-to-talk -f`

**New AI Mode:**
- Implementation: `push-to-talk.py` - create new Session class (e.g., `MyModeSession` at line 628+)
- Add mode constant to config defaults (line 216-240)
- Route in `on_press()` method (line 2088-2135)
- Add corresponding `_my_mode_*()` methods for state transitions
- Update indicator UI in `indicator.py` to display new mode option (line 310-320)

**New Voice Command:**
- Dictation commands: Add to `transcribe_and_type()` conditional chain (line 966-1046)
- AI commands: Add to relevant `*_process_answer()` or `transcribe_and_ask_ai()` method
- Built-in commands: Pattern match in text_lower, set status, return early

**New TTS Backend:**
- Implementation: Add method like `speak_<backend>()` (follow pattern of `speak_openai` at line 260)
- Integration: Modify `speak()` method (line 828) to dispatch to new backend
- Configuration: Add option to config defaults and indicator UI

**Utilities/Helpers:**
- General utilities: Module-level functions (line 308-408) like `show_prompt_dialog`, `check_verbal_hooks`
- Audio utilities: Functions like `get_audio_level()` (line 428)
- Indicator/UI utilities: Add to `indicator.py` (line 52-100 for config loading)

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD (Getting Stuff Done) analysis documents
- Generated: By `/gsd:map-codebase` command
- Committed: Yes (checked into git)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

**`claude-session/` (created at runtime):**
- Purpose: Persistent Claude CLI session directory for multi-turn conversations
- Generated: By Interview/Conversation mode
- Committed: No (runtime/temporary)
- Contents: System prompt, conversation history (in Claude's session store)

**`sessions/<timestamp>/` (created in ~/Audio/push-to-talk/):**
- Purpose: Interview mode output directory
- Generated: By Interview mode on-demand
- Committed: No (user data)
- Contents:
  - `001_interviewer.wav`, `002_user.wav`, ... - sequenced audio files
  - `.normalized/` - intermediate normalized audio
  - `.concat_list.txt` - ffmpeg concat list
  - `interview_<id>.wav` - final stitched audio
  - `interview_<id>.mp3` - MP3 version
  - `transcript.md` - Full conversation transcript
  - `show-notes.md` - AI-generated show notes
  - `metadata.json` - Session metadata

**`.normalized/` (temporary):**
- Purpose: Intermediate directory for audio normalization during interview post-processing
- Generated: By `_post_process_interview()` (line 1673)
- Deleted: After stitching complete (line 1740)
- Committed: No (automatically cleaned up)

## Testing Locations

**Manual Testing:**
- Logs: `journalctl --user -u push-to-talk -f`
- Status indicator: Floating dot or tray icon (configurable)
- Audio: PTT hotkey with echo=off to test recording
- UI: Right-click indicator for Settings > Hotkeys

**Testing Files:**
- No automated test suite present
- Vocabulary file: `vocabulary.txt.example` shows expected format
- Config validation: Hotkey conflict detection (line 700)

**Debug Mode:**
- Enable: Settings > Advanced > Debug Mode (sets `debug_mode: true` in config)
- Output: All logs go to systemd journal
- Info level: Printed with `print(..., flush=True)` statements throughout

## Configuration Structure

**Config Keys (config.json):**
```json
{
  "tts_backend": "piper",                    # "piper" or "openai"
  "openai_voice": "nova",                    # If tts_backend=openai
  "ai_mode": "claude",                       # "claude", "realtime", "interview", "conversation"
  "ptt_key": "ctrl_r",                       # Hotkey for dictation
  "ai_key": "shift_r",                       # Hotkey modifier for AI mode
  "interrupt_key": "escape",                 # Stop AI response
  "indicator_style": "floating",             # "floating" or "tray"
  "indicator_x": 1920,                       # Indicator position (null = auto)
  "indicator_y": 35,
  "smart_transcription": false,              # AI-based error correction
  "dictation_mode": "live",                  # "live", "prompt", "stream"
  "save_audio": false,                       # Save recordings to disk
  "audio_dir": "~/Audio/push-to-talk",       # Save location
  "interview_topic": "",                     # Pre-set topic
  "interview_context_dirs": [],              # Repos for context reading
  "conversation_project_dir": "",            # Project directory for conversation mode
  "verbose_hooks": [                         # Custom voice commands
    {"trigger": "search for *", "command": "xdg-open 'https://google.com/search?q={}'"}
  ]
}
```

## Design Patterns Used

**State Machine:**
- Recording state: `self.recording` boolean + recording process tracking
- Interview state: `InterviewSession.status` (starting, idle, recording, processing)
- Realtime state: `RealtimeSession.running` + `playing_audio` flag

**Threading:**
- Main thread: Hotkey listener (blocks in `listener.join()`)
- Worker threads: Spawned for transcription, AI calls, post-processing
- Lock: `self.model_lock` guards Whisper model access

**Configuration Management:**
- Load at startup: `load_config()` merges defaults with saved config
- Hot-reload: `start_recording()` reloads config to pick up indicator changes
- Persistence: `save_config()` writes back to config.json

**Subprocess Orchestration:**
- Recording: `pw-record` for audio capture
- Conversion: `ffmpeg` for format conversion
- Playback: `aplay` for audio output
- Desktop: `xdotool` for keyboard input, `zenity` for dialogs, `notify-send` for notifications
- External tools: Claude CLI via subprocess.run with session persistence

**Factory Pattern (implicit):**
- `speak()` method dispatches to backend implementation based on config
- `on_press()` routes to different start methods based on hotkey combinations

---

*Structure analysis: 2026-02-13*
