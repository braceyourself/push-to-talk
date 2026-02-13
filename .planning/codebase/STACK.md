# Technology Stack

**Analysis Date:** 2026-02-13

## Languages

**Primary:**
- Python 3.10+ - Main service code, all business logic
  - `push-to-talk.py` (2237 lines) - Core dictation and AI orchestration
  - `indicator.py` (1660 lines) - GTK status indicator UI
  - `openai_realtime.py` (527 lines) - OpenAI Realtime API integration

## Runtime

**Environment:**
- Linux (X11 with PipeWire audio)
- systemd user service for auto-start
- Requires DISPLAY for X11 interaction

**Package Manager:**
- pip (Python package management)
- Lockfile: Not detected (requirements.txt used instead)

## Frameworks

**Core:**
- OpenAI Python SDK (openai) - TTS, GPT-4o-mini completions, Realtime API
- Whisper (openai-whisper) - Local speech recognition
- Piper TTS (piper-tts) - Local text-to-speech fallback

**UI:**
- GTK 3 (gi.repository) - Status indicator window and settings dialogs
- Cairo (cairo) - Custom rendering for colored status dot
- AppIndicator3 (optional) - System tray integration

**System Integration:**
- pynput - Keyboard/mouse input detection and emulation
- websockets - OpenAI Realtime API WebSocket connections
- PulseAudio (pactl) - Microphone mute/unmute control
- PipeWire (pw-record) - Audio recording

**Testing:**
- Not detected

**Build/Dev:**
- Not detected (no build tool)

## Key Dependencies

**Critical:**
- `openai` (latest) - Required for TTS (OpenAI voices), Realtime API, smart transcription
- `openai-whisper` - Speech recognition (local, no API calls)
- `pynput` - Keyboard listener for PTT hotkeys
- `websockets` - WebSocket protocol for Realtime API
- `piper-tts` - Local TTS fallback when OpenAI unavailable

**Infrastructure:**
- `pathvalidate` - Path validation for audio file saving
- `numpy` - Audio processing helpers (noise detection, optional)

## Configuration

**Environment:**
- OpenAI API key: `$OPENAI_API_KEY` env var OR `~/.config/openai/api_key` file
- Runtime config: `config.json` in service directory (JSON)
  - Loaded with defaults, user overrides merged in
  - See `load_config()` in `push-to-talk.py` lines 215-248

**Build:**
- No build config files
- Service installed via manual installation or `install.sh`
- Configuration persists across runs in `config.json`

**Key Config Options:**
```
- tts_backend: "piper" | "openai"
- ai_mode: "claude" | "realtime" | "interview"
- ptt_key: hotkey for push-to-talk (default "ctrl_r")
- ai_key: hotkey for AI mode (default "shift_r")
- smart_transcription: Use OpenAI to fix transcription errors
- dictation_mode: "live" | "prompt" | "stream"
- save_audio: Save recorded audio to disk
- audio_dir: Where to save sessions (default "~/Audio/push-to-talk")
- interview_topic: Pre-set topic for interview mode
- conversation_project_dir: Project directory for Claude conversation mode
- indicator_style: "floating" or "tray"
```

## Platform Requirements

**Development:**
- Python 3.10+
- Linux (X11 required for keyboard listener and xdotool)
- GTK 3 development files
- PipeWire or PulseAudio audio system
- System tools: xdotool, ffmpeg, sox, zenity, notify-send, pactl, pw-record

**Production:**
- Deployment target: Linux desktop (tested on systemd-based systems)
- Runs as systemd user service: `~/.config/systemd/user/push-to-talk.service`
- Must be run in user session with DISPLAY and XAUTHORITY set

---

*Stack analysis: 2026-02-13*
