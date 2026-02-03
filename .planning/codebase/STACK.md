# Technology Stack

**Analysis Date:** 2026-02-03

## Languages

**Primary:**
- Python 3.10+ - Core application, service, and UI components
- Bash - Installation and setup scripts

## Runtime

**Environment:**
- Python 3.10+ (required for f-strings and typing features)

**Package Manager:**
- pip (Python dependency management)
- Lockfile: `requirements.txt` (present)

## Frameworks

**Core Services:**
- OpenAI Whisper - Local speech-to-text recognition (`openai-whisper` package)
- OpenAI Realtime API - Voice-to-voice AI conversations (via `openai` SDK with websockets)
- Claude CLI - AI assistant backend for text processing

**Audio:**
- PipeWire - Audio capture and playback (`pw-record`, `aplay` commands)
- FFmpeg - Audio format conversion (WAV to 16kHz mono)
- Piper TTS - Local text-to-speech synthesis (`piper-tts` package)
- OpenAI TTS API - Cloud text-to-speech alternative

**Desktop Integration:**
- GTK 3 (`gi` bindings via `python3-gi`) - Status indicator UI
- xdotool - Keyboard input simulation for typing transcribed text
- systemd user services - Auto-start and lifecycle management

**Input Handling:**
- pynput - Low-level keyboard listener for global hotkeys

## Key Dependencies

**Critical:**
- `openai-whisper` - Speech recognition (small model by default, ~140MB)
- `openai` [1.0+] - OpenAI API client for TTS and Realtime API
- `pynput` - Global keyboard event capture (enables push-to-talk)
- `websockets` - WebSocket protocol for Realtime API (optional but required for realtime mode)
- `piper-tts` - Local TTS engine
- `pathvalidate` - File path validation for TTS output

**Infrastructure:**
- `numpy` - Audio processing helpers (optional, for noise detection)

**System Binaries (not Python packages):**
- `ffmpeg` - Audio conversion
- `pw-record` - PipeWire audio recording
- `aplay` - Audio playback
- `xdotool` - Simulate keyboard input
- `sox` - Audio analysis (audio level detection)
- `pactl` - PulseAudio control (microphone mute/unmute)
- `notify-send` - Desktop notifications
- `journalctl` - Service logging access
- `gnome-terminal` - Terminal for API key prompts and log viewing

## Configuration

**Environment:**
- `OPENAI_API_KEY` - Optional env var for OpenAI API key (checked first)
- Falls back to file-based storage at `~/.config/openai/api_key` or `~/.openai/api_key`

**Build/Service:**
- `config.json` - Application configuration at `~/.local/share/push-to-talk/config.json`
  - TTS backend selection (piper or openai)
  - AI mode selection (claude or realtime)
  - Hotkey configuration
  - Voice selection for OpenAI TTS
- `systemd` user service - `~/.config/systemd/user/push-to-talk.service`

**Model Files:**
- Whisper model cache - Auto-downloaded to `~/.cache/whisper/` (varies by model size)
- Piper TTS model - `~/.local/share/push-to-talk/piper-voices/en_US-lessac-medium.onnx`
- Python virtual environment - `~/.local/share/push-to-talk/venv/`

## Platform Requirements

**Development/Runtime:**
- Linux (PipeWire/PulseAudio audio stack required)
- GTK 3 (libgtk-3-0)
- Python 3.10+
- ffmpeg with WAV support
- X11 display server with xdotool support

**Production:**
- Same as above - runs as systemd user service on Linux desktop systems

## Audio Processing Pipeline

**Recording:**
- `pw-record` captures 44.1kHz 2-channel 16-bit signed PCM

**Transcription:**
- FFmpeg converts to 16kHz mono (required by Whisper)
- Whisper model (default: "small") transcribes English audio locally

**Text-to-Speech:**
- Piper: Local synthesis (22050Hz output)
- OpenAI TTS: Cloud API (24000Hz PCM)
- `aplay` handles playback

## Network & API Integration

**OpenAI Services:**
- TTS API endpoint: `https://api.openai.com/v1/audio/speech` (requires `OPENAI_API_KEY`)
- Realtime API endpoint: `wss://api.openai.com/v1/realtime` (WebSocket, requires auth)
- Model: `gpt-4o-realtime-preview-2024-12-17` for realtime conversations

**Local Claude:**
- Uses installed Claude CLI at `~/.local/bin/claude`
- Runs as subprocess with stdin/stdout communication

## Key Configuration Files

**Application:**
- `~/.local/share/push-to-talk/push-to-talk.py` - Main service entry point
- `~/.local/share/push-to-talk/indicator.py` - GTK status indicator
- `~/.local/share/push-to-talk/openai_realtime.py` - Realtime API integration
- `~/.local/share/push-to-talk/vocabulary.txt` - Custom words for Whisper

**Hotkeys:**
- Configurable via settings UI with validation (PTT and AI keys must differ)
- Stored in `config.json`

---

*Stack analysis: 2026-02-03*
