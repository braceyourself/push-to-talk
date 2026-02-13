# External Integrations

**Analysis Date:** 2026-02-13

## APIs & External Services

**OpenAI (Primary AI Provider):**
- Realtime API (`wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17`)
  - SDK/Client: `openai` Python package
  - Auth: Environment var `OPENAI_API_KEY` OR file `~/.config/openai/api_key`
  - Usage: Voice-to-voice conversations with function calling (`openai_realtime.py` lines 223-460)
  - Features: Server-side voice activity detection (VAD), audio transcription, tool execution

- TTS (Text-to-Speech)
  - Model: `tts-1`
  - Voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
  - Usage: Convert response text to speech (`push-to-talk.py` lines 260-306)
  - Audio format: PCM 24kHz mono, streamed to `aplay`

- Chat Completions (Smart Transcription)
  - Model: `gpt-4o-mini`
  - Usage: Fix transcription errors from Whisper (`push-to-talk.py` lines 377-407)
  - Flow: Whisper output → OpenAI correction → final text

**Claude AI (via Claude CLI):**
- Claude Code CLI at `~/.local/bin/claude`
- Mode: Headless subprocess invocation
- Invocations:
  - Interview mode: Generate interviewer questions (`push-to-talk.py` lines 1543-1600)
  - Conversation mode: Full tool access to project directory (`push-to-talk.py` lines 1861-1918)
- Permission modes:
  - Interview: `--permission-mode acceptEdits` (read-only access)
  - Conversation: `--permission-mode bypassPermissions` (full access for hands-free operation)
- Session directories:
  - Interview: `.claude-session/` per session
  - Conversation: Runs in project directory with session state

## Speech Recognition

**OpenAI Whisper:**
- Model: `small` (default, configurable: tiny, base, small, medium, large)
- Scope: Local processing (no API calls)
- Input: WAV files from recording
- Output: Text transcription
- Usage: `push-to-talk.py` lines 922-1000 (transcribe_and_type method)

## Audio System

**PipeWire (Recording):**
- Command: `pw-record --format s16 --rate 24000 --channels 1 -` (for Realtime API)
- Command: `pw-record --format s16 --rate 44100 --channels 2 [file]` (for conversation auto-listen)

**PulseAudio (Microphone Control):**
- Mute: `pactl set-source-mute @DEFAULT_SOURCE@ 1`
- Unmute: `pactl set-source-mute @DEFAULT_SOURCE@ 0`
- Usage: Echo prevention during Realtime API conversations (`openai_realtime.py` lines 296-297, 344-346)

**ALSA (Playback):**
- Command: `aplay -r 24000 -f S16_LE -t raw -q`
- Usage: Stream TTS output and Realtime API audio to speakers

**SoX (Audio Analysis):**
- Command: `sox [audio_file] -n stat`
- Usage: Get maximum amplitude for silence detection (`push-to-talk.py` lines 428-446)

## System Integrations

**X11 & xdotool:**
- xdotool for typing transcribed text into focused input
- X11 DISPLAY detection and connection (`push-to-talk.py` lines 26-85)
- XAUTHORITY auto-detection from GDM or /run/user

**Keyboard Input (pynput):**
- Listener for modifier keys: Ctrl, Shift, Alt
- Listener for special keys: Escape, Space, Pause, Scroll Lock
- Custom hotkey mapping for PTT and AI mode (`push-to-talk.py` lines 177-191)

**Desktop Notifications:**
- Command: `notify-send`
- Usage: Session start/end, verbal hook triggers, status updates

**systemd:**
- User service: `~/.config/systemd/user/push-to-talk.service`
- Auto-start at login
- Service logging via `journalctl --user -u push-to-talk`

**GTK 3 & Wayland (UI):**
- `gi.repository.Gtk` - Settings window, dialogs, configuration UI
- `gi.repository.Gdk` - Window events, positioning
- `cairo` - Custom rendering for status indicator dot
- `AppIndicator3` (optional) - System tray integration (`indicator.py` lines 19-25)
- Zenity for dialogs:
  - `zenity --entry` - Topic/question input
  - `zenity --file-selection --directory` - Directory picker for conversation mode

## Data Storage

**Local Files Only:**
- Config: `config.json` (user-editable settings)
- Vocabulary: `vocabulary.txt` (custom words for Whisper)
- Session recordings: `~/Audio/push-to-talk/sessions/{timestamp}/` (optional, `save_audio` setting)
  - Numbered WAV files (user and interviewer sides)
  - `transcript.md` (interview transcript)
  - `show-notes.md` (AI-generated summary)
  - `metadata.json` (session metadata)

**Caching:**
- No persistent caching layer detected
- Whisper model cached by openai-whisper (default: `~/.cache/huggingface/hub/`)

## Authentication & Identity

**Auth Provider:**
- OpenAI API Key (user-provided, no OAuth flow)
  - Stored locally: `~/.config/openai/api_key` (600 permissions, user-read-only)
  - Loaded at runtime from env var or file
  - Used for all OpenAI API calls (TTS, Realtime, smart transcription)
  - Fallback: Prompt user to enter key if not found (`push-to-talk.py` lines 119-175)

- Claude CLI auth (delegated)
  - No separate auth in push-to-talk
  - Claude CLI handles authentication at `~/.local/bin/claude`

## Monitoring & Observability

**Error Tracking:**
- Not detected (no Sentry, DataDog, etc.)

**Logs:**
- Console output: `print(..., flush=True)` throughout
- systemd journal: `journalctl --user -u push-to-talk -f`
- Log UI in indicator: Shows last 5 lines from journal (`indicator.py` line 109)
- Debug mode available in config: `debug_mode` setting

**Status Tracking:**
- Status file: `status` (plain text, updated by `set_status()`)
- Status values: `idle`, `recording`, `processing`, `success`, `error`, `listening`, `speaking`
- Consumed by: `indicator.py` for UI color updates

## CI/CD & Deployment

**Hosting:**
- Not applicable (desktop application)

**CI Pipeline:**
- Not detected (no GitHub Actions, etc.)

**Deployment:**
- Manual installation: `install.sh` script
- Installed to: `~/.local/share/push-to-talk/` (service files, vocab, session data)
- Service auto-start via systemd user service

## Environment Configuration

**Required env vars:**
- `OPENAI_API_KEY` - OpenAI API key (or read from `~/.config/openai/api_key`)
- `DISPLAY` - X11 display (auto-detected if not set, `push-to-talk.py` lines 26-85)
- `XAUTHORITY` - X11 authority file (auto-detected if not set)

**Optional env vars:**
- `CLAUDE_CLI` - Path to Claude CLI (default: `~/.local/bin/claude`)

**Secrets location:**
- OpenAI API key: `~/.config/openai/api_key` (permissions 600, plaintext)
- No other secrets stored locally

## Webhooks & Callbacks

**Incoming:**
- Not applicable

**Outgoing:**
- No external webhooks
- Audio saved to disk when `save_audio: true` (local filesystem only)

## Verbal Hooks (Custom Commands)

**Framework:**
- User-defined voice commands in config
- Commands matched via text patterns (exact or wildcard with `*`)
- Execution: `subprocess.Popen(command, shell=True)`

**Example Hook:**
```json
{
  "trigger": "search for *",
  "command": "xdg-open 'https://google.com/search?q={}'"
}
```

**Implementation:** `push-to-talk.py` lines 329-374

---

*Integration audit: 2026-02-13*
