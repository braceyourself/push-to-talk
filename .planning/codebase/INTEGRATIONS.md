# External Integrations

**Analysis Date:** 2026-02-03

## APIs & External Services

**OpenAI Services:**
- OpenAI Text-to-Speech (TTS) API
  - What it's used for: Cloud-based voice synthesis alternative to local Piper
  - SDK/Client: `openai` package (OpenAI 1.0+ SDK)
  - Auth: `OPENAI_API_KEY` environment variable or `~/.config/openai/api_key`
  - Endpoint: `https://api.openai.com/v1/audio/speech`
  - Voice options: alloy, echo, fable, onyx, nova, shimmer
  - Implementation: `push-to-talk.py` lines 183-215 (`speak_openai()` function)
  - Conditional: Optional - only available if API key is configured

- OpenAI Realtime API (GPT-4o)
  - What it's used for: Low-latency voice-to-voice conversations with AI
  - SDK/Client: `openai` package + `websockets` library
  - Auth: `OPENAI_API_KEY` (required for websocket connection)
  - Endpoint: `wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17`
  - Implementation: `openai_realtime.py` (entire module)
  - Feature: Function calling for system interaction (run_command, read_file, write_file, ask_claude, remember, recall)
  - Echo suppression: Active mic muting via PulseAudio during AI response
  - Conditional: Optional - requires websockets package and API key

**Anthropic Claude:**
- Claude CLI
  - What it's used for: AI reasoning and responses in Claude mode (non-realtime)
  - Implementation: Subprocess calls in `push-to-talk.py` lines 757-787, 810-835
  - Location: `~/.local/bin/claude`
  - Session persistence: `~/.local/share/push-to-talk/claude-session/`
  - Launch pattern: `claude -c -p <question> --permission-mode acceptEdits --add-dir ~/.claude`
  - Auto-listen integration: Follow-up questions trigger auto-listen for up to 4 seconds

## Data Storage

**Databases:**
- None - application uses only local files

**File Storage:**
- Local filesystem only (no cloud storage)
  - Configuration: `~/.local/share/push-to-talk/config.json`
  - Vocabulary: `~/.local/share/push-to-talk/vocabulary.txt`
  - Status: `~/.local/share/push-to-talk/status` (transient, for IPC between main service and indicator)
  - Claude session memory: `~/.local/share/push-to-talk/claude-session/memory.json`
  - API keys: `~/.config/openai/api_key` (mode 0600, read-only by user)

**Caching:**
- Whisper model cache: `~/.cache/whisper/` (auto-managed by openai-whisper)
- Piper TTS model: `~/.local/share/push-to-talk/piper-voices/`

## Authentication & Identity

**Auth Provider:**
- Custom file-based (OpenAI API key)
- Implementation: `push-to-talk.py` lines 41-52 (`get_openai_api_key()`)
- Fallback chain:
  1. `OPENAI_API_KEY` environment variable
  2. `~/.config/openai/api_key` file
  3. `~/.openai/api_key` file
- Interactive setup: Terminal prompt if no key found (`push-to-talk.py` lines 55-111)

**Secrets Management:**
- API keys stored with restricted permissions (0600)
- Environment variable support for sensitive values
- No secrets stored in config.json (only settings)

## Monitoring & Observability

**Error Tracking:**
- None (no third-party error tracking)
- Local logging via systemd journalctl

**Logs:**
- systemd user journal
- Access: `journalctl --user -u push-to-talk`
- Last 5 lines pulled by indicator UI for display
- Status indicators: idle, recording, processing, success, error, listening (realtime), speaking (realtime)

**Status Communication:**
- IPC via status file: `~/.local/share/push-to-talk/status`
- Read by indicator process to update UI colors
- Transient - written during operations, reset to 'idle' on completion

## CI/CD & Deployment

**Hosting:**
- None (desktop application)
- Runs on user's Linux system as systemd user service

**CI Pipeline:**
- None detected

**Auto-update:**
- Git-based installation/update flow (from git clone or git pull)
- Manual service restart required: `systemctl --user restart push-to-talk`

## Environment Configuration

**Required env vars (if not using defaults):**
- `OPENAI_API_KEY` - For OpenAI Realtime API and TTS

**Optional env vars:**
- `DISPLAY` - X11 display (for desktop integration)

**Settings in config.json:**
- `tts_backend` - "piper" or "openai"
- `openai_voice` - Voice choice when using OpenAI TTS
- `ai_mode` - "claude" or "realtime"
- `ptt_key` - Hotkey for push-to-talk (default: "ctrl_r")
- `ai_key` - Hotkey modifier for AI mode (default: "shift_r")
- `interrupt_key` - Key to interrupt AI responses (default: "escape")
- `indicator_style` - "floating" or "tray"
- `debug_mode` - Boolean for debug logging

**Secrets location:**
- `~/.config/openai/api_key` (created on first setup)
- Readable only by user

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None (unidirectional API calls only)

## Integration Patterns

**Realtime API Tool Calling:**
The Realtime integration includes function calling for system interaction. Tools available:
- `run_command` - Execute shell commands
- `read_file` - Read file contents
- `write_file` - Write to files
- `ask_claude` - Delegate tasks to Claude CLI
- `remember` - Save to memory.json
- `recall` - Retrieve saved memories

Implementation: `openai_realtime.py` lines 34-133 (tool definitions), lines 155-220 (execution)

**Cross-Service Communication:**
- Main service (`push-to-talk.py`) ↔ Indicator (`indicator.py`): Via status file
- Main service → Claude CLI: Subprocess with working directory for session persistence
- Audio flow: PipeWire recording → FFmpeg conversion → Whisper transcription → OpenAI/Claude processing → Piper/OpenAI TTS → aplay playback

## Conditional Features

**OpenAI Realtime:**
- Requires: `websockets` package + `OPENAI_API_KEY`
- Fallback behavior: Shows error if unavailable, suggests API key setup
- Detection: `openai_realtime.py` lines 525-527 (`is_available()`)

**OpenAI TTS:**
- Requires: `OPENAI_API_KEY`
- Fallback: Defaults to Piper if unavailable
- Toggle: Can switch backends via settings UI

**System Tray Indicator:**
- Requires: `gir1.2-appindicator3-0.1` (AppIndicator3 library)
- Fallback: Falls back to floating dot indicator
- Detection: `indicator.py` lines 20-25

## Security Considerations

**API Key Protection:**
- File-based keys stored with 0600 permissions
- Env var support for sensitive environments
- Settings UI hides key by default, shows "Show/Hide" toggle
- Key validation: Checks for "sk-" prefix in settings

**Function Calling Permissions:**
- Realtime API has unrestricted command execution capability
- No sandboxing - can run any shell command
- Relies on OpenAI's model safety, not system-level restrictions
- `subprocess.run()` used with shell=True in tool execution

**Audio Security:**
- Microphone muted when AI is speaking (prevents echo/feedback)
- Auto-unmute on disconnect
- No audio streams to third parties except to configured OpenAI/Claude services

---

*Integration audit: 2026-02-03*
