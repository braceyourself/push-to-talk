# Testing Patterns

**Analysis Date:** 2026-02-03

## Test Framework

**Status:** No automated testing framework detected

- No pytest, unittest, or vitest configuration
- No test files in codebase (`*_test.py`, `*_spec.py` not found)
- No test runner configuration files
- Testing appears to be manual/integration based

## Manual Testing Approach

**How this codebase is tested:**

The project relies on manual testing and systemd service integration testing:
- Systemd user service runs as background daemon
- Manual verification via `systemctl --user status push-to-talk`
- Logs inspected with `journalctl --user -u push-to-talk -f`
- GUI indicator provides real-time visual feedback (status dots)
- Status popup shows recent transcriptions and allows manual service restart

## Code Organization for Testability

**Separation of Concerns:**
- `push-to-talk.py` - Core dictation logic (recording, transcription, typing)
- `indicator.py` - Separate UI process (status display)
- `openai_realtime.py` - Isolated Realtime API implementation
- Configuration is externalized to JSON file (`config.json`)

**Dependency Injection:**
- Callback pattern for status updates: `on_status=set_status` parameter
- Configuration loaded from file, not hardcoded
- Optional dependencies feature-detected at import: `OPENAI_AVAILABLE`, `REALTIME_AVAILABLE`

**Error Handling for Testing:**
- Try/except blocks allow graceful degradation
- Status file as IPC mechanism between service and indicator
- Comprehensive logging to stdout (captured by journalctl)

## Test Data & Fixtures

**Configuration Test Data:**
```python
# From load_config() - lines 151-171
default = {
    "tts_backend": "piper",
    "openai_voice": "nova",
    "ai_mode": "claude",
    "ptt_key": "ctrl_r",
    "ai_key": "shift_r",
    "interrupt_key": "escape",
    "indicator_style": "floating",
    "indicator_x": None,
    "indicator_y": None,
}
```

**Test Vocabulary:**
```python
# From indicator.py - line 33
OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# From push-to-talk.py - lines 114-127
MODIFIER_KEY_OPTIONS = {
    "ctrl_r": ("Right Ctrl", keyboard.Key.ctrl_r),
    "ctrl_l": ("Left Ctrl", keyboard.Key.ctrl_l),
    # ... etc
}
```

**Status States for Testing:**
```python
# From indicator.py - lines 107-115
COLORS = {
    'idle': (0.5, 0.5, 0.5, 0.3),
    'recording': (1.0, 0.2, 0.2, 0.9),
    'processing': (1.0, 0.8, 0.0, 0.9),
    'success': (0.2, 0.9, 0.2, 0.9),
    'error': (1.0, 0.0, 0.0, 0.9),
    'listening': (0.2, 0.6, 1.0, 0.9),
    'speaking': (0.8, 0.4, 1.0, 0.9),
}
```

## Integration Points for Testing

**File-based IPC:**
- Status file at `BASE_DIR / "status"` - checked every 100ms by indicator
- Configuration file at `BASE_DIR / "config.json"` - persistent settings
- Vocabulary file at `BASE_DIR / "vocabulary.txt"` - learned words

**Process Management:**
- Indicator subprocess spawned/managed: `subprocess.Popen(['python3', str(INDICATOR_SCRIPT)])`
- PipeWire recording: `pw-record` subprocess
- FFmpeg conversion: `subprocess.run(['ffmpeg', ...])`
- Piper TTS: piped with shell command

**Mock-able Subprocess Calls:**
```python
# From push-to-talk.py - lines 536-542
self.record_process = subprocess.Popen([
    'pw-record',
    '--format', 's16',
    '--rate', '44100',
    '--channels', '2',
    self.temp_file.name
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

**External Tool Invocations (Testable):**
- OpenAI Whisper: `self.model.transcribe(converted_file, ...)`
- Claude CLI: `subprocess.run([str(CLAUDE_CLI), '-c', '-p', question, ...])`
- xdotool typing: `subprocess.run(['xdotool', 'type', '--delay', '12', '--', text])`
- notify-send: `subprocess.Popen(['notify-send', '-t', '3000', ...])`

## Manual Test Scenarios

**Basic Dictation Flow (Manual):**
1. Start service: `systemctl --user start push-to-talk.service`
2. Open text editor
3. Hold Right Ctrl key
4. Speak text ("hello world")
5. Release Right Ctrl
6. Verify: "hello world " appears in editor
7. Check logs: `journalctl --user -u push-to-talk -n 5`
   - Should see: "Recording...", "Transcribed: hello world", "Success"

**AI Assistant Mode (Manual):**
1. Hold Right Ctrl + Right Shift simultaneously
2. Ask a question ("what is the weather")
3. Release keys
4. Indicator turns yellow (processing)
5. Claude responds verbally
6. Indicator turns blue (auto-listening)
7. Can ask follow-up or press Escape to stop

**Configuration Changes (Manual):**
1. Click indicator dot → Settings
2. Change hotkey, TTS backend, AI mode
3. Service automatically detects config changes via file watch
4. Some changes require service restart (shown in UI)

**Error Conditions (Manual):**
1. Recording too short (< 5000 bytes) - status shows idle, no output
2. No speech detected (silence) - logged and ignored
3. OpenAI API unavailable - error notification shown, service continues
4. Missing Realtime dependencies - falls back to Claude mode

## Test Coverage Gaps

**Untested Areas:**
- Realtime WebSocket connection handling (`openai_realtime.py`)
  - Files: `openai_realtime.py:239-284` (connect), `318-438` (handle_events)
  - Risk: Network failures, API changes, protocol bugs
  - Current coverage: Manual integration only via systemd service

- GTK GUI event handling (`indicator.py`)
  - Files: `indicator.py:949-1011` (mouse events, drag handling)
  - Risk: Mouse tracking bugs, focus issues, coordinate miscalculations
  - Current coverage: Manual UI interaction only

- Concurrent Recording Edge Cases (`push-to-talk.py`)
  - Files: `push-to-talk.py:855-965` (on_press, on_release, key state machine)
  - Risk: Race conditions with spurious key events, overlapping recordings
  - Current coverage: Ad-hoc testing with various key timing scenarios

- Subprocess Error Handling
  - Files: All subprocess.Popen/run calls
  - Risk: Process failures, timeouts, I/O deadlocks
  - Current coverage: Timeout handling only for Claude CLI (120s)

**Why No Automated Testing:**
- Heavy I/O dependencies: audio recording, TTS, speech recognition
- System-level integration: keyboard listener, display server (X11/Wayland)
- External service dependencies: OpenAI Whisper, Claude CLI, OpenAI Realtime API
- GUI testing complexity: GTK windows, Cairo drawing, mouse events
- Difficult to mock: keyboard events (pynput), audio streams (pw-record)

## Debugging Practices

**Logging for Diagnosis:**
- All major operations have print statements with `flush=True`
- Realtime API prefixes messages: `"Realtime API: Event handler started"`
- Status changes logged: `"Switched to OpenAI TTS"`, `"Switched to Realtime AI mode"`

**View Logs in Real-time:**
```bash
journalctl --user -u push-to-talk -f --no-pager
```

**Manual Status Checks:**
```bash
# Check if service is running
systemctl --user status push-to-talk.service

# Check last 20 log lines
journalctl --user -u push-to-talk -n 20 --no-pager

# Check configuration
cat ~/.local/share/push-to-talk/config.json

# Check vocabulary learned
cat ~/.local/share/push-to-talk/vocabulary.txt
```

**Debug Mode:**
- Settings UI has "Enable debug mode" toggle (stored in config)
- Debug mode flag stored but not actively used in logging
- Could be expanded to increase verbosity in future

**Key Testing Artifacts:**
- Status file polling: `indicator.py:1040-1047` checks status every 100ms
- Config file persistence: Changes saved to `config.json` on every setting change
- Vocabulary persistence: Added words saved to `vocabulary.txt`
- Log file: All output captured by systemd journal

## Test Reliability Considerations

**Timing-Sensitive Code:**
- Auto-listen duration: `AUTO_LISTEN_SECONDS = 4` (lines 144)
- Success status timeout: `GLib.timeout_add(1500, self.return_to_idle)` (line 1058)
- Status indicator hover delay: `GLib.timeout_add(300, self.show_popup)` (line 955)
- Drag threshold: `self.drag_threshold = 5` pixels (line 867)

**Audio Processing Timing:**
- File size check: minimum 5000 bytes to process (lines 556, 730)
- Audio level check: SILENCE_THRESHOLD = -35 dB (line 145)
- Audio cooldown: 2 second wait after TTS finishes before accepting input (line 457)

**Flaky Test Areas (if automated tests were added):**
- Microphone input timing (device-dependent)
- Whisper transcription accuracy (model variance)
- Claude API response latency (network dependent)
- GUI rendering (display server dependent)
- Keyboard event ordering (OS-level timing)
