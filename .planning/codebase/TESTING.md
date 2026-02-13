# Testing Patterns

**Analysis Date:** 2026-02-13

## Test Framework

**Status:** No automated testing framework detected

**Runner:**
- Not configured. No pytest, unittest, or nose configuration found.
- No test files (`test_*.py`, `*_test.py`, or `tests/` directory) present in codebase.

**Assertion Library:**
- Not applicable. No test code exists.

**Run Commands:**
- No test commands available.

## Test Strategy (Current State)

**Manual Testing Only:**
- Functionality tested through direct interaction with running service
- systemd service integration tested via `systemctl --user restart push-to-talk.service`
- Settings verified through indicator GUI (`indicator.py`)
- Audio/transcription tested live with actual recordings

**Integration Testing (Manual):**
- Full workflow: Hold PTT key → record audio → transcribe → type output
- Verify config persistence: Change settings in GUI → restart service → verify settings applied
- Test hotkey binding changes: Modify keybindings → restart → verify new keys work
- Audio save feature: Enable save audio → record → verify files in audio directory

**E2E Testing (Manual):**
- Interview mode: Start AI hotkey → receive question → press PTT → speak answer → verify response
- Conversation mode: Start AI hotkey → ask question → Claude processes with tools → verify output
- Realtime mode: Connect to OpenAI Realtime → speak → receive voice response

## How Testing Is Currently Handled

**Service Health:**
- Check via `systemctl --user status push-to-talk`
- View logs: `journalctl --user -u push-to-talk -f --no-pager`
- Status file monitoring: `cat status` shows current service state

**UI Testing:**
- Manual interaction with `SettingsWindow` (GTK application in `indicator.py`)
- Verify dropdown selections persist to `config.json`
- Click buttons and verify dialogs appear/dismiss correctly

**Configuration Testing:**
- Load/save cycle: `load_config()` and `save_config()` work with `config.json` at project root
- Default values apply when config missing: See `load_config()` in `indicator.py` lines 52-80
- API key validation: Manual entry in settings, verify persistence to `~/.config/openai/api_key`

## Code Patterns That Enable Testing (if tests existed)

**Dependency Injection:**
- Classes accept callbacks for status updates: `def __init__(self, on_status=None):` in `RealtimeSession` (line 226)
- Allows mocking status handler: `session = RealtimeSession(api_key, on_status=mock_handler)`

**File Path Abstraction:**
- All paths use `pathlib.Path` for testability: `CONFIG_FILE = Path(__file__).parent / "config.json"`
- Enables test fixtures to override paths via monkey-patching

**Separable Components:**
- `VocabularyManager` (line 478): Isolated vocabulary operations
- `InterviewSession` (line 535): Encapsulates interview state and flow
- `ConversationSession` (line 628): Encapsulates conversation state
- `RealtimeSession` (line 223 in `openai_realtime.py`): WebSocket session handling

**Configuration as Dict:**
- `load_config()` returns plain dict, enabling easy fixture creation for tests
- No singleton pattern; config can be passed to functions as needed

## Testing Gaps

**Missing Unit Tests:**
- Config loading/saving (`load_config()`, `save_config()`)
- Hotkey validation logic (`on_hotkey_changed()` in `indicator.py`)
- API key handling (`get_openai_api_key()`, `save_openai_api_key()`)

**Missing Integration Tests:**
- Full audio pipeline: Record → transcribe → type
- Config persistence through service restart
- Multiple hotkey configurations
- Vocabulary learning from corrections

**Missing E2E Tests:**
- Interview mode workflow end-to-end
- Conversation mode with Claude tool execution
- Realtime mode audio quality and responsiveness
- Service autostart and cleanup on exit

## If Tests Were to Be Added: Recommended Structure

**Test Framework:** pytest (recommended over unittest for simplicity)

**Test File Organization:** Parallel structure to source
```
tests/
├── test_push_to_talk.py      # Main service logic
├── test_indicator.py          # UI and settings
├── test_openai_realtime.py   # Realtime integration
├── test_vocabulary.py         # VocabularyManager
├── test_config.py             # Config loading/saving
└── fixtures/
    ├── config_fixtures.py     # Test config dicts
    └── audio_fixtures.py      # Test audio data
```

**Unit Test Pattern (if implemented):**
```python
# Example: test config loading
import pytest
from pathlib import Path
from indicator import load_config

def test_load_config_returns_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("indicator.CONFIG_FILE", tmp_path / "nonexistent.json")
    config = load_config()
    assert config["tts_backend"] == "piper"
    assert config["debug_mode"] == False

def test_load_config_merges_with_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"debug_mode": true}')
    monkeypatch.setattr("indicator.CONFIG_FILE", config_file)
    config = load_config()
    assert config["debug_mode"] == True
    assert config["tts_backend"] == "piper"  # default
```

**Mocking Pattern (if implemented):**
```python
# Example: mock subprocess for safe testing
from unittest.mock import patch, MagicMock

@patch('subprocess.run')
def test_api_key_save_sets_permissions(mock_run):
    save_openai_api_key("sk-test-key")
    # Verify chmod 0o600 was called
    assert mock_run.called  # Would verify actual chmod in real test
```

## Manual Testing Checklist (Current Practice)

**Core Functionality:**
- [ ] PTT hotkey records and transcribes
- [ ] Transcribed text types into focused application
- [ ] Vocabulary file changes improve transcription
- [ ] Smart transcription mode works (if enabled)

**Configuration:**
- [ ] Settings persist after service restart
- [ ] Hotkey changes take effect on restart
- [ ] Audio save location is created and files saved
- [ ] Debug mode enables verbose logging

**Modes:**
- [ ] Claude mode: PTT → transcribe → type
- [ ] Conversation mode: AI hotkey → ask question → Claude responds
- [ ] Interview mode: AI hotkey → interviewer asks → PTT to answer
- [ ] Realtime mode (if enabled): Voice-to-voice with minimal latency

**Error Handling:**
- [ ] Missing DISPLAY handled gracefully at startup
- [ ] Missing API key prompts user
- [ ] Service restart without crashing
- [ ] Invalid hotkey combinations show error message

---

*Testing analysis: 2026-02-13*
