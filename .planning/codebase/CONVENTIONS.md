# Coding Conventions

**Analysis Date:** 2026-02-13

## Naming Patterns

**Files:**
- Lowercase with hyphens: `push-to-talk.py`, `openai_realtime.py`, `indicator.py`
- Executable scripts have `#!/usr/bin/env python3` shebang

**Functions:**
- snake_case for all functions: `get_openai_api_key()`, `load_config()`, `check_verbal_hooks()`
- Private/internal functions prefixed with underscore: `_detect_display()`, `_wait_for_display()`, `_interview_generate_question()`
- Descriptive names indicating operation: `get_`, `set_`, `load_`, `save_`, `check_`, `start_`, `stop_`

**Variables:**
- snake_case for all variables: `api_key`, `audio_dir`, `current_mode`, `interview_topic`
- Constants in UPPERCASE: `STATUS_FILE`, `SAMPLE_RATE`, `CHUNK_SIZE`, `COLORS`, `REALTIME_URL`
- Dictionary/module-level constants: `COLORS = {...}`, `STATUS_TEXT = {...}`, `TOOLS = [...]`
- Boolean flags use clear names: `OPENAI_AVAILABLE`, `REALTIME_AVAILABLE`, `save_audio`, `debug_mode`

**Classes:**
- PascalCase: `VocabularyManager`, `InterviewSession`, `ConversationSession`, `PushToTalk`, `StatusIndicator`, `SettingsWindow`
- Descriptive, noun-based names indicating purpose or responsibility

**Type Hints:**
- Not used. Code is un-typed Python 3.

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc`, `black` config, or `autopep8`)
- Spaces for indentation (4 spaces standard Python)
- Line length appears reasonable (~100-120 chars typical)
- String quotes: Mix of single and double quotes (no strict convention)

**Linting:**
- No linting configuration found (no `.eslintrc`, `pylintrc`, or `flake8` config)
- Code is not linted during development

**Docstrings:**
- Used for module documentation: `"""Push-to-Talk Dictation Service\n\nHold Right Ctrl to record..."""`
- Used for function documentation: `"""Detect the active X display if DISPLAY is not set..."""`
- Style: Triple-quoted docstrings at function/class start
- Format: Single-line or multi-line descriptive text (no type annotations in docstrings)

## Import Organization

**Order:**
1. Standard library: `os`, `sys`, `re`, `time`, `math`, `json`, `wave`, `shutil`, `tempfile`, `subprocess`, `threading`, `signal`, `atexit`
2. Collections: `from collections import deque`
3. Pathlib: `from pathlib import Path`
4. Third-party: `import whisper`, `from pynput import keyboard`, `from openai import OpenAI`
5. GTK/GUI libraries: `import gi`, `gi.require_version(...)`, `from gi.repository import ...`
6. Local modules: `from openai_realtime import RealtimeSession, get_api_key as get_realtime_api_key`

**Path Handling:**
- Consistent use of `pathlib.Path` instead of string paths: `Path(__file__).parent`, `Path.home()`, `path.exists()`, `path.read_text()`
- Expanduser for config: `os.path.expanduser(config.get('audio_dir', '~/Audio/push-to-talk'))`

## Error Handling

**Patterns:**
- Broad `except` clauses for non-critical operations: `except: pass` (silent failure acceptable for UI updates, status checks)
- Specific exception handling for subprocess operations: `except (subprocess.TimeoutExpired, FileNotFoundError):`
- Try-catch for optional imports: `try: from openai import OpenAI; OPENAI_AVAILABLE = True; except ImportError: OPENAI_AVAILABLE = False`
- Print to stderr for fatal errors: `print("ERROR: Could not connect to X display", file=sys.stderr)`
- Status file updates indicate error state: `set_status("error")` for user-visible failures

**No Exceptions Raised:**
- Code gracefully degrades rather than raising. Example: missing config file returns defaults, missing API key triggers prompt dialog
- Subprocess timeout handled: `subprocess.run(..., timeout=30)` to prevent hanging

## Logging

**Framework:** `print()` function with `flush=True` for immediate output

**Patterns:**
- Debug logging via `print()`: `print("Realtime API: Connected with tools", flush=True)`
- Error logging: `print(f"Failed to open API key prompt: {e}", flush=True)`
- Status messages: `print("[Listening...]", flush=True)`
- Verbose event logging available but controlled by debug mode flag
- systemd journal integration: Services log to journalctl, viewed via `journalctl --user -u push-to-talk`

**When to Log:**
- State transitions: `print("Realtime API: Mic muted", flush=True)`
- Tool execution: `print(f"Executing tool: {name}({arguments})", flush=True)`
- Events from long-running processes: `print(f"Realtime API: Sent {chunks_sent} audio chunks", flush=True)`
- Errors and exceptions: `print(f"Realtime API Error: {error}", flush=True)`

## Comments

**When to Comment:**
- Explaining non-obvious setup code: `# --- Auto-detect DISPLAY for systemd service startup ---`
- Inline clarifications for complex logic: `# Check threshold before starting drag`, `# Wait for motion threshold`
- Disabling code temporarily: Comments preserve intent

**Inline Comments:**
- Short clarifications on same line or line above: `return  # Current DISPLAY works fine`
- Limited use; code is generally self-documenting through clear naming

## Function Design

**Size:** Functions range from 10-50 lines typically. Larger methods (100+ lines) exist in classes managing state (`handle_events()`, `create_general_tab()`).

**Parameters:**
- Positional parameters for required values: `def load_config()`, `def save_config(config)`
- Keyword arguments for optional settings: `def speak_openai(text, voice="nova", save_path=None)`
- Default values used extensively: `def _wait_for_display(max_wait=30):`
- Callback functions as parameters: `def __init__(self, on_status=None):` then `self.on_status = on_status or (lambda s: None)`

**Return Values:**
- Boolean returns for success/failure: functions return `True`/`False` or status values
- Dictionary returns for structured data: `load_config()` returns config dict
- None for side-effect only operations: file writes, status updates
- Tuple returns for multiple values: Rare; structured dicts preferred

## Module Design

**Exports:**
- No explicit `__all__` defined
- Public functions at module level: `load_config()`, `save_config()`, `get_openai_api_key()`
- Classes defined at module level used directly: `from openai_realtime import RealtimeSession`
- Internal functions prefixed with underscore not exported

**Barrel Files:**
- Not used. Each file is self-contained:
  - `push-to-talk.py`: Main service with core `PushToTalk` class and helper functions
  - `indicator.py`: UI indicator and settings window
  - `openai_realtime.py`: OpenAI Realtime integration

**Circular Dependencies:**
- Avoided through separation: indicator communicates via status files, not direct imports

---

*Convention analysis: 2026-02-13*
