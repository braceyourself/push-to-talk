# Coding Conventions

**Analysis Date:** 2026-02-03

## Naming Patterns

**Files:**
- Lowercase with hyphens for executables: `push-to-talk.py`, `openai_realtime.py`, `indicator.py`
- Underscore separator for modules: `openai_realtime.py`

**Functions:**
- Snake case: `get_openai_api_key()`, `start_recording()`, `transcribe_and_type()`
- Descriptive action verbs for operations: `load_config()`, `save_config()`, `execute_tool()`
- Getter/setter pattern: `get_*()`, `set_*()` for configuration and status

**Classes:**
- PascalCase: `PushToTalk`, `VocabularyManager`, `RealtimeSession`, `SettingsWindow`, `StatusIndicator`
- Descriptive purpose-based names

**Variables:**
- Snake case throughout: `record_process`, `temp_file`, `api_key`, `audio_level`
- Private attributes prefixed with underscore: `_interrupt_requested`, `_set_status()`
- Boolean flags clearly named: `recording`, `dragging`, `playing_audio`
- Configuration dictionaries: `config`, `COLORS`, `STATUS_TEXT`, `TOOLS`

**Constants:**
- UPPER_CASE: `WHISPER_MODEL`, `BASE_DIR`, `HISTORY_SIZE`, `SILENCE_THRESHOLD`
- Configuration dictionaries as constants: `MODIFIER_KEY_OPTIONS`, `INTERRUPT_KEY_OPTIONS`, `CORRECTION_PATTERNS`

## Code Style

**Formatting:**
- No explicit formatter configured (no eslintrc, prettier, black config found)
- Consistent 4-space indentation observed throughout
- Module docstrings at file top describing purpose: `"""Push-to-Talk Dictation Service..."""`
- Type hints not used (pure Python 3 without annotations)

**Linting:**
- No lint configuration detected
- Code follows PEP 8 style informally
- Long lines acceptable (some exceed 100 chars, e.g., subprocess commands with complex arguments)

## Import Organization

**Order:** Standard library → Third-party → Local
1. Standard library (os, sys, json, asyncio, etc.)
2. Third-party packages (whisper, pynput, openai, gtk, cairo, websockets)
3. Path-based utilities from pathlib

**Examples from codebase:**
```python
# push-to-talk.py
import os
import sys
import re
import time
import math
import json
import shutil
import tempfile
import subprocess
import threading
import signal
import atexit
from collections import deque
from pathlib import Path
import whisper
from pynput import keyboard
```

**Path Aliases:**
- Uses `Path` from pathlib extensively for cross-platform file operations
- No import aliases (`as`) used except as required by libraries
- No relative imports; all imports are absolute

## Error Handling

**Patterns:**
- Try/except blocks used extensively for robustness
- Graceful degradation: missing optional dependencies handled with boolean flags
  - `OPENAI_AVAILABLE`, `REALTIME_AVAILABLE`, `APPINDICATOR_AVAILABLE`
- Bare `except:` used for non-critical operations (file operations, status updates)
- Specific exception handling for critical operations:
  - `subprocess.TimeoutExpired` - for long-running commands
  - `websockets.exceptions.ConnectionClosed` - for network operations
  - `ImportError` - for optional dependencies
- Error context preserved: exceptions logged with descriptive print statements
- GUI errors shown to user via notifications: `subprocess.Popen(['notify-send', ...])`

**Example pattern from `push-to-talk.py` (lines 164-171):**
```python
try:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            return {**default, **config}
except Exception as e:
    print(f"Error loading config: {e}", flush=True)
return default
```

## Logging

**Framework:** Console logging with `print()` statements
- No logging module used; direct to stdout
- All print statements use `flush=True` for systemd journal compatibility
- Status prefixed in realtime module: `"Realtime API: ..."` for debugging

**Patterns:**
- Informational: `print(f"Loaded {len(self.words)} vocabulary words", flush=True)`
- Progress: `print("Recording...", flush=True)`
- Errors: `print(f"Error during transcription: {e}", flush=True)`
- Status changes: `print(f"Switched to {mode}", flush=True)`
- Debug in async code: `print(f"Realtime API event: {event_type}", flush=True)`

**When to Log:**
- Service startup/shutdown
- Mode changes (TTS backend, AI mode, indicator style)
- Operation start/end (recording, processing, speaking)
- Configuration changes
- Errors and warnings
- System integration points (tool execution, file access)

## Comments

**When to Comment:**
- Complex algorithms: regex patterns (CORRECTION_PATTERNS), audio level detection logic
- Business logic branches: when recording should be blocked, when to auto-listen
- State machine transitions: recording → processing → success/error
- Non-obvious workarounds: spurious release event handling (line 932), cooldown timing for echo avoidance
- Code intent for critical sections: Realtime API tool calling flow

**JSDoc/TSDoc:**
- Not used; pure docstrings with triple quotes
- Module-level docstrings only (file purpose)
- Function docstrings are brief one-liners:
  ```python
  def load_config():
      """Load configuration from file."""
  ```
- Class docstrings describe purpose:
  ```python
  class VocabularyManager:
      """Manages custom vocabulary for Whisper."""
  ```

## Function Design

**Size:** Functions are moderate length
- Recording/transcription: 50-150 lines (handles multiple steps with clear sequencing)
- Event handlers: 10-40 lines (on_press, on_release, callbacks)
- Configuration: 5-20 lines (load/save/toggle operations)
- Window creation: 100-300 lines (UI building with many widget setup statements)

**Parameters:**
- Positional for required, keyword defaults for optional
- Config dictionaries merged with spread operator: `{**default, **config}`
- File paths passed as Path objects or strings (both acceptable)
- Callbacks passed as lambda or function reference: `on_press=self.on_press`

**Return Values:**
- Simple returns: status bools, values, None
- Complex returns use dictionaries from json parsing
- Thread-safe operations return immediately; actual work in daemon threads

**Example (lines 548-623):**
```python
def transcribe_and_type(self, temp_file):
    if not temp_file or not os.path.exists(temp_file.name):
        set_status('idle')
        return

    try:
        # Validation
        file_size = os.path.getsize(temp_file.name)
        if file_size < 5000:
            set_status('idle')
            print("Recording too short, skipping.", flush=True)
            return
        # Processing
        # ...
    except Exception as e:
        print(f"Error during transcription: {e}", flush=True)
        set_status('error')
    finally:
        # Cleanup
        for f in [...]:
            try:
                os.unlink(f)
            except:
                pass
```

## Module Design

**Exports:**
- No explicit `__all__` declarations
- Functions and classes available at module scope
- Private helpers prefixed with underscore: `_set_status()`, `_interrupt_requested`

**Barrel Files:**
- Not used in this codebase (no index.py or __init__.py files)
- Single-responsibility modules: one main class per file typically

**File-level initialization:**
- Constants defined at module top (COLORS, TOOLS, MODIFIER_KEY_OPTIONS)
- Global variables for processes: `indicator_process`, `APPINDICATOR_AVAILABLE`
- Try/except for optional feature detection at import time

## String Formatting

**Style:** F-strings used consistently
- `f"Loading {len(self.words)} vocabulary words"`
- `f"Error loading config: {e}"`
- Formatting within JSON dumps: `json.dumps(config, indent=2)`
- HTML/markup in GTK: `set_markup("<span foreground='#4ade80'>Valid key found</span>")`

## Threading and Async

**Threading:**
- Background operations spawn daemon threads: `threading.Thread(target=..., daemon=True).start()`
- Locks for shared resources: `self.model_lock` protects Whisper model access
- Thread-safe request pattern for async signaling: `_interrupt_requested` flag

**Async:**
- OpenAI Realtime uses `asyncio` for WebSocket handling
- `async def` for I/O operations (connect, send_audio, handle_events)
- `await` for all async operations
- Background tasks: `asyncio.create_task(delayed_unmute())`
