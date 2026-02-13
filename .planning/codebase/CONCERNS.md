# Codebase Concerns

**Analysis Date:** 2026-02-13

## Tech Debt

**Bare Exception Handlers Throughout Codebase:**
- Issue: Multiple bare `except:` clauses that silently catch and ignore all exceptions without logging or understanding failure modes
- Files: `push-to-talk.py` (lines 35, 59, 73, 425, 475, 591, 1110, 1275, 1311, 1342, 1362, 1444, 1744, 1808, 1956, 1983, 1999, 2047), `indicator.py` (lines 79, 89, 719, 1274, 1617), `openai_realtime.py` (lines 145, 353)
- Impact: Makes debugging difficult. Failures in file I/O, subprocess execution, and config operations are silently swallowed. Production issues become harder to diagnose.
- Fix approach: Replace with specific exception types and proper error logging. At minimum, add `print()` statements to log failures to journalctl.

**Shell=True in Subprocess Calls:**
- Issue: Two instances of `subprocess.Popen(..., shell=True)` in verbal_hooks execution
- Files: `push-to-talk.py` (lines 353, 365) in `check_verbal_hooks()`
- Impact: Potential security vulnerability. User voice input gets executed directly in shell without validation. If hooks contain untrusted input, could enable command injection.
- Fix approach: Use `subprocess.run(..., shell=False)` with proper argument list parsing. Validate hook triggers and commands before execution.

**Monolithic File Structure:**
- Issue: Main `push-to-talk.py` is 2237 lines with multiple responsibilities: hotkey handling, recording, transcription, TTS, interview mode, conversation mode, stream mode, realtime API integration
- Files: `push-to-talk.py`
- Impact: Difficult to test individual components, high cognitive load for maintenance, hard to reuse interview/conversation logic
- Fix approach: Extract into modules: `recorder.py`, `transcriber.py`, `ai_modes/interview.py`, `ai_modes/conversation.py`, `ai_modes/realtime.py`

**Indicator Process as Global State:**
- Issue: `indicator_process` is global variable modified in `start_indicator()` and `stop_indicator()`
- Files: `push-to-talk.py` (lines 417, 451, 464-470)
- Impact: Difficult to track state transitions, makes testing hard, potential race conditions
- Fix approach: Wrap indicator management in a class with proper lifecycle methods

**Hardcoded Paths and Constants:**
- Issue: Mix of absolute paths, relative paths, environment-based paths scattered throughout without validation
- Files: `push-to-talk.py` (lines 195-206, 554, 839-850), `indicator.py` (lines 27-30, 68)
- Impact: Breaks if directory structure changes, requires manual updates for different systems
- Fix approach: Centralize path configuration in a `paths.py` module with validation

## Known Bugs

**Stale Key Flags in Hotkey Handler:**
- Symptoms: Shift key stuck in pressed state after AI mode exits, affects PTT hotkey recognition
- Files: `push-to-talk.py` (lines 2141-2147)
- Trigger: Press PTT + AI key combo, release both, then press PTT alone. Sometimes shift_r_pressed flag remains True
- Workaround: Code attempts to clear with "Clearing stale shift_r_pressed flag" but the condition is reactive only
- Root cause: Key release events are unreliable; physical key release may not trigger on_release() callback

**Stream Mode Deduplication Fragile:**
- Symptoms: Stream mode may type duplicate words or skip words when chunks overlap
- Files: `push-to-talk.py` (lines 1235-1266)
- Cause: Overlap detection uses simple word-matching heuristic (checking if words match last 5 words). Doesn't account for contractions, punctuation variations, or word order changes in transcription
- Trigger: Fast speech, technical terms, or transcription errors
- Impact: User sees garbled output in stream mode

**Realtime API Echo Cancellation:**
- Symptoms: Occasional feedback loops where AI voice is re-recorded and fed back
- Files: `openai_realtime.py` (lines 340-410)
- Cause: Mic muting via `pactl` is timing-dependent. There's a 2-second cooldown window but speaker audio may linger longer
- Workaround: 1.5-second delayed unmute, but still vulnerable to slow speaker systems
- Impact: Realtime mode intermittently produces feedback noise

**Interview Post-Processing Assumption:**
- Symptoms: Post-processing silently fails if ffmpeg is not installed or audio files are missing
- Files: `push-to-talk.py` (lines 1661-1759)
- Cause: No upfront validation of ffmpeg availability, audio file existence before starting post-processing
- Impact: Interview completes successfully but stitched audio/MP3 never created with no error notification

## Security Considerations

**API Key in Memory:**
- Risk: OpenAI API keys loaded from file and stored in Python variables, then passed to subprocess
- Files: `push-to-talk.py` (lines 105-116, 268, 262), `openai_realtime.py` (lines 511-522)
- Current mitigation: Keys stored with 0o600 file permissions, not exposed in debug output
- Recommendations:
  - Use environment variables exclusively instead of reading from files
  - Mask key in any log output (show only last 4 chars)
  - Consider using credential manager (systemd user-provided secrets)

**Voice Command Execution (`verbal_hooks`):**
- Risk: User voice input can trigger arbitrary shell commands with `shell=True`
- Files: `push-to-talk.py` (lines 329-374)
- Current mitigation: Trigger matching is case-insensitive but otherwise unrestricted
- Recommendations:
  - Remove `shell=True` and use explicit argument lists
  - Whitelist allowed commands, not arbitrary user input
  - Add confirmation prompt before executing commands that modify system state
  - Log all executed hooks to journalctl for audit trail

**Claude CLI Invocation:**
- Risk: Subprocess calls to Claude CLI with `--permission-mode bypassPermissions` grant full system access
- Files: `push-to-talk.py` (lines 1406-1416, 1880-1889, 1878-1888), `openai_realtime.py` (lines 186-196)
- Current mitigation: Runs in constrained working directory, user controls what questions are asked
- Recommendations:
  - Consider downgrading to `acceptEdits` mode for conversation mode
  - Add option to disable full tool access for untrusted contexts
  - Log all Claude operations with timestamps and input/output summaries

**Interview Context Directory Reading:**
- Risk: User can provide arbitrary directory paths for context, which are read and sent to Claude
- Files: `push-to-talk.py` (lines 585-591)
- Impact: Could expose sensitive information in CLAUDE.md files to the LLM
- Recommendations: Add warning dialog before reading context, sanitize paths, option to exclude sensitive files

## Performance Bottlenecks

**Whisper Model Loading on Every Run:**
- Problem: Entire Whisper "small" model (~461MB) loaded at startup, blocks main thread
- Files: `push-to-talk.py` (lines 730-732)
- Cause: Model is CPU/memory intensive, no caching across sessions
- Improvement path:
  - Cache model to disk after first load
  - Load asynchronously in background thread during startup
  - Consider switching to faster model for stream mode (tiny/base)

**Realtime API Connection Overhead:**
- Problem: Each AI hotkey press establishes new WebSocket, fetches API key, initializes session
- Files: `openai_realtime.py` (lines 239-284), `push-to-talk.py` (lines 768-801)
- Cause: No connection pooling or persistent session
- Improvement path: Keep realtime session alive across multiple uses (toggle behavior exists but connection drops)

**Interview Post-Processing Synchronous ffmpeg Calls:**
- Problem: ffmpeg normalization, stitching, and MP3 conversion happen sequentially (can take 30+ seconds for 1-hour interview)
- Files: `push-to-talk.py` (lines 1676-1706)
- Improvement path: Run ffmpeg operations in parallel, show progress indication

**Transcription File Size Truncation:**
- Problem: Smart transcription and file reading truncate output to 5000 chars max
- Files: `openai_realtime.py` (lines 176), `push-to-talk.py` (line 944-950 context building)
- Impact: Large config files or long interviews lose context for AI
- Fix approach: Stream processing instead of truncation, or implement pagination

## Fragile Areas

**Interview/Conversation Session State Management:**
- Files: `push-to-talk.py` (lines 535-660, 1501-1809)
- Why fragile: Sessions can be interrupted mid-operation (e.g., user force-closes indicator, network dropout). No recovery or resumption mechanism. Temp files may be orphaned.
- Safe modification:
  - Add session persistence to disk
  - Implement recovery on startup
  - Clean up orphaned files on session init
- Test coverage: No unit tests for session state transitions

**Stream Mode Chunk Management:**
- Files: `push-to-talk.py` (lines 1125-1275)
- Why fragile: Overlapping chunks with deduplication heuristic is prone to race conditions if transcription completes out-of-order or if audio level detection fails
- Safe modification: Add explicit sequence numbers, validate chunk ordering before dedup
- Test coverage: No unit tests for dedup logic

**Multi-Mode Hotkey Logic:**
- Files: `push-to-talk.py` (lines 2073-2212)
- Why fragile: Complex state machine with PTT key, AI key, indicator status, interview/conversation status all interacting. Flag management is ad-hoc.
- Safe modification: Refactor to explicit state machine (e.g., enum `OperationMode` with clear transitions)
- Test coverage: No unit tests for hotkey sequences

## Scaling Limits

**Whisper Model Size:**
- Current capacity: ~461MB for "small" model in RAM continuously
- Limit: Can't run on systems with <1GB available RAM. Blocks startup for ~3-5 seconds on first launch
- Scaling path:
  - Lazy-load model only when first PTT press occurs
  - Add option to switch to "tiny" model for lower-end systems
  - Consider streaming transcription to cloud API instead

**Interview Session Audio Memory:**
- Current capacity: Stores all audio files (.wav) + transcript in memory until post-processing. 1-hour interview = ~500MB+ disk I/O
- Limit: Very long interviews (2+ hours) may hit disk space limits on embedded systems
- Scaling path: Stream audio to codec during recording instead of storing raw PCM

**Realtime API Simultaneous Users:**
- Current capacity: Single connection per instance
- Limit: Can't handle multiple PTT devices in same session concurrently
- Scaling path: Not applicable for single-user tool, but document this limitation

## Dependencies at Risk

**OpenAI Whisper Maintenance:**
- Risk: Whisper is AI model package maintained by OpenAI but not actively developed (last update 2023)
- Impact: No future optimizations, security fixes may lag
- Migration plan: Evaluate faster alternatives (Faster-Whisper, AssemblyAI API, Google Speech-to-Text)

**pynput Keyboard Listener:**
- Risk: pynput global keyboard listener can conflict with some desktop environments (Wayland, some KDE setups)
- Impact: Hotkey triggering fails silently on incompatible systems
- Migration plan: Add fallback to xdotool-based key detection, or systemwide hotkey service

**Piper TTS Local Model:**
- Risk: Piper voice models are static (trained once). No updates for better pronunciation
- Impact: Synthetic speech quality plateaus
- Alternative: Embed OpenAI TTS as primary with Piper fallback

**AppIndicator3 GTK3 Library:**
- Risk: GTK3 is deprecated in favor of GTK4. AppIndicator3 support is declining
- Impact: System tray indicator may break on next GNOME major release
- Migration plan: Use org.freedesktop.systemtray D-Bus API directly, or switch to GTK4 port

## Missing Critical Features

**No Persistent Configuration Schema Validation:**
- Problem: Config loaded from JSON with no schema validation. Adding new options or renaming old ones has no migration path
- Blocks: Large-scale config changes, team collaboration on settings
- Fix: Implement Pydantic config model with migration functions

**No Hotkey Conflict Detection:**
- Problem: User can set PTT and AI keys to same value. System will malfunction silently
- Blocks: User self-service setup. Requires Settings UI validation refactor
- Fix: Add validation in config save path, prevent equal key pairs

**No Rate Limiting on AI Calls:**
- Problem: Rapid-fire questions to Claude CLI or Realtime API have no throttling
- Risk: Accidental API quota exhaustion, runaway costs
- Fix: Add cooldown timers and queue system

## Test Coverage Gaps

**No Unit Tests for Core Logic:**
- What's not tested:
  - Hotkey state machine transitions
  - Stream mode deduplication
  - Interview session lifecycle
  - Transcription prompt building
- Files: `push-to-talk.py` (entire file)
- Risk: Regressions in core logic undetected
- Priority: High — hotkey logic has known flag issues

**No Integration Tests for Multi-Mode Flows:**
- What's not tested:
  - PTT → interview → stream follow-ups
  - Conversation mode with network interruption
  - Realtime session toggle stress
- Risk: Complex user journeys fail in production but work in isolation
- Priority: High

**No Tests for Error Paths:**
- What's not tested:
  - Missing ffmpeg during post-processing
  - Whisper model load failure
  - Claude CLI timeout
  - API key missing at runtime
- Files: All files
- Risk: Errors handled with bare `except:` blocks, actual behavior unknown
- Priority: Medium

**No GUI Tests:**
- What's not tested: Settings window, indicator popup, dialogs
- Files: `indicator.py`
- Risk: UI regressions undetected
- Priority: Low (UI less critical than core functionality)

---

*Concerns audit: 2026-02-13*
