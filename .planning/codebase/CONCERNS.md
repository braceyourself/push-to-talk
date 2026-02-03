# Codebase Concerns

**Analysis Date:** 2026-02-03

## Tech Debt

**Bare Exception Handlers:**
- Issue: Multiple bare `except:` clauses throughout codebase silently swallow all exceptions, making debugging difficult
- Files: `push-to-talk.py` (lines 231, 275, 281, 621, 630, 670, 698, 718, 799), `openai_realtime.py` (lines 144, 352), `indicator.py` (lines 71, 81, 576, 817, 830, 1045, 1166)
- Impact: Critical bugs may fail silently. Error conditions are not properly logged or handled. Makes troubleshooting user issues nearly impossible
- Fix approach: Replace with specific exception types (IOError, ValueError, etc.) and log all exceptions with context

**Subprocess Shell Injection Risk:**
- Issue: Piper TTS command uses `shell=True` with user-controlled input via `subprocess.list2cmdline()` and string interpolation
- Files: `push-to-talk.py` lines 501 (Piper command) - uses shell=True with echo piped to piper
- Impact: If TTS text contains shell metacharacters, could lead to unintended command execution
- Fix approach: Convert to list-based subprocess call without shell=True, pass text via stdin instead of echo

**Global Mutable State in Module:**
- Issue: Global variables like `indicator_process`, TTS process tracking, and config state can lead to race conditions
- Files: `push-to-talk.py` lines 224, 349 - global indicator_process and instance-level tts_process
- Impact: With threading (recordings happen in background threads), state can be corrupted. Auto-listen and concurrent TTS operations may interfere
- Fix approach: Wrap state management in thread-safe patterns (locks, queues) or use proper async/await throughout

**Unchecked File Operations:**
- Issue: Files are read/written without existence validation in several places, may fail silently
- Files: `push-to-talk.py` lines 230-232 (set_status writes to STATUS_FILE without error handling), `indicator.py` lines 814-818 (VOCAB_FILE.read_text() called without existence check before parsing)
- Impact: If status file write fails, indicator becomes out of sync. Vocab parsing could crash if file is malformed
- Fix approach: Validate file operations and log failures; use try/except with specific error types

**Memory Leak in Vocabulary Manager:**
- Issue: Words are stored in a set but never pruned; vocabulary file only grows
- Files: `push-to-talk.py` lines 303-327 (VocabularyManager.add and save)
- Impact: Over time, vocabulary.txt becomes large with potentially invalid/misspelled words. No deduplication on load
- Fix approach: Implement word validation (length, character set), deduplicate on load, optionally implement LRU eviction

## Known Bugs

**Realtime API Mic Muting Race Condition:**
- Symptoms: Microphone can become stuck muted or unmuted if rapid keystrokes or interrupts occur
- Files: `openai_realtime.py` lines 340-346, 413-418, 423-427 - multiple places calling pactl set-source-mute
- Trigger: Quickly pressing interrupt key, or user speaking while AI response is ending
- Workaround: Press interrupt key or manually unmute with `pactl set-source-mute @DEFAULT_SOURCE@ 0`
- Root cause: Concurrent mute/unmute calls from event handlers and delayed_unmute task without synchronization

**Recording Release Event Spurious Triggering:**
- Symptoms: Recording stops immediately after starting when using AI mode with both keys held
- Files: `push-to-talk.py` lines 928-936 (on_release logic with elapsed time check)
- Trigger: Rapidly pressing both Right Ctrl and Right Shift in succession
- Workaround: Hold both keys for at least 0.5 seconds before starting to speak
- Root cause: pynput may generate spurious release events; the 0.5s threshold is arbitrary and may not work for fast typists

**Audio Level Detection Unreliable:**
- Symptoms: Auto-listen feature may not trigger reliably on silent/low-volume audio
- Files: `push-to-talk.py` lines 661-672 (get_audio_level with SILENCE_THRESHOLD check)
- Trigger: User speaking very softly or with poor microphone
- Workaround: Set SILENCE_THRESHOLD lower or disable auto-listen
- Root cause: sox amplitude detection can vary widely depending on audio equipment; threshold is hardcoded

**Claude Session Directory Not Cleaned:**
- Symptoms: `~/.local/share/push-to-talk/claude-session/` grows indefinitely with temp files
- Files: `push-to-talk.py` lines 760, 808, 820 - CLAUDE_SESSION_DIR created but never cleaned
- Trigger: Every Claude AI query creates session state
- Workaround: Manually clean directory with `rm -rf ~/.local/share/push-to-talk/claude-session/*`
- Root cause: No cleanup routine; claude-session meant to persist context but may accumulate cruft

## Security Considerations

**API Key Storage in Text Files:**
- Risk: OpenAI API key stored in plain text at `~/.config/openai/api_key`
- Files: `push-to-talk.py` lines 41-52 (reads from multiple paths including ~/.config/openai/api_key), `openai_realtime.py` lines 511-522
- Current mitigation: File is chmod 600 (read-only by owner); key not logged to stdout
- Recommendations:
  - Consider using system keyring (python-keyring) for storage
  - Never pass API key to subprocess command line (vulnerable to `ps` inspection)
  - Current implementation passes to OpenAI client correctly (not on cmdline)

**Subprocess Command Execution from Voice Input:**
- Risk: Realtime AI can execute arbitrary shell commands via `run_command` tool
- Files: `openai_realtime.py` lines 162-171 (execute_tool run_command implementation)
- Current mitigation: Limited to 30s timeout, output capped at 2000 chars
- Recommendations:
  - Add whitelist of allowed commands or deny list of dangerous commands (rm, dd, rm -rf, etc.)
  - Use subprocess with shell=False and argument list instead of shell=True
  - Log all executed commands to audit trail
  - Consider sandboxing or capability restrictions

**File Write via Voice Input:**
- Risk: Realtime AI can write arbitrary files via `write_file` tool
- Files: `openai_realtime.py` lines 180-184 (write_file implementation)
- Current mitigation: None - tool accepts any path
- Recommendations:
  - Restrict file writing to specific directories (e.g., ~/Documents)
  - Implement filename validation to prevent path traversal (../../etc/passwd)
  - Require confirmation before writing to system directories or dotfiles
  - Log all file operations

**Plaintext Config with Sensitive Data:**
- Risk: config.json may contain API keys or other sensitive configuration
- Files: `push-to-talk.py` lines 148, 174-180 (CONFIG_FILE storage)
- Current mitigation: Currently only stores hotkeys and UI preferences (not keys)
- Recommendations: Keep current separation - never store API keys in config.json

## Performance Bottlenecks

**Whisper Model Loaded Once, Shared Across Threads:**
- Problem: `self.model` is shared between main thread and transcription threads with lock contention
- Files: `push-to-talk.py` lines 393-395 (model loaded once), 578-584, 744-749, 685-690 (transcribe calls with model_lock)
- Cause: Whisper model is large (~1.5GB for small model); can't load per-thread. Lock serializes all transcriptions
- Improvement path:
  - If multiple simultaneous recordings needed, pre-load model into GPU memory (requires CUDA)
  - Consider lighter models (tiny/base) for faster transcription with acceptable accuracy
  - Profile actual lock contention - may not be significant in practice for single-user tool

**Vocabulary Prompt Generation Every Transcription:**
- Problem: Vocabulary prompt is regenerated on every transcription (sorted words, string concat)
- Files: `push-to-talk.py` lines 329-333 (get_prompt called on every transcription at lines 570)
- Cause: Naive implementation regenerates prompt even if vocabulary unchanged
- Improvement path: Cache prompt, only regenerate on vocabulary add. Pre-sort vocabulary at load time

**TTS Blocking on Output:**
- Problem: `self.tts_process.wait()` blocks entire event thread while audio plays
- Files: `push-to-talk.py` line 846 (TTS blocking wait in handle_ai_response)
- Cause: Synchronous subprocess wait prevents interrupt key handling during playback
- Improvement path: Use non-blocking wait with threading.Event or select on stderr

## Fragile Areas

**Realtime Event Handling Complexity:**
- Files: `openai_realtime.py` lines 318-433 (handle_events async function, 116 lines)
- Why fragile: Complex state machine with mic mute/unmute, playing_audio flag, audio_done_time tracking. Multiple concurrent async tasks. Easy to introduce echo or stuck mute states
- Safe modification: Add comprehensive logging before state changes. Use state machine pattern explicitly. Test with rapid interrupts
- Test coverage: No unit tests for event handling. Manual testing required

**Key Release Detection Logic:**
- Files: `push-to-talk.py` lines 916-940 (on_release method)
- Why fragile: Complex conditions checking if PTT mode vs AI mode, elapsed time, stale releases. Spurious release events from pynput cause state corruption
- Safe modification: Consider debouncing key events. Log state transitions. Test with hardware keyboard monitoring
- Test coverage: No automated key event testing

**Thread Safety of Recording State:**
- Files: `push-to-talk.py` lines 522-545 (start_recording) and 942-964 (stop_recording_with_mode)
- Why fragile: Recording state (self.recording, self.record_process, self.temp_file) accessed from main thread and keyboard listener without locks
- Safe modification: Add Lock around all state changes. Use threading.Event for synchronization instead of flag variables
- Test coverage: No concurrency testing

**Indicator Status File Sync:**
- Files: `push-to-talk.py` lines 227-232 (set_status), `indicator.py` lines 1039-1047 (check_status polling)
- Why fragile: Status file is inter-process communication between main process and indicator process. Polling every 100ms. Race condition if status changes between polls
- Safe modification: Use inotify or named pipes instead of polling files. Or use dbus for proper IPC
- Test coverage: None - would need to test process synchronization

## Scaling Limits

**Single Whisper Model Instance:**
- Current capacity: 1 concurrent transcription (serialized by model_lock)
- Limit: If user tries to transcribe while auto-listen is running, second request blocks
- Scaling path: Load model into GPU if available. Or use faster inference with quantization (ONNX)

**Realtime Audio Chunk Buffering:**
- Current capacity: CHUNK_SIZE=4096 bytes at 24kHz = ~85ms of audio per read
- Limit: If network latency spikes, audio buffer could underrun or WebSocket recv could block
- Scaling path: Implement proper audio queue with overflow handling. Add jitter buffer

**Indicator Polling Interval:**
- Current capacity: Checks status file every 100ms
- Limit: With many processes writing status simultaneously, could miss updates or have stale reads
- Scaling path: Switch to event-based notification (inotify, dbus, or file descriptor notifications)

## Dependencies at Risk

**Deprecated pynput Library:**
- Risk: pynput has known issues with modern Linux DE event handling; project has limited maintenance
- Impact: Key listener might stop working with next GNOME/KDE release; spurious release events documented in known bugs
- Migration plan: Consider `python-xlib` with direct X11 event handling, or `evdev` for device-level input (requires root)

**OpenAI Realtime API Beta:**
- Risk: Using preview model `gpt-4o-realtime-preview-2024-12-17` - subject to breaking changes
- Impact: API could change or be deprecated; prompt format may become incompatible
- Migration plan: Monitor OpenAI API changelog. Plan for fallback to non-realtime API if needed

**Whisper Model Download:**
- Risk: Model is downloaded on first use from OpenAI's CDN (~1.5GB for small model)
- Impact: First run requires internet and long download. Network failures leave broken installation
- Migration plan: Document model caching. Consider pre-packaging model with installer

**Piper TTS via Shell:**
- Risk: piper-tts requires bash/shell subprocess for piping echo output
- Impact: Shell metacharacters in TTS text could cause issues (see subprocess shell injection above)
- Migration plan: Use piper Python API directly if available, or write text to temp file instead of echo

## Missing Critical Features

**No Persistent Session History:**
- Problem: Claude conversation context is lost after auto-listen timeout
- Blocks: Can't have multi-turn conversations without manually re-asking context
- Workaround: Claude session dir attempts to persist, but no explicit context management

**No Voice Selection for Realtime API:**
- Problem: Hardcoded to "alloy" voice in Realtime mode
- Blocks: Users can't switch voices during session (OpenAI has voice.alloy, voice.echo, voice.fable, voice.onyx, voice.shimmer)
- Workaround: Edit REALTIME_URL in `openai_realtime.py` line 27 to change voice

**No Error Recovery in TTS:**
- Problem: If TTS fails silently, user gets no feedback
- Blocks: Can't diagnose why AI response isn't spoken
- Workaround: Check logs with journalctl

**No Explicit Tool Permission Prompts:**
- Problem: Realtime AI can execute any tool without confirmation
- Blocks: User may not realize AI just ran a shell command
- Workaround: None - trust the AI not to do bad things

## Test Coverage Gaps

**No Unit Tests:**
- What's not tested: VocabularyManager, CORRECTION_PATTERNS parsing, audio level detection, config file I/O
- Files: `push-to-talk.py` lines 285-333 (VocabularyManager), lines 507-520 (detect_correction), lines 235-253 (get_audio_level)
- Risk: Vocabulary parsing could be broken silently. Correction detection may not work as expected. Config corruption undetected
- Priority: High - these are core features

**No Integration Tests:**
- What's not tested: Recording → transcription → typing workflow end-to-end. AI conversation flow. Status indicator sync
- Files: Full workflow involves multiple processes and async operations
- Risk: A change that breaks the main feature could go unnoticed. Regressions in audio pipeline not caught
- Priority: High - this is the main feature

**No Concurrency Tests:**
- What's not tested: Rapid key presses, overlapping recordings, simultaneous transcription + auto-listen
- Files: Threading model at `push-to-talk.py` lines 336-349 and subprocess spawning
- Risk: Race conditions in production (stuck mutes, corrupted state) could occur intermittently
- Priority: Medium - only affects edge cases but hard to debug

**No Realtime API Integration Tests:**
- What's not tested: WebSocket reconnection, function tool execution, voice quality, error handling
- Files: `openai_realtime.py` entire module
- Risk: Features may not work in production. Tool execution might be silent failures. Error recovery untested
- Priority: Medium - complex subsystem that's hard to debug in field

**No Indicator Popup Tests:**
- What's not tested: GTK UI rendering, drag-and-drop, popup positioning, log display
- Files: `indicator.py` lines 128-690 (SettingsWindow), 692-851 (StatusPopup), 854-1066 (StatusIndicator)
- Risk: UI could be broken in different GTK versions or desktop environments
- Priority: Low - UI issues are user-visible and reported quickly

---

*Concerns audit: 2026-02-03*
