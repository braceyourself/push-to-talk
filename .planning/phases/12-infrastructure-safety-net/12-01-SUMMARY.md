---
phase: 12-infrastructure-safety-net
plan: 01
subsystem: stt
tags: [deepgram, websocket, streaming, vad, silero, transcript-buffer]

# Dependency graph
requires:
  - phase: none (first plan in phase 12)
    provides: n/a
provides:
  - DeepgramSTT class with same interface as ContinuousSTT
  - Streaming STT via Deepgram Nova-3 WebSocket
  - VAD-gated connection lifecycle (active/idle/sleep)
  - KeepAlive management for idle connections
  - Reconnection with exponential backoff
  - on_unavailable callback for fallback signaling
  - Comprehensive mock-based unit tests
affects: [12-02 (live_session integration), 12-03 (STT stage rewrite), 12-04 (dashboard events)]

# Tech tracking
tech-stack:
  added: [deepgram-sdk 5.3.2 (V1SocketClient, ListenV1ControlMessage, EventType)]
  patterns: [is_final/speech_final accumulation, VAD lifecycle gating, KeepAlive heartbeat]

key-files:
  created: [deepgram_stt.py, test_deepgram_stt.py]
  modified: []

key-decisions:
  - "DeepgramClient(api_key=...) keyword arg required (v5.3.2 positional arg rejected)"
  - "SDK v5.3.2 uses send_media()/send_control() not send()/keep_alive()/finish()"
  - "ListenV1ControlMessage(type='KeepAlive') for heartbeat, type='Finalize' for graceful close"
  - "VAD gates connection lifecycle (active/idle/sleep), not per-chunk audio filtering (Pitfall 1)"
  - "All connect() params must be strings (SDK v5.3.2 type annotations specify str)"

patterns-established:
  - "Transcript accumulation: accumulate is_final segments, flush on speech_final"
  - "DeepgramSTT mock strategy: use real ListenV1ResultsEvent objects, mock connection send_media/send_control"
  - "on_unavailable callback pattern for reconnection exhaustion fallback"

# Metrics
duration: 7min
completed: 2026-02-22
---

# Phase 12 Plan 01: DeepgramSTT Core Class Summary

**DeepgramSTT streaming STT class with mock WebSocket tests, VAD lifecycle gating, and is_final/speech_final accumulation**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-22T16:09:31Z
- **Completed:** 2026-02-22T16:16:08Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files created:** 2

## Accomplishments
- DeepgramSTT class (581 lines) replicating ContinuousSTT's external interface
- 13 unit tests (405 lines) covering all core behaviors with mock Deepgram SDK objects
- VAD lifecycle gating architecture: active (stream audio) / idle (KeepAlive) / sleep (disconnect)
- Discovered and adapted to Deepgram SDK v5.3.2 API surface (send_media/send_control, not send/keep_alive)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests for DeepgramSTT (RED)** - `5fcbfd5` (test)
2. **Task 2: Implement DeepgramSTT class (GREEN)** - `a42f48b` (feat)

## Files Created/Modified
- `deepgram_stt.py` - DeepgramSTT class: streaming STT, VAD lifecycle, transcript accumulation, reconnection
- `test_deepgram_stt.py` - 13 unit tests using mock Deepgram SDK objects

## Decisions Made
- **SDK v5.3.2 API adaptation:** The plan's ARCHITECTURE.md assumed `connection.send()`, `connection.keep_alive()`, `connection.finish()`. SDK v5.3.2 actually uses `send_media(bytes)`, `send_control(ListenV1ControlMessage)`. Adapted all call sites.
- **DeepgramClient keyword arg:** v5.3.2 rejects positional `api_key` argument; must use `DeepgramClient(api_key=...)`.
- **Connect params as strings:** All `connect()` parameters typed as `Optional[str]` in v5.3.2, so numeric values like `sample_rate=24000` must be passed as `"24000"`.
- **Context manager pattern:** `client.listen.v1.connect()` returns an iterator/context manager, yielding `V1SocketClient`. Must use `__enter__()/__exit__()` explicitly.
- **_on_message callback accepts *args, **kwargs:** SDK may pass extra arguments to event handlers; accept and ignore them for forward compatibility.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adapted to Deepgram SDK v5.3.2 actual API surface**
- **Found during:** Task 2 (implementation)
- **Issue:** Plan assumed `send()`, `keep_alive()`, `finish()` methods from research docs. SDK v5.3.2 has `send_media()`, `send_control(ListenV1ControlMessage)`, context manager pattern.
- **Fix:** Used `send_media()` for audio, `send_control(ListenV1ControlMessage(type='KeepAlive'))` for heartbeat, `send_control(ListenV1ControlMessage(type='Finalize'))` for graceful close. Used context manager for connection lifecycle.
- **Files modified:** deepgram_stt.py, test_deepgram_stt.py
- **Verification:** All 13 tests pass
- **Committed in:** a42f48b

**2. [Rule 3 - Blocking] DeepgramClient requires keyword api_key argument**
- **Found during:** Task 2 (implementation)
- **Issue:** `DeepgramClient(api_key)` as positional arg raises TypeError in v5.3.2
- **Fix:** Changed to `DeepgramClient(api_key=self._api_key)`
- **Files modified:** deepgram_stt.py
- **Verification:** Import and instantiation works
- **Committed in:** a42f48b

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both were SDK API surface mismatches between research and actual v5.3.2 implementation. No scope creep; same architecture, different method names.

## Issues Encountered
- The `_test_name` attribute on the test decorator function gets picked up as a spurious passing test in the test runner. This is a pre-existing pattern from test_live_session.py and is harmless (14 reported vs 13 real tests).

## User Setup Required
None - no external service configuration required. API key handling will be wired in plan 12-02 (live_session integration).

## Next Phase Readiness
- DeepgramSTT class ready for integration into live_session.py (plan 12-02)
- Interface matches ContinuousSTT exactly: start(), stop(), set_playing_audio(), running, stats
- on_unavailable callback ready for Whisper fallback wiring
- transcript_q ready for pipeline STT stage consumption

---
*Phase: 12-infrastructure-safety-net*
*Completed: 2026-02-22*
