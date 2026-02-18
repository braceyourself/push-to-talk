---
phase: 06-polish-verification
plan: 01
subsystem: stt
tags: [whisper, filtering, logprob, compression-ratio, overlay, gtk]

# Dependency graph
requires:
  - phase: 05-barge-in
    provides: VAD-gated STT pipeline with energy gate and hallucination phrase list
provides:
  - Multi-layer Whisper segment filtering (no_speech_prob, avg_logprob, compression_ratio)
  - stt_rejected status emission and overlay flash
affects: [06-02, 06-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "3-layer Whisper segment filtering: no_speech_prob >= 0.6, avg_logprob < -1.0, compression_ratio > 2.4"
    - "Transient overlay status: stt_rejected handled as visual flash without state transition"

key-files:
  created: []
  modified:
    - live_session.py
    - indicator.py

key-decisions:
  - "avg_logprob threshold -1.0: conservative enough to not reject quiet speech, aggressive enough to catch coughs"
  - "compression_ratio threshold 2.4: standard Whisper hallucination detection threshold"
  - "stt_rejected as transient flash, not a state transition — avoids cluttering status history"

patterns-established:
  - "Transient overlay cue pattern: intercept in update_status, flash via GLib.timeout_add, return early without state change"
  - "Per-segment diagnostic logging with metric values for STT debugging"

# Metrics
duration: 1min
completed: 2026-02-18
---

# Phase 6 Plan 1: STT Filtering Summary

**3-layer Whisper segment filtering (no_speech_prob, avg_logprob, compression_ratio) with 300ms overlay rejection flash**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-18T12:47:30Z
- **Completed:** 2026-02-18T12:48:40Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Enhanced _whisper_transcribe with 3-layer segment filtering to catch throat clears, coughs, and hallucinated text
- Added stt_rejected status emission when all segments are rejected
- Overlay dims status dot for 300ms as transient rejection indicator
- Per-segment diagnostic logging with metric values for debugging

## Task Commits

Each task was committed atomically:

1. **Task 1: Add multi-layer Whisper segment filtering** - `4008eed` (feat)
2. **Task 2: Add STT rejection flash to overlay** - `3d99bed` (feat)

## Files Created/Modified
- `live_session.py` - Multi-layer Whisper segment filtering in _whisper_transcribe (no_speech_prob, avg_logprob, compression_ratio)
- `indicator.py` - stt_rejected flash handling in LiveOverlayWidget (300ms dot dim)

## Decisions Made
- avg_logprob threshold set to -1.0: conservative enough to preserve quiet speech, catches low-confidence gibberish from coughs/clears
- compression_ratio threshold set to 2.4: standard Whisper hallucination detection value for repetitive text
- stt_rejected implemented as transient visual cue (300ms dot dim) rather than state transition -- avoids polluting status history with noise events

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- STT filtering in place, ready for Plan 02 (status label polish) and Plan 03 (overlay visual refinements)
- stt_rejected status pattern established for Plan 03 to potentially enhance visual treatment
- Existing hallucination phrase list and energy gate remain untouched as independent defense layers

---
*Phase: 06-polish-verification*
*Completed: 2026-02-18*
