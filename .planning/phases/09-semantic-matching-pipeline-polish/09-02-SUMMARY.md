---
phase: 09-semantic-matching-pipeline-polish
plan: 02
subsystem: pipeline
tags: [asyncio, tts, audio-pipeline, barge-in, cadence]

# Dependency graph
requires:
  - phase: 08
    provides: "Pipeline frames infrastructure and filler playback patterns"
provides:
  - "StreamComposer asyncio class with unified segment queue"
  - "SegmentType enum and AudioSegment dataclass"
  - "FrameType.SENTENCE_DONE first-class frame type"
affects: [09-03, 09-04, 09-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lookahead buffer pattern for FIFO-safe queue peeking"
    - "Synchronous pause() for instant barge-in response"
    - "Background TTS prefetch via asyncio.create_task"

key-files:
  created:
    - stream_composer.py
  modified:
    - pipeline_frames.py

key-decisions:
  - "Lookahead buffer instead of get+put-back for queue peek (preserves FIFO order)"
  - "Silence emitted as TTS_AUDIO frames (same type as spoken audio for playback stage compatibility)"
  - "SENTENCE_DONE inserted between BARGE_IN and CONTROL in FrameType enum"

patterns-established:
  - "StreamComposer pattern: content producers enqueue AudioSegments, composer handles TTS+cadence+output"
  - "Cadence parameters as tunable instance attributes (inter_sentence_pause, post_clip_pause, thinking_pause)"

# Metrics
duration: 4min
completed: 2026-02-19
---

# Phase 9 Plan 2: StreamComposer Summary

**Unified audio sentence queue with per-sentence TTS, lookahead pre-buffering, and synchronous barge-in drain**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T03:52:17Z
- **Completed:** 2026-02-19T03:56:56Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- StreamComposer class manages unified queue accepting FILLER_CLIP, TTS_SENTENCE, SILENCE, and NON_SPEECH segments
- Per-sentence TTS generation via async callback with automatic inter-segment pauses (150ms/250ms/400ms)
- Lookahead-based pre-buffering starts next sentence's TTS while current segment plays
- Synchronous pause() for instant barge-in response with unplayed segment return
- SENTENCE_DONE added as first-class FrameType (replaces CONTROL frame with data="sentence_done")

## Task Commits

Each task was committed atomically:

1. **Task 1: Create StreamComposer class with unified audio queue** - `6305d3f` (feat)

## Files Created/Modified
- `stream_composer.py` - StreamComposer class, SegmentType enum, AudioSegment dataclass
- `pipeline_frames.py` - Added SENTENCE_DONE to FrameType enum

## Decisions Made
- **Lookahead buffer for queue peek:** asyncio.Queue lacks a peek/put-front operation. Initial get+put-back approach caused FIFO reordering (peeked item went to back of queue, behind EOT sentinel). Switched to a `_lookahead` slot: `_try_get_next()` extracts the next item, stores it in `_lookahead`, and `_next_segment()` consumes it before the queue on the next iteration. This preserves strict FIFO ordering.
- **Silence as TTS_AUDIO frames:** Inter-segment silence is emitted as zero-byte TTS_AUDIO frames rather than a distinct frame type, maintaining compatibility with the existing playback stage which already handles TTS_AUDIO.
- **SENTENCE_DONE placement:** Added between BARGE_IN and CONTROL in the enum to keep related frame types grouped. Existing code references FrameType values by name, so the shifted CONTROL integer value is safe.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed queue peek reordering bug**
- **Found during:** Task 1 (verification testing)
- **Issue:** `_peek_next()` used get+put-back on asyncio.Queue, which put the peeked item at the back of the queue instead of the front. This caused the EOT sentinel to be consumed before the peeked sentence, breaking frame ordering.
- **Fix:** Replaced peek with a lookahead buffer pattern -- `_try_get_next()` gets the next item non-blockingly, stores it in `_lookahead`, and `_next_segment()` checks the buffer before the queue. Added `_NO_LOOKAHEAD` sentinel to distinguish empty buffer from None/EOT.
- **Files modified:** stream_composer.py
- **Verification:** Frame ordering test passes -- FILLER before TTS_AUDIO before SENTENCE_DONE before END_OF_TURN
- **Committed in:** 6305d3f (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix was essential for correct frame ordering. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- StreamComposer is ready for integration into live_session.py (Phase 9 Plan 3+)
- The tts_fn callback interface matches the existing _tts_to_pcm pattern
- No blockers or concerns

---
*Phase: 09-semantic-matching-pipeline-polish*
*Completed: 2026-02-19*
