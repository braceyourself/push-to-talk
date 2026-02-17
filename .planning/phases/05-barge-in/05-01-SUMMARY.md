---
phase: 05-barge-in
plan: 01
subsystem: voice-pipeline
tags: [vad, barge-in, stt-gating, silero, asyncio, pyaudio]

# Dependency graph
requires:
  - phase: 04-filler-overhaul
    provides: filler clip system (nonverbal clips, _pick_filler, _play_filler_audio)
provides:
  - STT gating mechanism (_stt_gated flag replaces mic muting during playback)
  - VAD-based barge-in detection in STT stage (sustained speech triggers interruption)
  - _trigger_barge_in method with fade-out, trailing filler, cooldown
  - BARGE_IN frame type in pipeline_frames.py
  - Gated->ungated transition handling (_was_stt_gated) for clean STT state reset
affects: [05-02 context-aware-response]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "STT gating: mic stays live during playback, STT stage gates audio processing"
    - "VAD-in-STT: VAD runs inline in STT stage during gated state, not in separate stage"
    - "Generation ID interrupt: barge-in increments generation_id to discard all stale frames"
    - "Trailing filler: brief faded nonverbal clip after interruption for natural audio transition"

key-files:
  created: []
  modified:
    - live_session.py
    - pipeline_frames.py

key-decisions:
  - "Gate STT instead of muting mic via pactl — mic stays live so VAD can hear user speech during AI playback"
  - "VAD runs inline in STT stage (Branch 2) rather than in a separate monitor stage — eliminates need for shared audio queue"
  - "6 consecutive VAD-positive chunks (~0.5s) threshold for barge-in trigger — balances responsiveness vs false positives"
  - "1.5s cooldown after barge-in prevents rapid-fire re-triggers"
  - "Trailing filler: 150ms of nonverbal clip with 0.8->0.0 linear fade for natural audio cutoff"

patterns-established:
  - "Three-branch audio processing in _stt_stage: muted / gated (VAD) / normal"
  - "Gated->ungated transition detection via _was_stt_gated flag to reset silence tracking state"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 5 Plan 1: Core Barge-in Mechanism Summary

**STT gating replaces mic muting during AI playback, VAD detects sustained user speech (~0.5s) in STT stage, triggers playback interruption with generation_id increment, queue drain, trailing faded filler clip, and 1.5s cooldown**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-17T23:19:03Z
- **Completed:** 2026-02-17T23:21:54Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- Replaced mic muting (pactl) with STT gating during AI playback — mic stays physically live for VAD
- Wired Silero VAD into STT stage to detect sustained speech during playback (6 chunks / ~0.5s threshold)
- Implemented _trigger_barge_in with full state cleanup: delayed_unmute cancellation, generation_id increment, queue drain, filler cancellation, 1.5s cooldown, VAD state reset, trailing faded filler clip
- Added gated->ungated transition detection (_was_stt_gated) to reset all silence tracking state cleanly
- Removed _vad_monitor_stage stub (VAD now runs inline in STT stage)

## Task Commits

Each task was committed atomically:

1. **Task 1a: Replace mic muting with STT gating in playback stage** - `1c64a7f` (feat)
2. **Task 1b: Wire VAD into STT stage, handle gated->ungated transition, remove monitor stub** - `1ed1b2a` (feat)
3. **Task 2: Implement barge-in trigger with fade-out, trailing filler, and cooldown** - `b32827a` (feat)

## Files Created/Modified
- `pipeline_frames.py` - Added BARGE_IN frame type to FrameType enum
- `live_session.py` - STT gating flags, 3-branch audio processing in _stt_stage, _trigger_barge_in method, _reset_vad_state helper, VAD model loading in run()

## Decisions Made
- Gate STT instead of muting mic via pactl — mic stays live so VAD can hear user speech during AI playback
- VAD runs inline in STT stage (Branch 2) rather than in a separate monitor stage — eliminates shared audio queue complexity
- 6 consecutive VAD-positive chunks (~0.5s) threshold — balances responsiveness vs false positives from coughs/background noise
- 1.5s cooldown after barge-in prevents rapid-fire re-triggers
- Trailing filler: 150ms of nonverbal clip with 0.8->0.0 linear fade — provides natural audio transition instead of abrupt silence

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Core barge-in mechanism complete and wired into pipeline
- Ready for Plan 02: context-aware response (feeding interrupted context + user speech to Claude for intelligent continuation)
- VAD model (silero_vad.onnx) must be present in models/ directory for barge-in to activate

---
*Phase: 05-barge-in*
*Completed: 2026-02-17*
