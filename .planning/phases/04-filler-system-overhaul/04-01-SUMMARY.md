---
phase: 04-filler-system-overhaul
plan: 01
subsystem: audio
tags: [piper-tts, wav, numpy, audio-generation, clip-pool]

# Dependency graph
requires: []
provides:
  - clip_factory.py daemon for generating, evaluating, and rotating non-verbal clips
  - Seeded nonverbal clip pool (10 WAV files) ready for live session playback
  - pool.json metadata tracking with quality scores and generation params
affects:
  - 04-02 (wires clip factory into live session, removes smart filler)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Background daemon pattern (like learner.py) for clip generation subprocess"
    - "numpy-based audio quality evaluation (RMS, clipping, silence ratio)"
    - "JSON metadata sidecar for pool management"

key-files:
  created:
    - clip_factory.py
    - audio/fillers/nonverbal/*.wav
    - audio/fillers/pool.json
  modified: []

key-decisions:
  - "Single nonverbal/ category instead of multiple subcategories (YAGNI)"
  - "Synchronous subprocess.run (not asyncio) for Piper calls in factory"
  - "Sequential filename pattern (prompt_NNN.wav) with dedup against existing names"

patterns-established:
  - "Clip quality gate: duration 0.2-2.0s, RMS > 300, clipping < 1%, silence < 70%"
  - "Pool metadata schema: {filename, created_at, params, scores}"

# Metrics
duration: 2min 30s
completed: 2026-02-17
---

# Phase 4 Plan 01: Clip Factory Summary

**Piper TTS clip factory daemon with quality evaluation, pool rotation, and 10 seeded non-verbal WAV clips**

## Performance

- **Duration:** 2 min 30s
- **Started:** 2026-02-17T21:37:05Z
- **Completed:** 2026-02-17T21:39:35Z
- **Tasks:** 2
- **Files created:** 12 (1 Python + 10 WAV + 1 JSON)

## Accomplishments

- clip_factory.py generates non-verbal clips via Piper TTS with randomized synthesis params (length_scale, noise_w, noise_scale)
- Quality evaluation rejects bad clips: checks duration, RMS energy, clipping ratio, and silence ratio
- Pool rotation caps at 20 clips, removing oldest when exceeded
- 10 clips seeded and verified: all 22050Hz mono 16-bit, durations 0.31s-1.34s, all passing quality checks
- Daemon mode available for long sessions (periodic pool top-up)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create clip_factory.py daemon** - `0397ace` (feat)
2. **Task 2: Seed nonverbal clip pool** - `eed6393` (feat)

## Files Created/Modified

- `clip_factory.py` - Background daemon: generate, evaluate, save, rotate non-verbal clips via Piper TTS
- `audio/fillers/nonverbal/*.wav` - 10 generated non-verbal WAV clips (hmm, mmm, mhm, hm, ahh, uhh variants)
- `audio/fillers/pool.json` - Metadata tracking: filename, creation timestamp, synthesis params, quality scores

## Decisions Made

- Used single `nonverbal/` category rather than splitting into subcategories -- all non-verbal clips serve the same purpose, subcategories add complexity without value
- Synchronous `subprocess.run()` for Piper (not asyncio) -- clip factory is a standalone script, not part of the event loop
- Sequential filenames (`prompt_NNN.wav`) with dedup against existing pool -- simple and predictable

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None -- Piper generated all clips successfully on first run with zero rejections.

## User Setup Required

None -- no external service configuration required. Piper TTS and voice model already installed.

## Next Phase Readiness

- Non-verbal clip pool is seeded and ready for Plan 02 to wire into the live session
- clip_factory.py can be spawned as a subprocess by live_session.py (same pattern as learner.py)
- Plan 02 will: remove smart filler code, update `_load_filler_clips()` to use nonverbal/ category, add factory spawn to session start

---
*Phase: 04-filler-system-overhaul*
*Completed: 2026-02-17*
