---
phase: 12-infrastructure-safety-net
plan: 02
subsystem: infra
tags: [deepgram-sdk, requirements, config, api-key]

# Dependency graph
requires:
  - phase: none
    provides: existing push-to-talk codebase with config.json pattern
provides:
  - deepgram-sdk pinned to stable v5 range (>=5.3,<6.0)
  - Deepgram config defaults (idle_timeout, sleep_timeout, endpointing_ms, utterance_end_ms, daily_budget_cents)
  - Verified API key loading pipeline (env var or ~/.config/deepgram/api_key)
affects: [12-01, 12-03, 12-04, 12-05]

# Tech tracking
tech-stack:
  added: [deepgram-sdk 5.3.x]
  patterns: [config defaults in load_config(), API key from env or file]

key-files:
  created: []
  modified: [requirements.txt, push-to-talk.py]

key-decisions:
  - "Pin deepgram-sdk >=5.3,<6.0 (not exact pin) to allow patch updates within stable v5"
  - "Add Deepgram config as flat keys with deepgram_ prefix in existing config.json pattern (no nested YAML)"
  - "No code changes for Task 2 -- existing get_deepgram_api_key() and LiveSession wiring already complete"

patterns-established:
  - "Deepgram config keys use deepgram_ prefix in config.json defaults"
  - "API key loading: env var first, then ~/.config/<provider>/api_key file"

# Metrics
duration: 1min
completed: 2026-02-22
---

# Phase 12 Plan 02: Config and Requirements Update Summary

**Deepgram SDK pinned to v5 stable range with config defaults for streaming STT lifecycle management**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-22T16:10:52Z
- **Completed:** 2026-02-22T16:12:06Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Pinned deepgram-sdk to >=5.3,<6.0, avoiding v6 breaking changes while allowing patch updates
- Added 5 Deepgram-specific config defaults: idle_timeout (10s), sleep_timeout (60s), endpointing_ms (300), utterance_end_ms (1000), daily_budget_cents (200)
- Verified existing API key loading pipeline works end-to-end (env var -> file -> LiveSession constructor)
- Confirmed faster-whisper retained as offline STT fallback

## Task Commits

Each task was committed atomically:

1. **Task 1: Update requirements.txt and config** - `b01f905` (chore)
2. **Task 2: Wire Deepgram API key loading** - no commit (verification only, existing code already complete)

## Files Created/Modified
- `requirements.txt` - Pinned deepgram-sdk from >=3.0 to >=5.3,<6.0; updated comment
- `push-to-talk.py` - Added 5 Deepgram config defaults to load_config() default dict

## Decisions Made
- Used range pin (>=5.3,<6.0) instead of exact pin (==5.3.2) since the plan specified this range and it allows patch updates within the stable v5 line
- Added config as flat keys with `deepgram_` prefix in config.json (matching the existing flat config pattern) rather than creating a config.yaml or nested structure
- Task 2 required no code changes -- the existing `get_deepgram_api_key()` function and LiveSession constructor wiring were already complete and correct

## Deviations from Plan

None - plan executed exactly as written. The plan correctly anticipated that Task 2 might require no changes ("If the function exists and works: No changes needed").

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. Deepgram API key setup is documented in existing codebase (env var or ~/.config/deepgram/api_key).

## Next Phase Readiness
- requirements.txt ready for `pip install -r requirements.txt` in the deployment venv
- Config defaults available for DeepgramSTT class (Plan 01) to read via load_config()
- API key pipeline verified for LiveSession integration (Plan 04)

---
*Phase: 12-infrastructure-safety-net*
*Completed: 2026-02-22*
