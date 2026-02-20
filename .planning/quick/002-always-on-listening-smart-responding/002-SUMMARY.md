---
phase: quick
plan: 002
subsystem: pipeline
tags: [asyncio, idle-timeout, live-session, config]

# Dependency graph
requires:
  - phase: quick-001
    provides: Personality restructure (always-on philosophy)
provides:
  - Configurable idle timeout for live sessions
  - Always-on mode (idle_timeout=0) as default
affects: [live-session, deployment, config]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config-driven constructor params with 0-means-disabled pattern"

key-files:
  created: []
  modified:
    - live_session.py
    - push-to-talk.py
    - indicator.py
    - test_live_session.py

key-decisions:
  - "Default idle_timeout is 0 (always-on) instead of 120s -- matches always-on philosophy"
  - "No UI control for idle timeout -- config.json-only setting for power users"

patterns-established:
  - "Idle timeout guard: check <= 0 before scheduling timer"

# Metrics
duration: 2min
completed: 2026-02-20
---

# Quick Task 002: Configurable Idle Timeout Summary

**Configurable live_idle_timeout with 0=always-on default, replacing hardcoded 120s disconnect**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-20T16:56:05Z
- **Completed:** 2026-02-20T16:57:55Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- LiveSession now accepts idle_timeout constructor param (default 0 = never timeout)
- Timer is never scheduled when timeout <= 0, so session stays alive indefinitely
- Config option `live_idle_timeout` wired from config.json through push-to-talk.py to LiveSession
- 4 new TDD tests covering default value, custom value, timer skip, and timer creation

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests for configurable idle timeout** - `5307d80` (test)
2. **Task 2: Implement configurable idle timeout** - `d5450f8` (feat)

## Files Created/Modified
- `live_session.py` - Added idle_timeout param to __init__, guard in _reset_idle_timer
- `push-to-talk.py` - Added live_idle_timeout config default, plumbed to LiveSession constructor
- `indicator.py` - Added live_idle_timeout to config defaults (sync)
- `test_live_session.py` - 4 new tests (Test Group 20: Configurable idle timeout)

## Decisions Made
- Default idle_timeout changed from 120 to 0 (always-on) to match the always-on listening philosophy
- No Settings UI toggle added -- this is a config.json-only option for users who want the old behavior

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required. Users who want the old 120s timeout behavior can set `"live_idle_timeout": 120` in config.json.

## Next Phase Readiness
- Always-on listening now works without idle disconnection
- Config infrastructure ready for additional timeout-related settings if needed

---
*Quick task: 002*
*Completed: 2026-02-20*
