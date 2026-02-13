---
phase: 01-mode-rename-and-live-voice-session
plan: 01
subsystem: dictation-mode
tags: [rename, config-migration, voice-commands, ui]
requires: []
provides:
  - "dictate" naming for dictation mode across codebase
  - config migration from "live" to "dictate"
  - documented dictation modes in README
affects:
  - 01-02 (live voice session can now use "live" name without collision)
tech-stack:
  added: []
  patterns:
    - config migration pattern (check old value, replace, save)
key-files:
  created: []
  modified:
    - push-to-talk.py
    - indicator.py
    - README.md
key-decisions:
  - Migration saves from push-to-talk.py only (indicator reads but does not persist migration)
  - Voice commands changed to "dictate mode" / "go dictate" / "dictation mode"
duration: ~2 minutes
completed: 2026-02-13
---

# Phase 01 Plan 01: Rename Dictation Mode Summary

Renamed dictation mode from "live" to "dictate" with config migration and updated voice commands, freeing "live" for real-time voice session mode.

## Performance

- Duration: ~2 minutes
- Tasks: 2/2 completed
- Deviations: 1 (auto-fixed stale fallback default)

## Accomplishments

1. Renamed all dictation mode references from "live" to "dictate" across push-to-talk.py and indicator.py
2. Added config migration logic that auto-converts existing "live" configs to "dictate"
3. Updated voice commands: "dictate mode" / "go dictate" / "dictation mode"
4. Updated Settings combo box: "Dictate (instant typing)"
5. Updated QuickControlWindow and PopupWindow mode mappings
6. Added Dictation Modes documentation section to README.md
7. Confirmed website files already use "Dictate" naming

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Rename dictation mode from "live" to "dictate" | a94d1f0 | push-to-talk.py, indicator.py |
| 2 | Update README.md with dictation mode rename | e5edb27 | README.md |

## Files Modified

- **push-to-talk.py**: Default config, config migration, voice commands (2 locations), prompt mode fallback default
- **indicator.py**: Default config, config migration, settings combo box, info text, PopupWindow mode_names, QuickControlWindow modes/defaults
- **README.md**: Added Dictation Modes section with mode descriptions and voice commands

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Indicator migration does not save to disk | push-to-talk.py is the primary service and handles persistence; indicator just reads |
| Voice command triggers are "dictate mode", "go dictate", "dictation mode" | Natural speech patterns for the new naming |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed stale fallback default at line 1085**
- **Found during:** Task 1 verification
- **Issue:** `self.config.get('dictation_mode', 'live')` still used 'live' as fallback default in the prompt mode check
- **Fix:** Changed fallback to 'dictate'
- **Files modified:** push-to-talk.py
- **Commit:** a94d1f0

## Issues Encountered

None.

## Next Phase Readiness

The "live" name is now free for use in Plan 02 (live voice session mode). All existing references point to "dictate", and config migration ensures users with old configs are automatically updated. No blockers for Plan 02.
