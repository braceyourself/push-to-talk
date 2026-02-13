---
phase: 01-mode-rename-and-live-voice-session
plan: 02
subsystem: voice-session
tags: [openai-realtime, websocket, gtk, pipewire, pactl]

requires:
  - phase: 01-01
    provides: "dictation mode renamed to 'dictate', freeing 'live' name"
provides:
  - "LiveSession class for OpenAI Realtime voice conversation"
  - "Personality system loaded from markdown files"
  - "LiveOverlayWidget with click-to-cycle modes (listening/muted/idle)"
  - "Audio ducking of other streams while AI speaks"
  - "Config watcher for auto-start/stop live sessions"
affects: [phase-2-task-infra, phase-3-voice-orchestration]

tech-stack:
  added: []
  patterns:
    - "Signal file IPC between indicator and live session"
    - "Personality loaded from personality/*.md directory"
    - "Floating transparent GTK overlay with Cairo drawing"

key-files:
  created:
    - live_session.py
    - personality/core.md
    - personality/voice-style.md
    - personality/context.md
  modified:
    - push-to-talk.py
    - indicator.py

key-decisions:
  - "Voice validation: unsupported voices fall back to 'ash' for Realtime API"
  - "Overlay click cycles through listening→muted→idle→restart"
  - "Audio ducking to 15% for other sink inputs while AI speaks"
  - "Signal file IPC (live_mute_toggle) for overlay↔session communication"
  - "Config watcher polls config.json mtime every 500ms for mode changes"

patterns-established:
  - "Signal file pattern: overlay writes command file, session reads and deletes"
  - "Personality directory: sorted *.md files concatenated as system prompt"

duration: 40min
completed: 2026-02-13
---

# Phase 1 Plan 02: Live Voice Session Summary

**LiveSession with OpenAI Realtime API, personality system, overlay widget with click-to-cycle modes, and audio ducking**

## Performance

- **Duration:** 40 min
- **Started:** 2026-02-13T19:58:12Z
- **Completed:** 2026-02-13T20:38:00Z
- **Tasks:** 2 auto + 1 checkpoint (verified)
- **Files modified:** 6

## Accomplishments
- LiveSession class with WebSocket connection to OpenAI Realtime API, conversation state tracking, idle timeout, and context summarization
- Personality system: 3 markdown files (core, voice-style, context) loaded as system prompt
- LiveOverlayWidget: floating transparent GTK window with colored status dot, draggable, click-to-cycle (listening→muted→idle→restart)
- Audio ducking: other audio streams lowered to 15% while AI speaks, restored after
- Config watcher thread for auto-start/stop live sessions on AI mode change
- Voice commands "live mode" / "go live" / "going live" switch to live AI mode

## Task Commits

Each task was committed atomically:

1. **Task 1: Create LiveSession class and personality system** - `2a2abfd` (feat)
2. **Task 2: Wire live mode into push-to-talk.py and indicator.py** - `af3d808` (feat)
3. **Verification fixes** - `6d1e0df`..`ad6b728` (auto-sync, 6 commits)

## Files Created/Modified
- `live_session.py` - LiveSession and ConversationState classes (350+ lines)
- `personality/core.md` - AI behavioral rules, contradiction handling, English language
- `personality/voice-style.md` - Concise spoken responses, fillers, no formatting
- `personality/context.md` - Placeholder for future session context
- `push-to-talk.py` - LiveSession import, lifecycle methods, config watcher, on_press handler, voice commands, mic unmute on startup
- `indicator.py` - AI mode combo entry, LiveOverlayWidget class, overlay lifecycle polling, click-to-cycle modes

## Decisions Made
- Realtime API voices validated at init; unsupported voices (e.g. "nova") fall back to "ash"
- Personality prompt includes explicit "Always respond in English" to prevent language drift
- Audio ducking uses pactl to lower all non-aplay sink inputs to 15% while AI speaks
- Overlay communicates with session via signal files (live_mute_toggle) rather than shared memory
- Three-state cycle on overlay click: listening → muted → idle (disconnect) → listening (reconnect)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Voice validation for Realtime API**
- **Found during:** Checkpoint verification
- **Issue:** Config had `openai_voice: nova` which isn't supported by Realtime API, causing session error
- **Fix:** Added REALTIME_VOICES set and validation in __init__, falls back to "ash"
- **Files modified:** live_session.py

**2. [Rule 1 - Bug] Spanish language responses**
- **Found during:** Checkpoint verification
- **Issue:** AI responded in Spanish — personality prompt didn't specify language
- **Fix:** Added "Always respond in English" to personality/core.md
- **Files modified:** personality/core.md

**3. [Rule 2 - Missing Critical] Mic stuck muted after restart**
- **Found during:** Checkpoint verification
- **Issue:** Service restart during active session left mic muted (cleanup didn't run)
- **Fix:** Added mic unmute on startup in push-to-talk.py
- **Files modified:** push-to-talk.py

**4. [Rule 2 - Missing Critical] Audio ducking**
- **Found during:** User request during verification
- **Issue:** Music/other audio stayed at full volume while AI spoke
- **Fix:** Added _duck_other_audio/_unduck_other_audio using pactl sink-input volume control
- **Files modified:** live_session.py

**5. [Rule 2 - Missing Critical] Overlay click-to-cycle modes**
- **Found during:** User request during verification
- **Issue:** No way to mute/stop session from overlay UI
- **Fix:** Added three-state cycle (listening→muted→idle) with signal file IPC
- **Files modified:** indicator.py, live_session.py, push-to-talk.py

---

**Total deviations:** 5 (2 bugs, 3 missing critical)
**Impact on plan:** All fixes essential for usability. Audio ducking and overlay cycling were user-requested enhancements during verification.

## Issues Encountered
None — all issues discovered during verification were fixed inline.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Live voice session fully functional with personality, overlay, and audio ducking
- Phase 1 complete — ready for Phase 2 (Async Task Infrastructure)
- LiveSession's `tools: []` placeholder ready for Phase 3 tool definitions

---
*Phase: 01-mode-rename-and-live-voice-session*
*Completed: 2026-02-13*
