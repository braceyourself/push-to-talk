---
phase: 06-polish-verification
plan: 03
subsystem: overlay, status
tags: gtk, json-status, tool-intent, overlay-states

# Dependency graph
requires:
  - phase: 06-02
    provides: TOOL_INTENT_MAP, JSON-capable _set_status, acknowledgment clips
provides:
  - JSON status parsing in overlay
  - Dynamic tool intent labels during tool_use
  - History coalescing for consecutive tool entries
  - STT rejection flash (from Plan 01 integration)
affects:
  - None (final plan in phase)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "JSON status detection: startswith('{') → json.loads, else plain string"
    - "History coalescing: consecutive tool_use entries update in place"
    - "MCP tool name stripping: rsplit('__', 1)[-1] for intent lookup"

key-files:
  created: []
  modified:
    - indicator.py
    - live_session.py

key-decisions:
  - "Dropped nonverbal filler clips entirely — Piper TTS can't produce natural interjections"
  - "Acknowledgment phrases used for all filler roles (pre-response + pre-tool)"
  - "MCP tool name prefix stripped with rsplit for TOOL_INTENT_MAP lookup"
  - "Label truncation at 18 chars for overlay width constraint"
  - "Ack cancel event must be set on turn completion and post-tool text arrival"

patterns-established:
  - "JSON-or-string status protocol: overlay detects and parses both formats"
  - "Acknowledgment-only filler strategy: verbal phrases work, nonverbal sounds don't with Piper"

# Metrics
duration: ~15min (including iterative fixes during human verification)
completed: 2026-02-18
---

# Phase 6 Plan 3: Dynamic Tool Intent Overlay Summary

**JSON status parsing, dynamic tool intent labels, history coalescing, and iterative fixes from human verification**

## Performance

- **Duration:** ~15 min (including verification fixes)
- **Tasks:** 2 (1 auto + 1 human-verify)
- **Files modified:** 2 (indicator.py, live_session.py)

## Accomplishments
- Overlay parses JSON status metadata and displays dynamic tool intent labels
- History panel shows enriched entries with tool intent text instead of bare "tool_use"
- Consecutive tool_use entries coalesce into single evolving history entry
- STT rejection flash integrated (300ms dot dim from Plan 01)

## Task Commits

1. **Task 1: JSON status parsing and dynamic tool intent overlay** — `fb98552` (feat)
2. **Task 2: Human verification** — approved after iterative fixes

## Verification Fixes (discovered during human testing)

1. **MCP prefix not stripped** — `112182c` — Tool names came through as `mcp__ptt-task-tools__spawn_task`, TOOL_INTENT_MAP only had bare names. Fixed with `rsplit("__", 1)[-1]`.
2. **Nonverbal fillers spelled out** — `bdfa5a2`, `7b11d99`, `099c831` — Piper TTS spelled out "Hmm" as "H-M". Changed prompts multiple times but Piper fundamentally can't do interjections.
3. **Stuck tool_use state** — `83a18c8` — `_ack_cancel` event was never set after turn completion, causing rogue FILLER frames with no END_OF_TURN. Fixed by cancelling ack on turn end and post-tool text.
4. **Dropped nonverbal clips** — `b3238b4` — Removed nonverbal filler system entirely, acknowledgment phrases handle all filler roles.

## Issues Encountered
- Piper TTS (lessac-medium model) cannot produce natural nonverbal interjections — any consonant-heavy prompt gets spelled out letter-by-letter. This is a fundamental limitation of the model's phonemizer (espeak-ng). Acknowledged as motivation for TTS engine upgrade in v1.2.

## User Setup Required
None

## Next Phase Readiness
- Phase 6 complete — all v1.1 plans executed
- TTS engine upgrade (Piper → Kokoro) identified as v1.2 priority
---
*Phase: 06-polish-verification*
*Completed: 2026-02-18*
