---
phase: quick
plan: 003
subsystem: voice-pipeline
tags: [haiku, backchannel, tts, piper, anthropic-api, filler]
dependency-graph:
  requires: []
  provides: [haiku-backchannel, contextual-filler-responses]
  affects: [live-session-filler-system]
tech-stack:
  added: [anthropic-sdk]
  patterns: [lazy-singleton-client, async-timeout-race, graceful-fallback]
key-files:
  created:
    - backchannel.py
  modified:
    - live_session.py
    - requirements.txt
    - test_live_session.py
decisions:
  - id: bc-001
    title: "800ms combined budget for Haiku + Piper"
    choice: "Single 800ms asyncio.wait_for wrapping both API call and TTS synthesis"
    reason: "Keeps worst-case filler latency under 1.3s (800ms Haiku + 500ms LLM gate)"
  - id: bc-002
    title: "Haiku runs before clip lookup, not in parallel"
    choice: "Sequential: Haiku first, clip lookup only if Haiku fails"
    reason: "Avoids wasting response library lookups when Haiku succeeds; clip lookup is cheap as fallback"
  - id: bc-003
    title: "Module-level lazy singleton for Anthropic client"
    choice: "_get_client() creates AsyncAnthropic on first call"
    reason: "Avoids import-time side effects and API key validation; matches clip_factory pattern"
metrics:
  duration: "6m"
  completed: "2026-03-09"
---

# Quick Task 003: Haiku-Driven Backchannel Responses Summary

**One-liner:** Haiku API generates context-aware 1-5 word backchannel responses spoken via Piper TTS, with 800ms timeout falling back to pre-recorded clips.

## What Was Done

### Task 1: Create backchannel.py module with Haiku API integration
- Created self-contained `backchannel.py` module with no live_session.py imports
- `generate_backchannel()`: Calls Claude Haiku (claude-haiku-4-5-20251001) with a compact system prompt for ultra-short responses
- `generate_backchannel_tts()`: Chains API call with Piper TTS subprocess (22050Hz raw PCM output)
- Lazy-initialized AsyncAnthropic singleton client (reads ANTHROPIC_API_KEY from env)
- Optional last_assistant_text context (last 100 chars) prepended to user message
- Added `anthropic` to requirements.txt
- **Commit:** 391a427

### Task 2: Wire Haiku backchannel into _filler_manager with 800ms race
- Added Haiku race as Step 4 in _filler_manager (800ms combined timeout for API + TTS)
- Clip lookup only happens if Haiku fails/times out (Step 5 fallback)
- 500ms LLM gate preserved after Haiku race (Step 7)
- Backchannel text logged to conversation history with `backchannel=True` flag
- `_last_assistant_text` tracked in `_read_cli_response` for context
- Entire existing clip fallback path preserved as safety net (Steps 9-10)
- **Commit:** f0c7214

### Task 3: Tests (TDD -- written before implementation)
- 5 tests covering all backchannel paths:
  1. `test_generate_backchannel_returns_short_text` -- API success returns 1-5 words
  2. `test_generate_backchannel_returns_none_on_error` -- API error returns None
  3. `test_generate_backchannel_tts_returns_tuple` -- API + Piper success returns (text, bytes)
  4. `test_generate_backchannel_tts_returns_none_on_tts_failure` -- Piper failure returns None
  5. `test_backchannel_includes_context_when_provided` -- last_assistant_text appears in API call
- **Committed with Task 1:** 391a427

## Decisions Made

| ID | Decision | Choice | Reason |
|----|----------|--------|--------|
| bc-001 | Haiku + Piper timeout | 800ms combined | Keeps worst-case under 1.3s with 500ms LLM gate |
| bc-002 | Haiku before clip lookup | Sequential, not parallel | Avoids wasted lookups; clip is cheap fallback |
| bc-003 | Client initialization | Lazy singleton via _get_client() | No import-time side effects |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

1. `pip install anthropic` -- anthropic 0.84.0 installed
2. `from backchannel import generate_backchannel_tts` -- imports cleanly
3. `import live_session` -- imports cleanly (no circular imports)
4. Backchannel tests -- all 5 pass
5. Full test suite -- 176/176 pass (no regressions)

## Architecture

```
User speaks -> _filler_manager
  |-> classify input (existing)
  |-> trivial? return (existing)
  |-> Race Haiku backchannel (800ms timeout)  <-- NEW
  |     |-> generate_backchannel() via Anthropic API
  |     |-> generate_backchannel_tts() via Piper subprocess
  |-> If Haiku fails: lookup pre-recorded clip (existing fallback)
  |-> 500ms LLM gate (existing)
  |-> Play backchannel audio OR clip via composer
```

## Commit Log

| Hash | Message |
|------|---------|
| 391a427 | feat(quick-003): add Haiku-driven backchannel module with tests |
| f0c7214 | feat(quick-003): wire Haiku backchannel into _filler_manager with 800ms race |
