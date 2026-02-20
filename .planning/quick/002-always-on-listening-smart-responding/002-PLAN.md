---
phase: quick
plan: 002
type: execute
wave: 1
depends_on: []
files_modified:
  - live_session.py
  - push-to-talk.py
  - indicator.py
  - test_live_session.py
autonomous: true

must_haves:
  truths:
    - "Live session stays connected indefinitely when idle timeout is 0"
    - "Live session still disconnects after configured seconds when timeout > 0"
    - "Config option live_idle_timeout controls the behavior"
  artifacts:
    - path: "live_session.py"
      provides: "Configurable idle timeout via constructor param"
      contains: "idle_timeout"
    - path: "push-to-talk.py"
      provides: "Config plumbing from load_config to LiveSession"
      contains: "live_idle_timeout"
    - path: "test_live_session.py"
      provides: "Tests for idle timeout disable and custom values"
  key_links:
    - from: "push-to-talk.py"
      to: "live_session.py"
      via: "idle_timeout constructor param"
      pattern: "idle_timeout="
---

<objective>
Make the live session idle timeout configurable and allow disabling it entirely (value 0) so "always-on" listening works without the session disconnecting after 2 minutes of silence.

Purpose: The always-on listening feature (auto_start_listening) is useless if the session kills itself after 120s of idle. Users need the session to stay alive indefinitely.
Output: Configurable `live_idle_timeout` with 0 = never timeout.
</objective>

<execution_context>
@/home/ethan/.claude/get-shit-done/workflows/execute-plan.md
@/home/ethan/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@live_session.py (lines 179-212: __init__, line 766-783: idle timer methods)
@push-to-talk.py (lines 236-281: load_config, lines 906-955: start_live_session)
@indicator.py (lines 53-75: load_config defaults)
@test_live_session.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write tests for configurable idle timeout</name>
  <files>test_live_session.py</files>
  <action>
Add tests to `test_live_session.py` following TDD — write these FIRST before any production code changes:

1. **Test default idle timeout is 0 (always-on)**:
   - Create a `LiveSession` with no `idle_timeout` param
   - Assert `session._idle_timeout == 0`
   - This changes the default from 120 to 0 to match the always-on philosophy

2. **Test custom idle timeout value**:
   - Create a `LiveSession` with `idle_timeout=300`
   - Assert `session._idle_timeout == 300`

3. **Test idle timer not scheduled when timeout is 0**:
   - Create a `LiveSession` with `idle_timeout=0`
   - Call `session._reset_idle_timer()` inside a running event loop
   - Assert `session._idle_timer is None` (timer was never created)

4. **Test idle timer IS scheduled when timeout > 0**:
   - Create a `LiveSession` with `idle_timeout=60`
   - Call `session._reset_idle_timer()` inside a running event loop
   - Assert `session._idle_timer is not None`
   - Clean up by cancelling the timer

Run: `python3 test_live_session.py` — tests should FAIL (the constructor doesn't accept idle_timeout yet).
  </action>
  <verify>python3 /home/ethan/code/push-to-talk/test_live_session.py 2>&1 | tail -5</verify>
  <done>Tests exist and fail with TypeError (unexpected keyword argument 'idle_timeout')</done>
</task>

<task type="auto">
  <name>Task 2: Implement configurable idle timeout</name>
  <files>live_session.py, push-to-talk.py, indicator.py</files>
  <action>
Make these changes to pass the tests and wire up the config:

**live_session.py:**
1. Add `idle_timeout=0` parameter to `LiveSession.__init__` (line ~182). Change the default from hardcoded 120 to the param value:
   ```python
   def __init__(self, openai_api_key=None, deepgram_api_key=None,
                voice="ash", model="claude-sonnet-4-5-20250929", on_status=None,
                fillers_enabled=True, barge_in_enabled=True, whisper_model=None,
                idle_timeout=0):
   ```
   Then on line ~212: `self._idle_timeout = idle_timeout`

2. In `_reset_idle_timer` (line ~766), add an early return when timeout is 0:
   ```python
   def _reset_idle_timer(self):
       self._cancel_idle_timer()
       if self._idle_timeout <= 0:
           return
       try:
           loop = asyncio.get_event_loop()
           if loop.is_running():
               self._idle_timer = loop.call_later(self._idle_timeout, self._on_idle_timeout)
       except RuntimeError:
           pass
   ```

**push-to-talk.py:**
1. Add `"live_idle_timeout": 0` to the `default` dict in `load_config()` (line ~261, after `live_auto_mute`). Default 0 means always-on by default — matches the new always-on philosophy.
2. In `start_live_session()` (line ~929), read the config and pass to constructor:
   ```python
   idle_timeout = self.config.get('live_idle_timeout', 0)
   self.live_session = LiveSession(
       ...,
       idle_timeout=idle_timeout,
   )
   ```

**indicator.py:**
1. Add `"live_idle_timeout": 0` to the `default` dict in `load_config()` (line ~74, after `live_auto_mute`). Keep defaults in sync.

No UI control needed for this setting — it's a config.json-only option. Users who want the old 120s behavior can set `"live_idle_timeout": 120` in config.json.

Run tests: `python3 test_live_session.py` — all tests should pass.
  </action>
  <verify>python3 /home/ethan/code/push-to-talk/test_live_session.py 2>&1 | tail -5</verify>
  <done>All tests pass including the new idle timeout tests. Default is 0 (never timeout). Custom values work. Timer skipped when 0.</done>
</task>

</tasks>

<verification>
1. `python3 /home/ethan/code/push-to-talk/test_live_session.py` — all tests pass (including the 27 existing + new idle timeout tests)
2. `grep -n 'idle_timeout' /home/ethan/code/push-to-talk/live_session.py` — shows param in __init__ and guard in _reset_idle_timer
3. `grep -n 'live_idle_timeout' /home/ethan/code/push-to-talk/push-to-talk.py` — shows config default and plumbing to LiveSession
</verification>

<success_criteria>
- LiveSession accepts idle_timeout param, defaults to 0
- idle_timeout=0 means no timer is ever scheduled (session lives forever)
- idle_timeout>0 works exactly as before (timer fires, session disconnects)
- Config option live_idle_timeout plumbed from config.json through push-to-talk.py to LiveSession
- All existing tests still pass
- New tests cover: default value, custom value, timer skip on 0, timer created on >0
</success_criteria>

<output>
After completion, create `.planning/quick/002-always-on-listening-smart-responding/002-SUMMARY.md`
</output>
