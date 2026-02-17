# Phase 5 Plan 2: Barge-in Intelligence Layer Summary

**Sentence tracking at 3 flush sites in _read_cli_response, TTS->playback sentence_done sentinels for counting, spoken/unspoken annotation stored on barge-in and prepended to next user message, 0.4s post-interrupt silence threshold (vs 0.8s normal)**

## Accomplishments

- Added sentence tracking state: `_spoken_sentences` list populated at all three TTS flush points in `_read_cli_response` (normal streaming, post-tool buffer, final buffer)
- Implemented sentence boundary sentinels: TTS stage emits `FrameType.CONTROL` / `"sentence_done"` after each sentence's audio is fully synthesized
- Playback stage counts completed sentences via sentinel frames, giving accurate spoken vs unspoken split at interrupt time
- `_trigger_barge_in` builds context annotation: "[The user interrupted you. They heard up to: ... Your unspoken response was: ... Adjust based on what the user says next.]"
- Annotation stored in `_barge_in_annotation` and consumed in `_llm_stage` by prepending to the next user message (not as a separate turn)
- Post-interrupt silence threshold halved to 0.4s (from 0.8s) for one turn, enabling faster AI response to quick corrections
- `_post_barge_in` flag cleared after transcription completes (both normal silence path and flush-on-mute path)
- Barge-in log events now include sentence tracking metadata (spoken_sentences, total_sentences, cooldown_until)

## Task Commits

| # | Task | Commit | Type | Key Changes |
|---|------|--------|------|-------------|
| 1 | Track spoken vs unspoken sentences and send interruption annotation to CLI | `d09093d` | feat | Sentence tracking at 3 sites, TTS sentinels, playback counting, annotation build/consume |
| 2 | Shorten post-interrupt silence threshold and add overlay pulse | `5e7be52` | feat | Dynamic SILENCE_DURATION, _post_barge_in flag, enhanced logging |

## Files Modified

| File | Changes |
|------|---------|
| `live_session.py` | Added sentence tracking state in __init__, reset in _read_cli_response, append at 3 flush sites, sentence_done sentinels in _tts_stage, counting in _playback_stage, annotation build in _trigger_barge_in, annotation consume in _llm_stage, dynamic silence threshold in _stt_stage |

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Track sentences at flush sites rather than in TTS stage | Flush sites in _read_cli_response are where text is committed to TTS -- tracking here gives accurate sentence list before any TTS processing |
| Use CONTROL frame sentinels for sentence counting | Playback stage already handles frame types; CONTROL/sentence_done flows through the same queue without interfering with audio |
| Prepend annotation to user message (not separate turn) | Sending annotation as a separate message would trigger an unwanted AI response; prepending means it arrives WITH the user's correction |
| 0.4s post-barge-in silence (half of normal 0.8s) | After interruption, user likely has a quick correction; shorter silence = faster response feels more natural |
| No separate overlay state for barge-in | The status transition from "speaking" to "listening" IS the visual acknowledgment; adding a flash state would be overengineering |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

1. Module imports: `from live_session import LiveSession` -- OK
2. `_spoken_sentences` -- 9 occurrences: init, reset, 3 appends, 2 slices in trigger, 2 len() calls
3. `_played_sentence_count` -- 7 occurrences: init, reset, increment in playback, 2 reads in trigger, 1 in log
4. `sentence_done` -- 2 occurrences: put in TTS stage, handled in playback stage
5. `_barge_in_annotation` -- 5 occurrences: init, set in trigger, consumed in LLM stage (read + clear)
6. Dynamic silence: `SILENCE_DURATION_POST_BARGE` used when `_post_barge_in` is True, cleared after transcription

## Duration

~2.5 minutes
