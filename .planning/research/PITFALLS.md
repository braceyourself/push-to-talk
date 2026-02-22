# Domain Pitfalls: Deepgram Streaming STT + Local Decision Model

**Domain:** Adding streaming cloud STT and local decision model to existing PTT voice assistant
**Researched:** 2026-02-22
**Confidence:** HIGH (Deepgram official docs, SDK issues, community discussions, codebase inspection)

**System context:** Existing push-to-talk voice assistant with:
- RTX 3070 8GB VRAM (Whisper being removed, freeing ~3GB)
- PipeWire echo cancellation already configured
- TranscriptBuffer ring buffer already built
- Silero VAD already running for barge-in detection
- Existing PTT mode must still work alongside always-on
- Single user, single room, Linux desktop
- Audio capture at 24kHz 16-bit mono via pasimple/PipeWire

---

## Critical Pitfalls

Mistakes that cause the feature to not work, create runaway costs, or require architectural rework.

---

### Pitfall 1: VAD Gating Breaks Deepgram Endpointing and Reduces Accuracy

**What goes wrong:**
The v2.0 architecture plans to use Silero VAD as a "cost gate" -- only streaming speech segments to Deepgram. This sounds logical (why pay to transcribe silence?) but Deepgram's own engineering team explicitly recommends against it. In a GitHub discussion (#1216), a Deepgram team member stated: "I would not recommend that approach for the vast majority of use cases."

The problems are threefold:
1. **Endpointing breaks.** Deepgram's endpointing feature (`speech_final: true`) uses audio-based VAD to detect when speech transitions to silence. If you pre-filter silence out with your own VAD, Deepgram never sees the silence gap, so `speech_final` never fires. Your downstream logic that waits for `speech_final: true` to process a complete utterance hangs indefinitely.
2. **Accuracy degrades.** Silence is context for the STT model. Removing it changes the acoustic properties of the audio stream, reducing Word Error Rate performance. The Deepgram team stated: "Silence is critical context for the model."
3. **Latency increases.** Pre-filtering introduces "finalization latency" because the model receives choppy audio fragments instead of a continuous stream.

**Why it matters for THIS project:**
The PROJECT.md states: "Local VAD cost gate (Silero) -- only stream speech segments to Deepgram (saves API cost)." This is the planned architecture, and it directly contradicts Deepgram's official recommendation.

**How to avoid:**

1. **Stream continuously, use KeepAlive for true silence.** When there is genuinely no audio activity (room is empty, user left), send KeepAlive messages instead of audio. KeepAlive messages are free -- they don't count toward billing. Send them every 3-5 seconds to prevent the 10-second timeout (NET-0001 error). When speech resumes, switch back to streaming audio. This gives Deepgram the silence context it needs while not paying for extended idle periods.

2. **Use Silero VAD for connection lifecycle, not audio filtering.** VAD detects "room is active" vs "room is dead quiet." When VAD shows no speech for 30+ seconds, stop streaming audio and switch to KeepAlive. When VAD detects any activity, resume streaming audio (including the silence between words). This preserves Deepgram's endpointing while avoiding billing for extended idle.

3. **Accept the cost of continuous streaming during active periods.** At $0.0077/min ($0.462/hr), 8 hours of continuous streaming is ~$3.70/day. But you won't be speaking for 8 hours straight. Realistic usage: 2-4 hours of active audio per day = $0.92-$1.85/day. With the KeepAlive approach during idle, actual costs will be lower.

4. **If cost is unacceptable, consider a hybrid approach.** Stream audio continuously during active conversation (preserving endpointing), but disconnect the WebSocket entirely during extended idle (> 5 minutes of silence). Reconnect when VAD detects speech. Accept the 1-2 second reconnection latency as a tradeoff for cost savings during idle hours.

**Warning signs:**
- `speech_final: true` never fires
- Transcripts arrive but endpointing doesn't work -- you have to manually detect silence
- Transcript accuracy is noticeably worse than Deepgram demos
- Endpointing latency is erratic or extremely high

**Detection:** Compare WER and endpointing behavior with VAD-gated vs continuous streaming using the same audio. If endpointing fails with VAD gating, the architecture must change.

**Phase to address:** Phase 1 (Deepgram integration). This is an architectural decision that must be made before writing the streaming code. Getting this wrong means reworking the entire audio pipeline.

**Severity:** CRITICAL

---

### Pitfall 2: Audio Format Mismatch -- 24kHz Capture vs Deepgram Expectations

**What goes wrong:**
The existing codebase captures audio at 24000 Hz, 16-bit mono PCM via pasimple (`SAMPLE_RATE = 24000` in both `live_session.py` and `continuous_stt.py`). Deepgram's documentation and examples overwhelmingly use 16000 Hz. While Deepgram supports 24kHz (their docs list 8000, 16000, 24000, 44100, 48000 Hz), the format parameters MUST be explicitly declared when streaming raw audio: `encoding=linear16` and `sample_rate=24000`. If you omit these or set them wrong, Deepgram receives raw bytes it can't decode and returns empty transcripts with zero error messages.

The failure mode is silent: Deepgram accepts the connection, receives the audio, but produces no transcripts. No error, no warning, just silence. This is documented in their troubleshooting: "Deepgram will be unable to decode the audio and will fail to return a transcript."

**Why it matters for THIS project:**
The Silero VAD in `continuous_stt.py` resamples from 24kHz to 16kHz internally (lines 183-186: "take 2 of every 3 samples"). If someone assumes Deepgram also needs 16kHz and resamples the audio, or forgets to specify 24kHz in the connection parameters, the audio format becomes garbled. Additionally, the existing Whisper pipeline uses 24kHz directly -- switching to Deepgram might tempt a developer to "normalize" to 16kHz thinking it's required.

**How to avoid:**

1. **Explicitly set encoding and sample_rate in every Deepgram connection.**
   ```python
   options = LiveOptions(
       model="nova-3",
       encoding="linear16",
       sample_rate=24000,
       channels=1,
       language="en",
   )
   ```
   Never rely on Deepgram auto-detecting format for raw audio streams.

2. **Keep the existing 24kHz capture rate.** Don't resample. Deepgram supports 24kHz natively. Resampling adds latency, complexity, and potential artifacts. The only component that needs resampling is Silero VAD (which already handles its own 24kHz->16kHz conversion).

3. **Add a startup validation.** When connecting to Deepgram, send a known test phrase within the first few seconds and verify you receive a transcript. If no transcript arrives within 5 seconds of clearly spoken audio, the format parameters are probably wrong.

4. **Watch for the "container vs raw" distinction.** If you ever wrap audio in a WAV container before sending, do NOT set encoding/sample_rate -- Deepgram reads the WAV header. But if sending raw PCM bytes (which is what pasimple produces), you MUST set both parameters. Mixing these up is a common source of empty transcripts.

**Warning signs:**
- Connection opens successfully but no transcripts arrive
- Transcripts arrive but are garbled or nonsensical
- The same audio works fine with Whisper but produces nothing from Deepgram

**Detection:** Log the first transcript received after connection. If no transcript arrives within 10 seconds of streaming audio that contains speech, check format parameters immediately.

**Phase to address:** Phase 1 (Deepgram integration). This is literally the first thing to get right -- nothing else works without correct audio format.

**Severity:** CRITICAL

---

### Pitfall 3: WebSocket Disconnection Drops Mid-Utterance Transcript

**What goes wrong:**
Deepgram uses a persistent WebSocket connection. Network blips, Deepgram server maintenance, WiFi momentary drops, or the 10-second inactivity timeout can kill the connection. If a disconnection happens mid-utterance (user is speaking), the partial transcript from that utterance is lost entirely. Each new WebSocket connection starts a fresh transcription session with no memory of the previous one. Timestamps reset to 00:00:00.

For a push-to-talk system, this is annoying but tolerable (user can re-speak). For an always-on system, it's worse: the user doesn't know the system disconnected, continues speaking, and their words vanish into the void. The decision model never sees the transcript, so the AI never responds to what was said.

**Why it matters for THIS project:**
The existing system has Whisper running locally -- there's no network dependency for STT. Moving to Deepgram introduces a cloud dependency in the critical path. The `CircuitBreaker` class (line 154 in `live_session.py`) already exists for service fallback, but it's designed for per-request failures, not persistent WebSocket lifecycle management.

**How to avoid:**

1. **Buffer audio locally during disconnection.** When the WebSocket drops, continue capturing audio into a local ring buffer (the existing `bytearray` pattern from `_stt_stage`). When the connection is re-established, replay the buffered audio before resuming live streaming. Caveat: Deepgram processes audio at max 1.25x realtime, so a 30-second buffer takes 24 seconds to replay. Keep the buffer short (10-15 seconds max) and accept some data loss for longer outages.

2. **Implement exponential backoff with jitter for reconnection.** Start at 1 second, double up to 30 seconds max. Add random jitter (0-500ms) to prevent reconnection storms if multiple clients are affected by a Deepgram outage. The SDK v3+ has built-in reconnection support, but verify it uses backoff -- naive immediate reconnection can hit rate limits.

3. **Track connection state explicitly.** Maintain a state machine: CONNECTED, CONNECTING, DISCONNECTED, RECONNECTING. The decision model should know the STT state: if disconnected, it should not penalize the "silence" (the AI isn't hearing anything, not that no one is speaking). Update the dashboard/event bus with connection state changes.

4. **Use the Finalize message before intentional disconnection.** When gracefully closing the WebSocket (session end, mode switch), send a `{"type": "Finalize"}` message and wait for the last transcript before closing. Skipping this drops the final words of the last utterance.

5. **Realign timestamps after reconnection.** Each reconnection resets Deepgram timestamps to 00:00:00. Maintain a local timestamp offset that gets updated on each reconnection. Add the offset to all Deepgram-returned timestamps so the TranscriptBuffer maintains chronological ordering.

**Warning signs:**
- Gaps in the transcript timeline (missing segments between disconnect and reconnect)
- User says something and AI doesn't respond, but moments later a similar statement gets a response
- Dashboard shows connection state toggling rapidly (reconnection storm)
- TranscriptBuffer timestamps have discontinuities or jumps backward

**Detection:** Log every WebSocket open/close event with timestamp. Count disconnections per hour. Healthy: 0-1 disconnections per 8 hours. Concerning: > 3 per hour. Critical: > 1 per minute (indicates a systemic issue).

**Phase to address:** Phase 1 (Deepgram integration). Reconnection logic is not optional for a production streaming client.

**Severity:** CRITICAL

---

### Pitfall 4: Cost Runaway -- Streaming 24/7 Without Lifecycle Management

**What goes wrong:**
Deepgram bills per second of audio processed. At $0.0077/min, continuous 24/7 streaming costs:
- 1 hour: $0.46
- 8 hours (workday): $3.70
- 24 hours: $11.09
- 30 days: $332.64

If the user starts the always-on session in the morning and forgets to stop it before bed, it streams (and bills) all night. If KeepAlive is misconfigured and the system streams silence instead, the full audio duration is billed. If the reconnection logic creates multiple overlapping connections, each connection bills independently.

The $200 free credit goes quickly. At 8 hours/day, it lasts ~54 days. At 24/7 continuous, ~18 days.

**Why it matters for THIS project:**
The project philosophy is "always-on" -- the session is designed to run indefinitely (`idle_timeout=0`). There's no natural stopping point that would limit costs. The user might leave the system running while sleeping, on vacation, or during extended periods away from the desk.

**How to avoid:**

1. **Implement an activity-based connection lifecycle.** Don't keep the Deepgram WebSocket open 24/7. Use Silero VAD (which runs locally, for free) to detect room activity. Strategy:
   - **Active mode:** VAD detects speech in the last 60 seconds -> WebSocket open, streaming audio
   - **Idle mode:** No speech for 60+ seconds -> Send KeepAlive every 5 seconds (free)
   - **Sleep mode:** No speech for 10+ minutes -> Close WebSocket entirely (zero cost)
   - **Wake:** VAD detects speech -> Open new WebSocket, buffer and replay

   This turns "$332/month always-on" into "$15-30/month active hours only."

2. **Add a cost tracking counter.** Track cumulative seconds of audio streamed per day/week/month. Display in the dashboard. Alert the user when approaching budget thresholds (e.g., 50% of $200 credit, $5/day, etc.).

3. **Configurable daily budget cap.** Add a config option `deepgram_daily_budget_cents: 200` (default $2/day). When the daily budget is hit, fall back to local Whisper (or disable STT and show a notification). The user can override but must do so consciously.

4. **Prevent connection leaks.** Every WebSocket open must have a corresponding close. Use try/finally or context managers. If the process crashes, orphaned WebSocket connections on Deepgram's side will timeout after 10 seconds (the KeepAlive timeout), but if the process is somehow sending data from a zombie thread, it could bill indefinitely. Use the `CircuitBreaker` pattern to detect and kill connections that aren't producing transcripts.

5. **Log billing events.** Every time audio is sent to Deepgram, log the duration. Every time KeepAlive is sent, log it distinctly. This makes it easy to audit "why was my bill $X?"

**Warning signs:**
- Deepgram dashboard shows hours of usage you don't remember
- Free credit depleting faster than expected
- Multiple WebSocket connections open simultaneously (check with connection state tracking)
- Audio streaming during periods when user is clearly away (late night, weekends)

**Detection:** Compare daily audio-sent duration to expected active hours. If streamed hours >> active hours, the lifecycle management is broken.

**Phase to address:** Phase 1 (connection lifecycle). Cost management must be designed into the architecture, not added as an afterthought.

**Severity:** CRITICAL

---

### Pitfall 5: Echo Cancellation Path -- AI Speech Reaches Deepgram and Creates Feedback Loop

**What goes wrong:**
PipeWire echo cancellation is already configured (`~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`). But the echo cancellation path changes when switching from local Whisper to cloud Deepgram:

- **Before (Whisper):** Mic -> PipeWire AEC -> pasimple capture -> local Whisper -> transcript. AEC removes AI speech from the mic signal before Whisper processes it. Everything is local, latency is minimal.
- **After (Deepgram):** Mic -> PipeWire AEC -> pasimple capture -> WebSocket -> Deepgram cloud -> transcript. The AEC-processed audio goes to the cloud. But AEC is never perfect -- residual echo (attenuated AI speech) still exists in the signal. Locally, Whisper's hallucination filter could catch "echo-like" transcripts. Deepgram processes whatever it receives and returns transcripts for it.

If AEC lets through even partial AI speech, Deepgram transcribes it. The decision model sees the AI's own words in the transcript stream. If the decision model isn't aware that these words are echo, it may trigger a response -- creating the exact feedback loop documented in the original PITFALLS.md (Pitfall 1).

The additional wrinkle: with cloud STT, there's network latency between the AI speaking and the echo transcript arriving. The existing `_stt_gated` mechanism (which suppresses STT during playback) relies on tight timing between playback start/stop and STT processing. With 100-200ms network latency, the timing windows are wider and harder to manage.

**How to avoid:**

1. **Keep the transcript fingerprinting defense.** The original PITFALLS.md recommended comparing STT output against recent AI speech. This is even more important with cloud STT because you can't rely on AEC being perfect over a network path. Maintain a ring buffer of the last N sentences the AI spoke and fuzzy-match incoming Deepgram transcripts against it.

2. **Tag transcripts with timing metadata.** When the AI is speaking, tag any incoming Deepgram transcripts with `during_ai_playback=True`. The decision model should heavily discount or ignore these transcripts. Use the existing `_playing_audio` / `_playback_end_time` state from `continuous_stt.py` (lines 67-69) but account for network latency: extend the "during playback" window by 200-500ms after playback ends.

3. **Don't rely solely on PipeWire AEC.** PipeWire's WebRTC AEC works well for near-field microphone/speaker setups but degrades with room reflections, speaker distance, and volume. Test with actual speaker output at normal conversation volume. If echo transcripts appear, increase the post-playback cooldown window.

4. **Consider using Deepgram's own endpointing behavior as an echo signal.** If `speech_final: true` fires during AI playback, it's almost certainly echo (the user is unlikely to be speaking a complete sentence while the AI is talking). Only treat barge-in-style interruptions (detected via Silero VAD) as genuine user speech during playback.

**Warning signs:**
- Transcript stream contains text matching what the AI just said
- Decision model triggers responses during AI speech
- Rapid back-and-forth conversation where the AI appears to be talking to itself
- `during_ai_playback` tagged transcripts are non-empty

**Detection:** Log all transcripts received during AI playback. Calculate echo_match_ratio = fuzzy matches against recent AI speech / total transcripts during playback. Target: < 5%. If higher, AEC is insufficient.

**Phase to address:** Phase 1 (audio pipeline). Must be validated before connecting Deepgram output to the decision model.

**Severity:** CRITICAL

---

## Major Pitfalls

Mistakes that cause significant UX degradation or rework.

---

### Pitfall 6: Endpointing Fails in Noisy Environments -- `speech_final` Never Fires

**What goes wrong:**
Deepgram's endpointing feature uses audio-based VAD to detect when speech transitions to silence. In environments with background noise (HVAC, fan, music, typing, etc.), the background noise prevents the VAD from detecting silence. The `speech_final: true` flag never fires. Deepgram's documentation explicitly warns: "Background noise may prevent the `speech_final=true` flag from being sent."

Without `speech_final`, the downstream logic never knows when the user finished speaking. Transcripts accumulate but are never processed as complete utterances. The decision model either processes every interim result (too chatty, high latency) or waits forever for a finalization that never comes.

This is NOT a Deepgram bug -- it's the inherent limitation of audio-based endpoint detection in noisy environments. The existing Whisper pipeline avoids this because it uses local RMS-based silence detection (`SILENCE_THRESHOLD = 150`) which is calibrated to the specific room's ambient noise level.

**How to avoid:**

1. **Use `utterance_end_ms` alongside endpointing.** Deepgram's UtteranceEnd feature analyzes word timing gaps in transcription results rather than raw audio. It fires when no new words appear for N milliseconds, regardless of background noise. Configure: `utterance_end_ms=1000` (1 second gap between words). This requires `interim_results=true` to be enabled.

2. **Implement dual-trigger end-of-speech detection.** Process a complete utterance when EITHER:
   - `speech_final: true` is received (endpointing detected silence), OR
   - `UtteranceEnd` message is received with no preceding `speech_final: true`

   This is Deepgram's officially recommended pattern for noisy environments.

3. **Add a local safety timeout.** If neither `speech_final` nor `UtteranceEnd` fires within 15 seconds of the last `is_final: true` transcript, force-process the accumulated transcript. This prevents indefinite waiting in edge cases where both Deepgram mechanisms fail.

4. **Configure endpointing for conversational use.** The default endpointing timeout is 10ms, which is extremely aggressive (designed for chatbots expecting short utterances). For a conversational always-on assistant, use `endpointing=300` (300ms) to allow natural pauses within sentences without premature finalization.

**Warning signs:**
- Transcripts arrive (interim and final) but `speech_final: true` never appears
- The system seems to "buffer" everything but never acts on it
- Works perfectly in a quiet room, fails when music/TV/fan is on

**Detection:** Log every `speech_final` and `UtteranceEnd` event. If `speech_final` count is zero over a 10-minute active conversation, endpointing is broken by noise.

**Phase to address:** Phase 1 (Deepgram integration). End-of-speech detection must work reliably before connecting to the decision model.

**Severity:** MAJOR

---

### Pitfall 7: Decision Model False Positives -- AI Responds to TV, Music, Phone Calls

**What goes wrong:**
With Deepgram's superior accuracy (Nova-3: 5-7% WER vs Whisper's 10%+), the transcript stream will be more accurate but also more comprehensive. It faithfully transcribes everything: TV dialogue, podcast audio, phone calls on speaker, other people in the room, music with lyrics. The decision model receives a stream of perfectly transcribed speech from multiple sources and must decide which is the user addressing the AI.

Without speaker identification, every question in a TV show becomes a potential trigger. "What time does the flight leave?" from a movie dialogue looks identical to the user asking "What time does the flight leave?" in the transcript. The decision model has no acoustic features to distinguish them -- only text.

**Why it's worse with cloud STT:**
Whisper's lower accuracy actually provided a crude filter: garbled or partial transcripts from TV audio were often rejected by the hallucination filter. Deepgram's higher accuracy means TV dialogue comes through clean and clear, making false positives MORE likely, not less.

**How to avoid:**

1. **Default to name-activation only for v2.0 launch.** Only respond when "Russel" (or variants) appears in the transcript. This eliminates virtually all false positives from TV/music/phone. Expand to proactive participation only after the name-activation path is proven reliable.

2. **Use Deepgram's keyterm prompting for name recognition.** Nova-3 supports `keyterm=["Russel", "Russell", "Hey Russel"]` which boosts recognition of these specific terms. This improves name detection accuracy significantly -- Deepgram claims up to 90% Keyword Recall Rate improvement with keyterm prompting.

3. **Use energy/proximity heuristics from the local audio stream.** The audio captured by pasimple includes energy information. Speech from speakers (TV, laptop, phone) typically has different energy profiles than speech from a person sitting at the desk:
   - Direct speech: higher RMS, more dynamic range
   - Speaker playback: more compressed, flatter RMS profile
   - This is a rough heuristic but can provide a signal to the decision model

4. **Implement a "conversation mode" state machine.** After the user addresses the AI by name, enter a "conversation active" state with a lower response threshold for 2-3 minutes. After the conversation ends (no user-directed speech for 3 minutes), return to "passive monitoring" with name-only activation. This avoids the AI randomly interjecting after long idle periods.

5. **Cooldown after proactive responses.** If the AI responds proactively (not name-triggered) and the user doesn't engage within 5 seconds, suppress proactive responses for 5 minutes. This prevents repeated false positives from ongoing TV/music.

**Warning signs:**
- AI responds during movie watching or podcast listening
- AI interjects into phone calls on speaker
- User frequently says "I wasn't talking to you" or hits mute
- Response rate during passive mode exceeds 2 per hour without name triggers

**Detection:** Track response triggers: name-activated vs proactive vs ambient. If proactive/ambient responses have a > 30% "user didn't engage" rate, the threshold is too aggressive.

**Phase to address:** Phase 2 (decision model). Conservative defaults first, loosen with data.

**Severity:** MAJOR

---

### Pitfall 8: Decision Model False Negatives -- Name Detection Failures

**What goes wrong:**
The user says "Hey Russel, what's the weather?" and nothing happens. Name detection fails because:
1. Deepgram transcribes "Russel" as "Russell", "Wrestle", "Rustle", "Brussel", "Muscle"
2. The name appears in an interim result that gets superseded before the final result arrives
3. Background noise garbles the name while the rest of the sentence is clear
4. The user says just "Russel" (no sentence context) and it's too short for reliable transcription
5. The name is at the beginning of an utterance where STT accuracy is lowest

Unlike Whisper (where you can bias transcription with `initial_prompt`), Deepgram uses a different mechanism: keyterm prompting. But keyterms only work with Nova-3 and Flux models, and they boost probability -- they don't guarantee recognition.

**How to avoid:**

1. **Use Deepgram keyterm prompting.** Add `keyterm=["Russel", "Russell", "Hey Russel"]` to the connection options. This significantly boosts the probability of correct transcription for these specific terms.

2. **Fuzzy match the name in transcripts.** Don't require exact string match. Use a variant list and Levenshtein distance:
   ```python
   NAME_VARIANTS = {"russel", "russell", "rusel", "russ", "hey russel",
                     "hey russell", "wrestle", "rustle"}  # Common misrecognitions

   def detect_name(transcript: str) -> bool:
       lower = transcript.lower()
       for variant in NAME_VARIANTS:
           if variant in lower:
               return True
       # Phonetic fallback: check for words starting with "russ" or "rus"
       for word in lower.split():
           if word.startswith("russ") or word.startswith("rus"):
               return True
       return False
   ```

3. **Check BOTH interim and final results for name detection.** The name might appear in an interim result but get corrected away in the final. For name detection specifically, treat interim results as valid triggers -- even if the final transcript changes the word, the user probably said the name. Don't wait for `is_final: true` for name checking.

4. **Keep Silero VAD as a parallel always-running name detector.** If the local decision model can run on the raw audio features (not just text), it can detect "someone said something that sounded like the wake word" independently of Deepgram's transcription. This provides a fast (<100ms) local signal that can be cross-referenced with the slower (~200ms) Deepgram transcript.

5. **Acknowledge immediately on name detection.** When the name is detected (even in interim results), play a brief acknowledgment sound or say "yes?" immediately. This gives the user feedback that the system heard them, even before the full transcript is processed. This dramatically reduces the perceived "AI didn't hear me" frustration.

**Warning signs:**
- User says "Russel" multiple times before getting a response
- Transcript logs show the name consistently misrecognized as another word
- Name detection works in quiet rooms but fails with background noise
- Users start over-enunciating the name (compensating for poor detection)

**Detection:** Log every transcript containing words phonetically similar to "Russel." Compare true positive rate (name said and detected) vs false negative rate (name said but not detected). Target: > 95% true positive rate.

**Phase to address:** Phase 2 (name detection). One of the first features to test with real speech.

**Severity:** MAJOR

---

### Pitfall 9: Interim Result Handling -- Processing Too Early or Too Late

**What goes wrong:**
Deepgram streaming returns three types of results:
1. **Interim results** (`is_final: false`): Preliminary, will change as more audio arrives
2. **Final results** (`is_final: true`): Text is stable for this segment, but utterance may continue
3. **Speech final** (`speech_final: true`): The speaker has paused, utterance is complete

The naive approach -- processing every interim result -- causes the decision model to evaluate incomplete, constantly-changing text. The model might decide "should respond" based on an interim result that later changes to mean something completely different. Example: interim "what time" -> model decides question -> final "what time did you say the meeting was" -> model should have waited.

The opposite mistake -- waiting only for `speech_final` -- adds latency. The whole point of streaming STT is near-real-time results. Waiting for `speech_final` (which requires a silence gap) adds 300ms-1s of waiting after the user finishes speaking.

**How to avoid:**

1. **Use a three-tier processing strategy:**
   - **Interim results:** Check for name detection only (fast path, no model inference)
   - **Final results (`is_final: true`):** Accumulate into an utterance buffer
   - **Speech final or UtteranceEnd:** Send the complete accumulated utterance to the decision model

2. **Display interim results in the dashboard for responsiveness.** Show the user that the system is "hearing" them by displaying interim text in the overlay. This provides instant feedback without triggering the decision model on unstable text.

3. **Consider a "speculative decision" pattern.** When a `is_final: true` segment looks like a question or contains the AI's name, start preparing a response speculatively. If `speech_final` confirms the utterance is complete, the response is already partially ready. If more speech follows, cancel the speculative work. This reduces perceived latency without acting on unstable interim text.

4. **Never act on interim results for response generation.** Interim text is explicitly unstable. Only name detection (which is a simple string check, not an expensive model call) should use interim results.

**Warning signs:**
- AI responds to half-finished sentences
- AI starts responding, then the user continues speaking, creating an interruption
- Same utterance triggers multiple decision model evaluations
- Transcript buffer contains duplicate or contradictory entries

**Detection:** Log how many decision model evaluations fire per user utterance. Target: exactly 1 per utterance. If > 1, interim results are leaking into the decision pipeline.

**Phase to address:** Phase 1 (Deepgram integration) and Phase 2 (decision model). The transcript accumulation logic must be correct before the decision model can process utterances.

**Severity:** MAJOR

---

### Pitfall 10: Latency Spikes -- Network Jitter Destroys Conversational Flow

**What goes wrong:**
Cloud STT introduces network dependency where none existed before. The expected path:
- Audio chunk sent over WebSocket: ~5ms
- Deepgram processing: ~50-150ms
- Transcript returned over WebSocket: ~5ms
- Total: ~60-160ms

But real-world network conditions add variability:
- WiFi congestion: +50-200ms jitter
- ISP routing issues: +100-500ms spikes
- Deepgram server load: +50-200ms at peak
- WebSocket frame reassembly: +10-50ms
- DNS resolution on reconnect: +50-200ms

A 500ms latency spike means the user's speech is transcribed 500ms later than expected. Combined with decision model inference (~200ms) and TTS latency, the total response time can spike from 1s to 2s+, which crosses the threshold where conversation feels unnatural.

**How to avoid:**

1. **Monitor and log Deepgram latency continuously.** Track time from audio-send to transcript-receive for every result. Maintain a rolling average and P99. Alert when P99 exceeds 500ms.

2. **Implement a latency budget.** If Deepgram latency exceeds 500ms consistently (P90 over 5 minutes), consider:
   - Showing a "slow network" indicator to the user
   - Switching endpointing to more aggressive settings (shorter timeout)
   - Playing faster acknowledgment clips to fill the gap

3. **Consider a local Whisper fallback for degraded network.** The `CircuitBreaker` class already exists. If Deepgram latency exceeds 1s consistently or the WebSocket drops repeatedly, fall back to local Whisper (which is still installed, just not loaded). Accept the higher latency of batch transcription vs no transcription at all. Load Whisper on demand -- it takes 2-3 seconds to load but then works offline.

4. **Pre-buffer audio for jitter smoothing.** Don't send audio chunks one-at-a-time as they arrive from pasimple. Batch 2-3 chunks (170-255ms of audio) into a single WebSocket send. This reduces the number of network round-trips and smooths jitter at the cost of 85-170ms added latency (which is acceptable).

**Warning signs:**
- Occasional 1-2 second gaps where the system seems unresponsive
- Transcript timestamps have irregular spacing
- Dashboard latency metric shows high variance (std dev > mean)
- Works perfectly on wired ethernet, degrades on WiFi

**Detection:** Plot Deepgram latency over time. Healthy: tight distribution around 100-200ms. Unhealthy: long tail extending to 500ms+. Critical: bimodal distribution (indicating intermittent connectivity issues).

**Phase to address:** Phase 1 (monitoring) and Phase 3 (fallback). Basic latency tracking in Phase 1; fallback to Whisper in Phase 3.

**Severity:** MAJOR

---

## Moderate Pitfalls

Mistakes that cause delays, confusion, or technical debt.

---

### Pitfall 11: Deepgram SDK Version Instability

**What goes wrong:**
The Deepgram Python SDK has a history of breaking changes between versions. Documented issues include:
- SDK 3.0 to 3.1.1 broke WebSocket connections (Issue #279)
- API key passing via `DeepgramClientOptions` stopped working, requiring direct parameter passing (Issue #493)
- `UtteranceEnd` event handler registration was easy to miss (Issue #385)
- The HTTP 401 error from incorrect API key passing persists in some configurations even in v5.3.2 (Feb 2026 comment on Issue #493)

The `requirements.txt` currently specifies `deepgram-sdk>=3.0` which allows any version from 3.0 to latest. An `pip install --upgrade` or fresh install could pull a version with different behavior.

**Prevention:**
- Pin the exact SDK version in requirements.txt: `deepgram-sdk==3.9.0` (or whatever version you test with). Don't use `>=3.0`.
- Write integration tests that verify: connection opens, audio is received, transcript is returned, `speech_final` fires, `UtteranceEnd` fires, reconnection works. Run these tests before any SDK upgrade.
- Subscribe to Deepgram SDK release notes. Breaking changes happen silently without deprecation warnings.
- Keep the Whisper fallback path working so you can survive an SDK regression.

**Phase to address:** Phase 1 (dependency management).

**Severity:** MODERATE

---

### Pitfall 12: KeepAlive Message Format and Timing

**What goes wrong:**
The KeepAlive message must be sent as a TEXT WebSocket frame containing `{"type": "KeepAlive"}`. Sending it as a BINARY frame (which is how audio data is sent) may cause "incorrect handling and potential connection issues" per Deepgram's documentation. If the KeepAlive interval exceeds 10 seconds, the connection closes with a NET-0001 error.

The subtle trap: if your code uses a single "send" function for both audio (binary) and KeepAlive (text), and that function always sends binary frames, KeepAlive silently fails to keep the connection alive.

**Prevention:**
- Use the SDK's built-in KeepAlive mechanism if available (SDK v3+ has native keep-alive support).
- If sending KeepAlive manually, explicitly verify the WebSocket frame type: `ws.send(json.dumps({"type": "KeepAlive"}))` (text frame) vs `ws.send(audio_bytes)` (binary frame).
- Set KeepAlive interval to 5 seconds (well within the 10-second timeout).
- Log every KeepAlive sent and every connection close. If a close happens within seconds of a KeepAlive, the format is probably wrong.

**Phase to address:** Phase 1 (Deepgram integration).

**Severity:** MODERATE

---

### Pitfall 13: Transcript Quality Differences Between Whisper and Deepgram

**What goes wrong:**
The existing system is tuned around Whisper's behavior: its hallucination patterns, its punctuation style, its handling of filler words. Deepgram Nova-3 has different characteristics:
- **Better:** Lower WER (5-7% vs 10%+), fewer hallucinations on silence, faster
- **Different:** Different punctuation patterns, different handling of "um"/"uh", different capitalization
- **Potentially worse:** Different handling of technical terms, code-related speech, unique names

The existing hallucination filter (`HALLUCINATION_PHRASES` in `transcript_buffer.py`) is tuned for Whisper's specific hallucination patterns ("thank you for watching", "subtitles by", etc.). Deepgram doesn't produce these same hallucinations but may produce different artifacts.

The input classifier (`input_classifier.py`) uses regex patterns and semantic matching calibrated to Whisper's output style. Deepgram's different punctuation and capitalization could affect classification accuracy.

**Prevention:**
- Retune the hallucination filter for Deepgram. Remove Whisper-specific patterns and add any Deepgram-specific artifacts discovered during testing. Deepgram is less prone to hallucination on silence, but verify.
- Test the input classifier with Deepgram-produced transcripts. The classifier should be robust to punctuation differences, but verify with real speech.
- Keep the hallucination filter as a safety net even if Deepgram hallucinations are rare. The cost of a false hallucination detection (rejecting valid speech) is lower than the cost of a missed hallucination (sending garbage to the decision model).
- Document the behavioral differences between Whisper and Deepgram output for future reference.

**Phase to address:** Phase 1 (integration testing). Run existing speech test cases through Deepgram and compare output.

**Severity:** MODERATE

---

### Pitfall 14: State Management During Mode Transitions

**What goes wrong:**
The system must support both PTT mode (existing, works well) and always-on mode (new, uses Deepgram). Transitions between modes create state management complexity:
- User starts in always-on mode, switches to PTT mid-session: must close Deepgram WebSocket, start local Whisper STT
- User is in PTT mode, switches to always-on: must stop Whisper, open Deepgram WebSocket, start streaming
- Deepgram connection drops, system falls back to Whisper: must load Whisper model (2-3 second cold start)
- User starts interview/conversation mode while always-on is running: which STT takes priority?

Each transition has cleanup requirements (close connections, flush buffers, reset state machines) and initialization requirements (open connections, load models, calibrate thresholds).

**Prevention:**
- Implement STT as a pluggable interface with `start()`, `stop()`, `send_audio()`, and callback-based `on_transcript()`. Both Deepgram and Whisper implement this interface. The pipeline doesn't know which STT is active -- it just calls the interface.
- Mode transitions go through a single `switch_stt(provider)` method that handles cleanup and initialization atomically. No partial state transitions.
- Design the TranscriptBuffer to be STT-agnostic. It receives `TranscriptSegment` objects regardless of whether they came from Deepgram or Whisper. Tag segments with `source="deepgram"` or `source="whisper"` for debugging.
- Test mode transitions explicitly: always-on -> PTT -> always-on in a single session. Verify no orphaned connections, no leaked threads, no state corruption.

**Phase to address:** Phase 1 (architecture) and Phase 3 (fallback). Design the interface in Phase 1; implement Whisper fallback in Phase 3.

**Severity:** MODERATE

---

### Pitfall 15: Testing a Streaming WebSocket Integration Without Real Audio

**What goes wrong:**
The existing test suite (`test_live_session.py`) tests the Whisper pipeline with synthetic audio buffers and mocked models. Testing Deepgram streaming requires:
1. A real or mock WebSocket server that accepts audio and returns transcript events
2. Realistic timing (interim results, final results, speech_final, UtteranceEnd arrive asynchronously)
3. Error simulation (disconnections, timeouts, rate limits)
4. Cost awareness (every test that hits real Deepgram burns credit)

Without proper testing infrastructure, the Deepgram integration is only tested manually by speaking into a microphone. Manual testing misses edge cases: What happens when KeepAlive fails? When the connection drops mid-word? When Deepgram returns empty results? When network latency spikes?

**Prevention:**
- **Use Deepgram's mock server for integration tests.** Deepgram provides a streaming test suite with a mock server that accepts WebSocket connections and raw audio but doesn't transcribe. It confirms audio format and data receipt. Use this for format validation tests.
- **Build a mock WebSocket server for unit tests.** Create a local asyncio WebSocket server that:
  - Accepts connections with expected parameters
  - Accepts audio data (validates format)
  - Returns pre-recorded transcript sequences (loaded from JSON fixtures)
  - Can simulate disconnections, delays, and errors on demand
- **Record real Deepgram interactions as test fixtures.** During manual testing, record the exact sequence of events (open, audio sent, interim received, final received, speech_final, close) with timestamps. Replay these sequences in the mock server for deterministic testing.
- **Keep the test costs low.** Only hit real Deepgram in a dedicated "integration test" suite that runs manually (not in CI). Use the mock server for all automated tests.
- **Test the reconnection path explicitly.** The mock server should support: `mock.disconnect_after(5_seconds)` to verify that reconnection logic works correctly.

**Phase to address:** Phase 1 (test infrastructure). Build the mock before building the integration.

**Severity:** MODERATE

---

### Pitfall 16: Existing Pipeline Architecture Assumes Synchronous Batch STT

**What goes wrong:**
The current `_stt_stage` in `live_session.py` is designed around batch-mode Whisper: accumulate audio in a buffer, detect silence, transcribe the entire buffer at once, emit one TRANSCRIPT frame. This is fundamentally different from Deepgram streaming where:
- Results arrive asynchronously via WebSocket callbacks while audio is still being sent
- Multiple result types arrive (interim, final, speech_final, UtteranceEnd)
- The "transcription" isn't a single blocking call but a continuous stream of events
- Audio sending and transcript receiving happen concurrently, not sequentially

Trying to shoehorn Deepgram's event-driven streaming model into the existing `while self.running: frame = queue.get(); ... transcribe(buffer)` pattern will create an impedance mismatch. The STT stage needs to be restructured from "pull audio, process, emit" to "push audio continuously, handle async callbacks."

**Prevention:**
- Design the Deepgram STT stage as an event-driven component, not a polling loop:
  ```
  DeepgramSTT:
    - Thread/task: read from audio queue, send to WebSocket
    - Callback: on_transcript -> accumulate, check for speech_final
    - Callback: on_utterance_end -> emit complete utterance
    - Callback: on_error -> log, reconnect
    - Callback: on_close -> reconnect or fallback
  ```
- Keep the existing `_stt_stage` as the Whisper fallback path. Don't modify it.
- Create a new `_stt_deepgram_stage` that consumes from the same `_audio_in_q` but uses the Deepgram event-driven model internally.
- The output interface remains the same: emit `PipelineFrame(type=FrameType.TRANSCRIPT)` to `_stt_out_q`. This way the LLM stage doesn't know or care which STT backend produced the transcript.

**Phase to address:** Phase 1 (Deepgram integration). The STT stage architecture must be redesigned for streaming before any features can be built on top.

**Severity:** MODERATE

---

## Minor Pitfalls

Mistakes that cause annoyance but are easily fixable.

---

### Pitfall 17: Free Credit Exhaustion Without Warning

**What goes wrong:**
Deepgram's free tier provides $200 of credit. At $0.0077/min, that's ~433 hours of streaming. With active development (starting/stopping sessions, running tests against real Deepgram), credit burns faster than expected. There's no notification when credit is about to run out. One day the WebSocket connections just start failing with auth errors, and the developer spends time debugging a "connection issue" that's actually a billing issue.

**Prevention:**
- Check credit balance via Deepgram API periodically: `GET https://api.deepgram.com/v1/billing/balance`
- Add a startup check that logs remaining credit and warns if below a threshold (e.g., $20 remaining)
- Consider upgrading to Pay-as-You-Go ($0.0077/min with no free credit) early if development is active, to avoid the surprise cutoff

**Phase to address:** Phase 1 (setup).

**Severity:** MINOR

---

### Pitfall 18: Dashboard and Event Bus Need New Event Types

**What goes wrong:**
The dashboard (`dashboard.py`) and event bus (`event_bus.py`) were designed for the Whisper pipeline's event model: `stt_start`, `stt_complete`, single discrete transcription events. Deepgram streaming produces a different event model: continuous interim results, periodic final results, speech_final markers, UtteranceEnd messages, connection state changes, KeepAlive sent/received. If the event bus isn't updated, the dashboard shows nothing during Deepgram operation (no events mapped) or shows a flood of unmapped events.

**Prevention:**
- Add new EventType values: `deepgram_connected`, `deepgram_disconnected`, `deepgram_interim`, `deepgram_final`, `deepgram_speech_final`, `deepgram_utterance_end`, `deepgram_keepalive`, `deepgram_latency`
- Rate-limit interim result events (emit at most 1 per 200ms to the event bus, even if Deepgram sends more)
- Update the dashboard to show Deepgram connection state and latency metrics
- Keep the existing Whisper event types for fallback mode

**Phase to address:** Phase 1 (integration) or Phase 2 (dashboard update).

**Severity:** MINOR

---

### Pitfall 19: API Key Management for Deepgram

**What goes wrong:**
The Deepgram API key is loaded from config and passed to LiveSession (`deepgram_api_key` parameter, line 190). The key must never appear in logs, event bus entries, or error messages. The existing codebase has security guards (`_is_sensitive_path`, `_is_sensitive_command`) for local files but no API key sanitization for error messages from the Deepgram SDK. If the SDK includes the API key in error messages or stack traces, it could leak to conversation logs.

**Prevention:**
- Store the API key in `~/.config/push-to-talk/secrets` or source from `.env`, never in `config.json` (which may be committed)
- Wrap all Deepgram SDK calls with error handlers that sanitize the API key from exception messages before logging
- The existing `CLAUDE.md` security rules about never reading `.env` files apply -- ensure the deployment script handles this correctly
- Test that the API key doesn't appear in `events.jsonl` or any log output

**Phase to address:** Phase 1 (setup and security).

**Severity:** MINOR

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Deepgram connection (Phase 1) | VAD gating breaks endpointing (#1) | Stream continuously during active periods, KeepAlive during idle, disconnect during sleep |
| Audio format (Phase 1) | 24kHz format mismatch (#2) | Explicitly set encoding=linear16, sample_rate=24000, channels=1 |
| Connection resilience (Phase 1) | Mid-utterance disconnect (#3) | Local audio buffer, exponential backoff, timestamp realignment |
| Cost management (Phase 1) | Runaway streaming costs (#4) | Activity-based lifecycle, daily budget cap, cost tracking |
| Echo cancellation (Phase 1) | AI hears itself through Deepgram (#5) | Transcript fingerprinting, playback-aware tagging, extended cooldown window |
| End-of-speech (Phase 1) | `speech_final` fails in noise (#6) | Dual trigger: endpointing + utterance_end_ms |
| Interim results (Phase 1) | Processing too early/late (#9) | Three-tier strategy: interim for name, final for accumulation, speech_final for processing |
| SDK stability (Phase 1) | Breaking changes (#11) | Pin exact version, integration tests before upgrade |
| Architecture (Phase 1) | Batch STT assumption (#16) | Event-driven Deepgram stage, keep Whisper stage as fallback |
| Testing (Phase 1) | No mock infrastructure (#15) | Build mock WebSocket server before integration |
| Decision model (Phase 2) | False positives from TV/music (#7) | Name-activation only at launch, keyterm prompting, conversation state machine |
| Name detection (Phase 2) | Name not recognized (#8) | Fuzzy matching, keyterm boosting, check interim results, immediate acknowledgment |
| Network reliability (Phase 3) | Latency spikes (#10) | Latency monitoring, local Whisper fallback, audio pre-buffering |
| Mode transitions (Phase 3) | State corruption (#14) | Pluggable STT interface, atomic switch, agnostic TranscriptBuffer |
| Transcript quality (All phases) | Whisper vs Deepgram differences (#13) | Retune hallucination filter, test classifier with Deepgram output |

---

## Sources

**Official Deepgram Documentation (HIGH confidence):**
- [Recovering From Connection Errors & Timeouts](https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio) -- reconnection, audio buffering, timestamp realignment
- [Audio Keep Alive](https://developers.deepgram.com/docs/audio-keep-alive) -- KeepAlive format, 10-second timeout, billing implications
- [Determining Your Audio Format](https://developers.deepgram.com/docs/determining-your-audio-format-for-live-streaming-audio) -- encoding requirements, container vs raw audio
- [Configure Endpointing and Interim Results](https://developers.deepgram.com/docs/understand-endpointing-interim-results) -- is_final vs speech_final, default values, configuration
- [End of Speech Detection](https://developers.deepgram.com/docs/understanding-end-of-speech-detection) -- endpointing vs UtteranceEnd, noise limitations
- [Encoding](https://developers.deepgram.com/docs/encoding) -- supported encodings list, linear16 specification
- [Keyterm Prompting](https://developers.deepgram.com/docs/keyterm) -- Nova-3 keyterm boosting, 90% KRR improvement
- [API Rate Limits](https://developers.deepgram.com/reference/api-rate-limits) -- 150 concurrent connections (PAYG), project-scoped
- [Voice Agent Echo Cancellation](https://developers.deepgram.com/docs/voice-agent-echo-cancellation) -- AEC approaches, browser-level config
- [Pricing](https://deepgram.com/pricing) -- $0.0077/min Nova-3 streaming, $200 free credit

**Deepgram SDK Issues (HIGH confidence):**
- [Issue #279: WebSocket issue in SDK 3.1.1](https://github.com/deepgram/deepgram-python-sdk/issues/279) -- version upgrade breaks connections
- [Issue #385: UtteranceEnd never triggers](https://github.com/deepgram/deepgram-python-sdk/issues/385) -- event handler registration gotcha
- [Issue #493: ListenWebSocketClient.start failed](https://github.com/deepgram/deepgram-python-sdk/issues/493) -- API key passing regression
- [Discussion #1216: VAD before Deepgram STT](https://github.com/orgs/deepgram/discussions/1216) -- official recommendation against VAD gating

**Deepgram Community (MEDIUM confidence):**
- [Discussion #409: speech_final never returns True](https://github.com/orgs/deepgram/discussions/409) -- background noise breaks endpointing
- [Discussion #565: Twilio->Deepgram disconnecting](https://github.com/orgs/deepgram/discussions/565) -- reconnection patterns
- [Deepgram Blog: Holding Streams Open with KeepAlive](https://deepgram.com/learn/holding-streams-open-with-stream-keepalive) -- KeepAlive patterns

**Accuracy Comparisons (MEDIUM confidence):**
- [Deepgram: Whisper vs Deepgram](https://deepgram.com/learn/whisper-vs-deepgram) -- Nova-3 5-7% WER vs Whisper 10%+
- [Modal: Whisper vs Deepgram](https://modal.com/blog/whisper-vs-deepgram) -- independent comparison
- [Deepgram: Whisper-v3 Results](https://deepgram.com/learn/whisper-v3-results) -- Whisper-v3 4x more hallucinations than v2

**Codebase (HIGH confidence):**
- `live_session.py` -- existing STT stage, CircuitBreaker, barge-in, echo gating
- `continuous_stt.py` -- ContinuousSTT module (being replaced), VAD resampling, audio capture
- `transcript_buffer.py` -- TranscriptBuffer, hallucination filter, TranscriptSegment
- `input_classifier.py` -- classification patterns, semantic fallback
- `pipeline_frames.py` -- frame types, generation IDs
- `requirements.txt` -- current `deepgram-sdk>=3.0` specification

---
*Pitfalls research for: v2.0 Always-On Observer (Deepgram streaming + local decision model)*
*Researched: 2026-02-22*
*Supersedes: Previous PITFALLS.md (2026-02-21, local Whisper + Ollama focused)*
