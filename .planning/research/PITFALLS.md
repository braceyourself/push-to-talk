# Domain Pitfalls: Always-On Listening + Local LLM Observer

**Domain:** Converting PTT-gated voice assistant to always-on listening with local LLM monitoring layer
**Researched:** 2026-02-21
**Confidence:** HIGH (verified against existing codebase, hardware specs, official docs, community reports)

**Hardware context:** NVIDIA RTX 3070 (8GB VRAM), currently running faster-whisper large-v3 int8_float16 (~3.1GB VRAM). Adding Ollama Llama 3.2 3B (~2-3GB VRAM) means ~6GB of 8GB occupied. PipeWire audio, Linux X11.

---

## Critical Pitfalls

Mistakes that cause rewrites, make the feature unusable, or create runaway resource consumption.

---

### Pitfall 1: Audio Feedback Loop -- AI Hears Itself and Responds to Its Own Speech

**What goes wrong:**
With always-on listening, the microphone captures everything continuously -- including the AI's own TTS output from the speakers. The AI speaks, Whisper transcribes that speech, the monitoring LLM sees the transcript, decides it should respond, generates more speech, which gets transcribed again. Within 2-3 cycles the system is talking to itself in an infinite loop, consuming GPU resources and producing nonsensical audio.

This is the single most reported problem in always-on voice assistant development. OpenAI's Realtime API users documented this exact issue: "When the bot gives a response from the speaker, it listens to itself and takes that response as input, giving a response to what it said, creating a loop." The problem is worse with desktop speakers than headsets because acoustic echo cancellation (AEC) is less effective with open-air speaker/mic arrangements.

**Why it happens:**
The current system avoids this via `_stt_gated`: during playback, STT is suppressed (line 2262 of `live_session.py`). The mic stays physically live only for VAD-based barge-in detection. But in always-on mode, there is no gating -- STT must run continuously. The AI's speech output goes through PyAudio at 24kHz on the same device whose mic PipeWire is capturing at 24kHz. Without echo cancellation, the mic picks up the speaker output with ~5-50ms delay and slight acoustic distortion -- enough for Whisper to transcribe it as coherent speech.

**Why it happens in THIS codebase specifically:**
The existing `_stt_gated` mechanism (lines 2262-2288) was designed as a binary on/off. It clears the audio buffer entirely when gated. Converting to always-on requires replacing this binary gate with a smarter approach that can distinguish between AI speech and user speech in a continuous stream.

**How to avoid:**

1. **PipeWire echo cancellation module (primary defense).** PipeWire has a built-in `libpipewire-module-echo-cancel` that creates virtual source/sink nodes using WebRTC AEC. Configure it in `~/.config/pipewire/pipewire.conf.d/`:
   ```
   context.modules = [
       {   name = libpipewire-module-echo-cancel
           args = {
               library.name = aec/libspa-aec-webrtc
               capture.props = { node.name = "echo-cancel-capture" }
               playback.props = { node.name = "echo-cancel-playback" }
           }
       }
   ]
   ```
   Then change the PaSimple capture in `_audio_capture_stage` to record from the `echo-cancel-capture` virtual source instead of the default. Similarly, route playback through the `echo-cancel-playback` sink. This correlates the playback signal with the mic capture and subtracts it. **Confidence: HIGH** -- this is official PipeWire functionality with WebRTC AEC.

2. **Software-level transcript fingerprinting (secondary defense).** Keep a ring buffer of the last N sentences the AI spoke (already tracked in `_spoken_sentences`). When STT produces a transcript, compare it against recent AI speech using fuzzy matching (Levenshtein ratio > 0.7 = echo). Reject echo-matched transcripts before they reach the monitoring LLM. This catches cases where AEC is imperfect.

3. **Playback-aware STT suppression (tertiary defense).** Even in always-on mode, apply a "soft gate" during AI speech: continue STT but tag transcripts as `during_ai_speech=True`. The monitoring LLM receives these tagged transcripts and knows to ignore content that's likely echo vs. genuine barge-in. A genuine barge-in has different acoustic characteristics (overlapping speakers) that Whisper handles poorly anyway -- so the existing VAD-based barge-in mechanism is still the right tool for interruption detection.

**Warning signs:**
- AI produces transcripts that match its own recent speech
- System enters rapid-fire response cycles with no user input
- CPU/GPU usage spikes with no user activity
- Conversation logs show the AI talking to itself

**Detection:** Log a metric `echo_match_ratio` for every STT transcript against recent AI speech. If > 5% of transcripts match, echo cancellation is insufficient.

**Phase to address:** Phase 1 (first thing to solve -- nothing else works without this). PipeWire AEC should be set up before any continuous listening code runs.

**Severity:** CRITICAL

---

### Pitfall 2: GPU Memory Exhaustion -- Whisper + Ollama + Existing Workloads Don't Fit

**What goes wrong:**
The RTX 3070 has 8GB VRAM. Current usage:
- faster-whisper large-v3 int8_float16: ~3.1GB VRAM
- Ollama Llama 3.2 3B (Q4_K_M): ~2-3GB VRAM (varies with context window size)
- CUDA runtime overhead: ~300-500MB
- Total: ~5.4-6.6GB baseline

This leaves only 1.4-2.6GB headroom. With continuous Whisper transcription (model stays loaded, inference runs every silence-detected segment) and continuous Ollama monitoring (model stays loaded, inference runs every few seconds on new transcript chunks), both models compete for VRAM. If context windows grow (Ollama default 4096 tokens, but transcript monitoring might push larger), memory can exceed 8GB. CUDA will either OOM-kill one process or fall back to CPU, which for Ollama means 5-20x slower inference (3-6 tokens/s CPU vs 50 tokens/s GPU for 3B).

The danger is that this works fine in testing (short sessions, small context) but fails in production (2-hour session, growing context window, multiple Whisper transcriptions queued up).

**Why it happens:**
faster-whisper and Ollama manage their own CUDA contexts independently. They don't coordinate memory allocation. Ollama uses llama.cpp which pre-allocates VRAM for the full context window at load time. faster-whisper/CTranslate2 allocates per-inference. During a Whisper inference, temporary activations spike VRAM usage by 0.5-1GB briefly. If this spike overlaps with an Ollama inference that's also spiking, total VRAM exceeds 8GB.

**How to avoid:**

1. **Use Whisper `small` or `medium` model for continuous monitoring, not `large-v3`.** The current system uses `large-v3` because accuracy matters when every transcript becomes an LLM prompt. But in always-on mode, most transcripts are ambient monitoring -- the monitoring LLM decides relevance. A smaller Whisper model with lower accuracy is acceptable because:
   - False negatives (missed speech) are caught on the next segment
   - False positives (hallucinations) are filtered by the monitoring LLM
   - The `small` model uses ~1GB VRAM in int8 vs ~3.1GB for `large-v3`

   Keep `large-v3` available for re-transcription when the monitoring LLM decides the user is addressing the AI. On "hey Russel" detection, re-transcribe the recent audio buffer with `large-v3` for maximum accuracy.

2. **Set explicit Ollama context window size.** Default is 4096 tokens. For monitoring, limit to 2048 or even 1024 tokens with `num_ctx` parameter. The monitoring LLM only needs recent transcript context (last ~30 seconds of conversation), not a full conversation history. Smaller context = less VRAM.
   ```python
   response = ollama.chat(
       model="llama3.2:3b",
       messages=messages,
       options={"num_ctx": 2048}
   )
   ```

3. **Monitor VRAM in real-time.** Add a periodic VRAM check (every 30s) using `nvidia-smi` or `pynvml`. If free VRAM drops below 500MB, reduce Ollama context size or defer non-critical Whisper transcriptions. Log VRAM usage to the event bus for dashboard visibility.

4. **Test with realistic session lengths.** The test suite runs short clips. Add a stress test that runs continuous audio for 30+ minutes to catch memory leaks and accumulation.

**Warning signs:**
- CUDA OOM errors in logs after extended sessions
- Ollama responses suddenly becoming very slow (fell back to CPU)
- System lag/stuttering during simultaneous Whisper + Ollama inference
- VRAM usage creeping up over time without decreasing

**Detection:** Log VRAM usage at each Whisper inference and each Ollama call. Plot over session duration. Any upward trend indicates a leak or unbounded growth.

**Phase to address:** Phase 1 (resource budgeting must be established before building the pipeline). Validate with the actual hardware before committing to model sizes.

**Severity:** CRITICAL

---

### Pitfall 3: Whisper Hallucinations Explode With Continuous Ambient Audio

**What goes wrong:**
The current system only transcribes audio after the user presses PTT and speaks. Whisper receives clean, speech-containing audio segments. In always-on mode, Whisper receives everything: keyboard typing, chair creaking, HVAC noise, music playing, phone notifications, other people in the room, TV/podcast audio, eating/drinking sounds. Research shows Whisper has a 40.3% hallucination rate on non-speech audio segments, producing phantom transcripts like "Thank you for watching," "Please subscribe," "I'm sorry," and other training-data artifacts.

The existing hallucination filter (lines 2193-2199) catches 18 known phrases. But continuous ambient audio produces much more diverse hallucinations. The monitoring LLM receives a steady stream of phantom transcripts, treats them as conversation, and the AI starts responding to non-existent speech. The user hears the AI say "You're welcome!" when nobody said anything.

**Why it happens:**
Whisper was trained on 680,000 hours of web audio, much of it YouTube content with "thank you for watching" outros. When Whisper encounters non-speech audio with any energy, its decoder generates the most likely text conditioned on acoustic features that vaguely resemble speech patterns. The existing multi-layer filter (no_speech_prob >= 0.6, avg_logprob < -1.0, compression_ratio > 2.4) was tuned for PTT audio where most segments contain actual speech. With continuous audio, the ratio inverts: most segments are non-speech, and the filters need to be much more aggressive.

**How to avoid:**

1. **Use Whisper's built-in VAD filter more aggressively.** The current config has `vad_filter=True` (line 1464). In continuous mode, increase the VAD threshold. faster-whisper supports `vad_parameters` dict:
   ```python
   segments_gen, info = self.whisper_model.transcribe(
       samples, language="en",
       vad_filter=True,
       vad_parameters={
           "threshold": 0.5,           # Default 0.5, increase to 0.6-0.7
           "min_speech_duration_ms": 250,  # Reject <250ms "speech"
           "min_silence_duration_ms": 300, # Merge nearby segments
       },
   )
   ```

2. **Expand the hallucination phrase list significantly.** The top 30 Whisper hallucinations are well-documented. Add all of them:
   ```python
   HALLUCINATION_PHRASES = {
       "thank you", "thanks for watching", "thanks for listening",
       "thank you for watching", "thanks for your time",
       "goodbye", "bye", "you", "the end", "to", "so",
       "please subscribe", "like and subscribe", "i'm sorry",
       "hmm", "uh", "um", "oh",
       # Additional documented hallucinations:
       "subtitles by the amara org community",
       "thank you very much", "see you next time",
       "please like and subscribe", "don't forget to subscribe",
       "thanks for tuning in", "until next time",
       "the following is a transcript",
       "this video is sponsored by", "link in the description",
       "leave a comment below", "hit the bell icon",
       # Single-word filler hallucinations common with noise:
       "the", "a", "i", "it", "and", "is",
   }
   ```

3. **Add a transcript confidence gate before the monitoring LLM.** Don't send every transcript to Ollama. Only forward transcripts where:
   - Whisper's overall confidence (average logprob across all kept segments) exceeds a threshold
   - The transcript is longer than 3 words (very short transcripts from ambient noise are almost always hallucinations)
   - The transcript doesn't repeat the previous transcript (ambient noise produces identical hallucinations repeatedly)

4. **Rate-limit STT in low-activity periods.** If no speech energy above `SPEECH_ENERGY_MIN` (200 RMS) has been detected for 30+ seconds, reduce transcription frequency. Don't transcribe every silence-detected segment -- skip segments where peak RMS never exceeded a low threshold.

**Warning signs:**
- Conversation logs show repeated identical phrases from STT
- AI responds when the room is empty
- Monitoring LLM receives dozens of short (1-3 word) transcripts per minute
- STT rejection rate exceeds 50% in always-on mode

**Detection:** Log transcript acceptance/rejection ratio. In always-on mode with no active conversation, acceptance rate should be < 5%. If higher, hallucination filtering is insufficient.

**Phase to address:** Phase 1 (continuous STT architecture). Must be solved before connecting STT output to the monitoring LLM.

**Severity:** CRITICAL

---

### Pitfall 4: Monitoring LLM Context Window Grows Unboundedly

**What goes wrong:**
The monitoring LLM (Ollama Llama 3.2 3B) observes a continuous transcript stream and decides when the AI should respond. The naive implementation appends every transcript to a conversation history and sends the full history on each inference. After 30 minutes of ambient conversation, the context contains thousands of tokens. At 2048 token context, this overflows and truncates from the beginning -- losing the conversation topic. At 4096 tokens, VRAM usage grows (see Pitfall 2). Either way, the monitoring LLM's decisions degrade: it either lacks context (truncated) or the system runs out of memory (too large).

Even with a sliding window, the monitoring LLM loses track of the overall conversation arc. It sees the last 30 seconds but doesn't know that 5 minutes ago the user said "Let me think about the architecture for a bit" -- which explains why they've been quiet.

**Why it happens:**
Continuous monitoring produces a fundamentally different context management challenge than request/response conversation. In the current system, each user utterance produces one LLM turn -- context grows linearly with turns. In always-on mode, ambient audio produces continuous text even when nobody is actively conversing. A 2-hour session with background conversation or TV audio could produce 50,000+ tokens of transcript.

**How to avoid:**

1. **Two-tier context: rolling summary + recent buffer.** Maintain two data structures:
   - `recent_buffer`: Last 60 seconds of raw transcript (verbatim, ~200-500 tokens)
   - `conversation_summary`: Periodically summarized context (updated every 2-3 minutes by the monitoring LLM itself, ~100-200 tokens)

   Each monitoring inference receives: system prompt + conversation_summary + recent_buffer. Total stays under 1024 tokens consistently.

2. **Event-driven monitoring, not continuous polling.** Don't call the monitoring LLM on every transcript. Instead:
   - Queue transcripts into a buffer
   - Trigger monitoring inference only on "events": new speech after silence, topic change detected, name mention, direct question pattern
   - Use cheap heuristics (keyword matching, question mark detection, name detection) as pre-filters before invoking the LLM

3. **Distinguish conversation from ambient audio in context.** Tag transcripts with metadata: `{text: "...", speaker: "unknown", energy_level: "high", during_ai_speech: false}`. The monitoring LLM can use these signals to weight relevance. Low-energy, unknown-speaker transcript during ambient periods gets lower context priority.

4. **Explicit context reset signals.** When the monitoring LLM detects a topic change or long silence (> 5 minutes), it should emit a summary of the previous conversation segment and reset the recent buffer. This prevents context accumulation across unrelated conversations.

**Warning signs:**
- Monitoring LLM inference latency increasing over session duration
- Ollama responses becoming slower (context window growing)
- AI responds to topics from 10+ minutes ago that are no longer relevant
- VRAM usage creeping up (context window expansion)

**Detection:** Log monitoring LLM prompt token count at each inference. It should stay roughly constant (within the budget). Any upward trend indicates unbounded growth.

**Phase to address:** Phase 2 (monitoring LLM architecture). The context strategy must be designed before building the monitoring pipeline.

**Severity:** CRITICAL

---

### Pitfall 5: False Positive Responses -- AI Speaks When It Shouldn't

**What goes wrong:**
The monitoring LLM decides to respond when it shouldn't. Common false positive scenarios:
- User is on a phone call; AI answers a question directed at someone else
- User is watching a video; AI responds to dialogue in the video
- User mutters to themselves; AI takes it as a request
- User is having a conversation with another person in the room; AI interjects inappropriately
- Background TV/podcast says something question-like; AI answers it

This is the core UX challenge of proactive AI. Every false positive response is deeply annoying -- more so than a missed response. Users will disable the feature after 2-3 false positives in a session.

**Why it happens:**
The monitoring LLM receives transcript text without reliable speaker identification. It can't distinguish between:
- "Hey Russel, what time is it?" (directed at AI)
- "What time is it?" (directed at another person in the room)
- "What time is it?" (said by a character in a video)

Without speaker diarization or explicit addressing (wake word), the LLM must guess intent from text alone. Even a well-tuned LLM will get this wrong frequently in multi-person environments.

**How to avoid:**

1. **Default to NOT responding.** The monitoring LLM's default output should be "no response needed." It should only trigger a response when it's highly confident the user is addressing the AI. Design the system prompt to be conservative:
   ```
   You are a passive observer. ONLY recommend responding when:
   1. The user explicitly says "Russel" or "hey Russel"
   2. The user asks a direct question AND no other person is present in the conversation
   3. The user explicitly asks for help ("can you help me", "look this up")

   When in doubt, DO NOT respond. A missed response is far better than an unwanted one.
   ```

2. **Require name-based activation for proactive responses.** For v2.0 initial launch, only respond proactively when the user says the AI's name. This is effectively a software wake word. Over time, as confidence in the monitoring LLM improves, gradually expand the activation criteria.

3. **Implement a response confidence threshold.** The monitoring LLM should output a confidence score (0-1) alongside its response decision. Only trigger a response above 0.8. Log all decisions (including suppressed ones) for tuning.

4. **Add a "recently active" context signal.** If the user was actively conversing with the AI in the last 2 minutes (a back-and-forth exchange), lower the activation threshold. If the AI hasn't been addressed in 10+ minutes, raise it. This prevents the AI from randomly interjecting after long silence.

5. **Cooldown after unprompted responses.** After each proactive (non-name-triggered) response, enforce a 60-second cooldown before the next proactive response can fire. This prevents rapid-fire false positives.

**Warning signs:**
- Users saying "I wasn't talking to you" or "stop"
- AI responding during phone calls or video watching
- AI interjecting into multi-person conversations inappropriately
- User disabling always-on mode frequently

**Detection:** Track `proactive_response_suppressed_by_user` (user interrupts AI within 2 seconds of proactive response = likely unwanted). If suppression rate exceeds 20%, the monitoring LLM is too aggressive.

**Phase to address:** Phase 2 (monitoring LLM prompt engineering) and Phase 3 (tuning). Start extremely conservative; it's much easier to loosen restrictions than to recover user trust after false positives.

**Severity:** CRITICAL

---

## Major Pitfalls

Mistakes that cause significant rework or degrade UX.

---

### Pitfall 6: Name Detection Accuracy -- "Hey Russel" Has No Dedicated Wake Word Engine

**What goes wrong:**
The project explicitly chose "software-based name recognition instead" of a dedicated wake word engine (PROJECT.md, "Out of Scope"). This means name detection relies on Whisper transcribing "hey Russel" accurately. But:
- Whisper may transcribe it as "hey Russell", "hey Rusel", "hey wrestle", "hey rustle"
- In noisy environments, Whisper may miss it entirely
- Short utterances like "Russel" (just the name) may get filtered as too short or hallucination
- Whisper processes audio in segments after silence detection, meaning there's a 0.8s silence delay + Whisper inference time before the name is even recognized -- total latency of 1-2 seconds after saying "hey Russel"

Benchmark data: dedicated wake word engines (Picovoice Porcupine, openWakeWord) achieve <5% false rejection at 1 false acceptance per 10 hours. STT-based name detection through Whisper will have significantly worse accuracy because Whisper was not trained for this task.

**Why it happens:**
Wake word detection and speech-to-text are fundamentally different tasks. Wake word engines run on streaming audio in real-time (~10ms latency), using lightweight models specifically trained to recognize one phrase. Whisper processes audio in batches after silence detection, using a general-purpose model trained on diverse speech. Using Whisper for wake word detection is like using a dictionary to detect Morse code -- it works sometimes, but it's the wrong tool.

**How to avoid:**

1. **Fuzzy match the name in transcripts.** Don't require exact string match "hey Russel". Match against variants:
   ```python
   NAME_VARIANTS = {"russel", "russell", "rusel", "russ", "hey russel", "hey russell"}

   def _detect_name(self, transcript: str) -> bool:
       words = transcript.lower().split()
       for i, word in enumerate(words):
           if word in NAME_VARIANTS:
               return True
           # Check bigrams for "hey russel" as one fuzzy match
           if i > 0:
               bigram = f"{words[i-1]} {word}"
               for variant in NAME_VARIANTS:
                   if variant in bigram or fuzz.ratio(bigram, variant) > 80:
                       return True
       return False
   ```

2. **Consider adding openWakeWord as a parallel lightweight detector.** openWakeWord is a Python library that can run a custom wake word model on streaming audio with ~10ms latency, consuming minimal CPU. It can run in parallel with the main STT pipeline on the raw audio stream, providing instant name detection without waiting for Whisper. This doesn't replace Whisper-based detection (which catches names mentioned mid-sentence) but provides fast activation for the explicit "hey Russel" case. **Confidence: MEDIUM** -- this is adding a dependency to avoid, per the "Out of Scope" decision. But the latency difference (10ms vs 1-2s) may be worth revisiting.

3. **Shorten the path from name to response.** Even with STT-based detection, optimize the pipeline: when "Russel" is detected in a transcript, skip the normal monitoring LLM decision path and immediately activate response mode. Don't wait for the monitoring LLM to decide -- the name IS the decision.

4. **Add the name to Whisper's initial prompt.** faster-whisper supports an `initial_prompt` parameter that biases transcription:
   ```python
   segments_gen, info = self.whisper_model.transcribe(
       samples, language="en",
       initial_prompt="Russel is an AI assistant.",
       ...
   )
   ```
   This makes Whisper more likely to transcribe the name correctly rather than as a similar-sounding word.

**Warning signs:**
- User says "hey Russel" multiple times before getting a response
- AI activates on words that sound like "Russel" but aren't (false positives)
- Noticeable delay (>2s) between saying the name and AI acknowledging
- Name detection works in quiet room but fails with background noise

**Detection:** Log every name detection event with the raw transcript and confidence. Compare true positive rate (user said name, detected) vs false negative rate (user said name, not detected). Target: >90% true positive, <1 false positive per hour.

**Phase to address:** Phase 2 (name detection implementation). Should be one of the first features tested with real users because it's the primary activation mechanism.

**Severity:** MAJOR

---

### Pitfall 7: Ollama Cold Start and Model Eviction Delays Response

**What goes wrong:**
Ollama unloads models from memory after 5 minutes of inactivity by default. In always-on mode, if no interesting conversation happens for 5+ minutes, the monitoring model gets unloaded. The next transcript triggers a cold load: model reads from disk, allocates VRAM, initializes -- taking 2-5 seconds on an SSD with GPU, or 10-30 seconds if falling back to CPU. The user says "hey Russel, what time is it?" and waits 5 seconds for the monitoring LLM to even start processing, then another 200ms for inference. Total: 5+ seconds of dead silence.

Even with `OLLAMA_KEEP_ALIVE=-1` (keep forever), there are documented bugs where models get evicted unexpectedly (GitHub issue #9410). And if the system is also used for other Ollama tasks (coding, chat), those tasks may trigger `OLLAMA_MAX_LOADED_MODELS` eviction of the monitoring model.

**Why it happens:**
Ollama was designed for interactive use (human types a query, waits for response), not for always-on monitoring where the model must be instantly available. The default 5-minute keep_alive assumes periods of inactivity between queries. For a monitoring task, "inactivity" of 5 minutes is normal (user goes to the kitchen, comes back, starts talking).

**How to avoid:**

1. **Set `OLLAMA_KEEP_ALIVE=-1` in the systemd service.** Configure Ollama to keep models loaded indefinitely:
   ```ini
   [Service]
   Environment="OLLAMA_KEEP_ALIVE=-1"
   ```
   This prevents the default 5-minute eviction.

2. **Heartbeat keepalive.** Send a lightweight inference request to Ollama every 2 minutes with `keep_alive: -1` to ensure the model stays loaded, even if the `OLLAMA_KEEP_ALIVE` environment variable has bugs:
   ```python
   async def _ollama_keepalive(self):
       while self.running:
           await asyncio.sleep(120)
           try:
               ollama.chat(
                   model="llama3.2:3b",
                   messages=[{"role": "user", "content": "ping"}],
                   options={"num_predict": 1},  # Generate just 1 token
                   keep_alive=-1,
               )
           except Exception:
               pass
   ```

3. **Pre-load on session start.** When the always-on session starts, immediately trigger a dummy Ollama inference to load the model into VRAM. Log the load time. If it exceeds 3 seconds, warn that the system may have performance issues.

4. **Measure and log cold start vs warm inference latency.** Every Ollama call should log wall-clock time. If inference suddenly jumps from ~200ms to 3000ms+, the model was evicted and reloaded. Alert on this pattern.

**Warning signs:**
- First response after idle period takes 3-5x longer than normal
- VRAM usage drops and then spikes (model unloaded/reloaded)
- Ollama logs show "loading model" entries during a session
- Monitoring decisions arrive too late to be useful

**Phase to address:** Phase 2 (Ollama integration). Configure keepalive before building the monitoring pipeline.

**Severity:** MAJOR

---

### Pitfall 8: Latency Between "Should Respond" and Actual Response

**What goes wrong:**
The monitoring LLM decides the AI should respond. But the response isn't instant -- it must traverse a multi-step pipeline:
1. Audio buffer fills (0.5-0.8s silence detection)
2. Whisper transcribes (0.5-2s for large-v3 on 3-10s audio)
3. Monitoring LLM decides (0.2-0.5s Ollama inference)
4. Response LLM generates text (Claude CLI: 2-5s first token; Ollama: 0.2-1s)
5. TTS generates audio (Piper: 0.1-0.3s per sentence)
6. Playback begins

Total latency from user finishing their sentence to AI starting to speak: **3-9 seconds**. For a proactive "I know the answer to that" response, 3-9 seconds is an eternity. The conversation has moved on. The user may have already answered their own question or started a new topic.

**Why it happens:**
The current pipeline was designed for PTT where latency is acceptable (user pressed the button, they expect to wait). Always-on expects ambient responsiveness -- the AI should feel like a participant, not someone joining a Zoom call with bad latency. Every pipeline stage adds latency, and the stages are sequential.

**How to avoid:**

1. **Speculative pre-processing.** When the monitoring LLM detects a likely-addressable question (confidence > 0.5 but below the response threshold), start preparing a response speculatively. If confidence crosses the threshold before the response is ready, the response pipeline is already partway done. If confidence drops, discard the speculative work.

2. **Parallel pipeline stages.** Overlap monitoring LLM decision with response generation. Instead of: transcribe -> decide -> generate -> speak, do:
   - Transcribe -> simultaneously: (a) send to monitoring LLM for decision, (b) send to response LLM optimistically
   - If monitoring LLM says "respond": the response LLM is already working, so latency is reduced
   - If monitoring LLM says "don't respond": cancel the response LLM work

   Cost: wasted Claude CLI / Ollama computation on false starts. Benefit: latency reduced by the monitoring LLM decision time (~200-500ms).

3. **Use quick response library for immediate acknowledgment.** When the monitoring LLM decides to respond, immediately play a contextual quick response clip ("Let me think about that", "Good question") from the existing response library. This fills the silence while the response LLM generates the real answer. The infrastructure for this already exists in `_filler_manager` and `StreamComposer`.

4. **Response backend selection should factor in latency.** For time-sensitive proactive responses (answering a quick factual question), use Ollama (~0.2-1s first token). For complex responses that benefit from Claude's depth, accept the higher latency but play an acknowledgment first.

**Warning signs:**
- Users repeat their question because they think the AI didn't hear
- AI responses feel disconnected from the conversation flow
- Response latency exceeds 3 seconds regularly
- Quick responses (fillers) play but the actual response takes 5+ more seconds

**Detection:** Log end-to-end latency from "silence detected" to "first audio byte played" for every response. Target: <3s for Ollama backend, <5s for Claude CLI backend (with filler).

**Phase to address:** Phase 3 (pipeline optimization). Get basic functionality working first, then optimize latency.

**Severity:** MAJOR

---

### Pitfall 9: Privacy -- Always-On Mic Crosses a Trust Boundary

**What goes wrong:**
The current PTT system has implicit user consent: you press a button, you know the mic is active, you release the button, the mic stops listening. Always-on fundamentally changes this contract. The mic is always active, recording everything -- private conversations, phone calls, sensitive discussions, family moments. Users (and other people in the room who didn't consent) may not realize audio is being continuously processed.

Even though processing is local (Whisper on-device, Ollama on-device), the transcript data is potentially stored in conversation logs (JSONL) and sent to Claude CLI for response generation (which goes to Anthropic's API). A continuous transcript of ambient room audio being sent to a cloud API is a privacy concern even if the audio itself stays local.

**Why it happens:**
Developers building for themselves often don't think about privacy because they consented implicitly by building the feature. But other people in the room didn't. And even the developer may forget the mic is on during sensitive phone calls.

**How to avoid:**

1. **Persistent visual indicator.** The system tray indicator must clearly show always-on mode is active. Use a distinct color (red/orange pulsing dot) that's impossible to confuse with "inactive." This is non-negotiable for any always-on mic feature.

2. **Audio never leaves the device in raw form.** The architecture should guarantee:
   - Raw audio stays in memory only (never written to disk in always-on monitoring mode)
   - Only STT transcripts (text) are stored or transmitted
   - Transcripts sent to Claude CLI are the filtered/summarized version, not raw continuous STT output

3. **Mute zones.** Allow the user to set time-based or trigger-based mute rules:
   - "Mute when screen is locked"
   - "Mute during phone calls" (detect PipeWire phone call streams)
   - Physical mute button always available (already exists -- Escape key interrupt)

4. **Clear local data retention policy.** Conversation JSONL logs from always-on mode should be auto-purged after a configurable period (default: 24 hours). Don't accumulate months of ambient conversation transcripts.

5. **Only send relevant context to Claude CLI.** When the monitoring LLM decides to respond and routes to Claude CLI, send only the relevant conversation snippet (the user's question + recent context), not the entire ambient transcript history. The monitoring LLM acts as a privacy filter.

**Warning signs:**
- Users forget the mic is on during sensitive calls
- Other people in the room are surprised to learn the mic is active
- Conversation logs contain unintended private content
- Large volumes of ambient transcript data accumulating on disk

**Phase to address:** Phase 1 (indicator/visibility) and Phase 2 (data flow architecture). Privacy must be designed in, not bolted on.

**Severity:** MAJOR

---

### Pitfall 10: Existing Pipeline Architecture Assumes PTT Gating

**What goes wrong:**
The current pipeline is deeply designed around PTT-triggered discrete turns:
- `_stt_stage` buffers audio until silence, transcribes, produces one `TRANSCRIPT` frame per utterance
- `_llm_stage` waits for a `TRANSCRIPT` frame, sends to CLI, reads response
- `_filler_manager` spawns per-transcript to fill silence during LLM thinking
- `generation_id` increments on barge-in/interrupt to discard stale frames
- `_stt_gated` flag suppresses STT during AI speech

Converting to always-on requires changing the data flow fundamentally: STT produces continuous transcript fragments (not discrete turns), a new monitoring stage sits between STT and LLM, the LLM stage is triggered by monitoring decisions (not by every transcript), and multiple "generations" may be active simultaneously (monitoring inference + response inference).

Attempting to bolt always-on behavior onto the existing pipeline without restructuring leads to state management nightmares: `generation_id` conflicts between monitoring and response tasks, `_stt_gated` no longer meaningful, filler manager firing on monitoring transcripts that aren't meant for LLM processing.

**Why it happens:**
The 5-stage pipeline (capture -> STT -> LLM -> TTS -> playback) is a clean sequential architecture for PTT. Adding a monitoring layer turns it into a branching pipeline: capture -> STT -> monitoring -> (maybe) LLM -> TTS -> playback. The "maybe" branch is the hard part -- the monitoring decision affects whether subsequent stages activate.

**How to avoid:**

1. **Introduce a transcript dispatcher between STT and LLM.** Instead of STT feeding directly to `_stt_out_q` which feeds `_llm_stage`, add a new stage:
   ```
   STT -> transcript_dispatcher -> [monitoring LLM, name detector, question detector]
                                 -> (on trigger) -> LLM -> TTS -> playback
   ```
   The dispatcher is the central routing point. It accumulates transcripts, runs cheap pre-filters, and decides when to invoke the monitoring LLM vs. when to directly route to the response LLM.

2. **Separate generation_ids for monitoring vs response.** The monitoring LLM runs continuously and shouldn't be affected by response generation interrupts. Give monitoring its own lifecycle:
   ```python
   self._monitoring_active = True  # Independent of generation_id
   self._response_gen_id = 0      # For response pipeline (existing generation_id)
   ```

3. **Keep the existing pipeline working for PTT mode.** Don't break the existing flow. The transcript dispatcher should support both modes:
   - PTT mode: transcript goes directly to LLM (current behavior)
   - Always-on mode: transcript goes to monitoring first

   Feature flag in config: `"listening_mode": "ptt"` vs `"always_on"`.

4. **Implement as a parallel pipeline, not a replacement.** The always-on monitoring runs alongside the existing pipeline, not instead of it. The monitoring pipeline observes STT output and emits "respond" events. These events trigger the existing LLM -> TTS -> playback pipeline. This minimizes changes to proven code.

**Warning signs:**
- State management bugs (wrong generation_id, stale frames)
- PTT mode broken after always-on changes
- Barge-in detection confused by monitoring LLM activity
- Filler manager playing clips during ambient monitoring (not during active response)

**Phase to address:** Phase 1 (architecture refactor). This is the foundational change that enables everything else.

**Severity:** MAJOR

---

## Moderate Pitfalls

Mistakes that cause delays, confusion, or technical debt.

---

### Pitfall 11: Continuous Whisper Inference Blocks the Event Loop

**What goes wrong:**
The current STT stage runs Whisper in a thread executor (`run_in_executor`, line 2359) to avoid blocking the asyncio event loop. This works fine for occasional transcriptions (one every 2-10 seconds during active conversation). In always-on mode, Whisper may need to transcribe every 1-3 seconds (continuous speech from TV, podcast, or multi-person conversation). If `large-v3` takes 0.5-2s per transcription, and transcriptions are requested every 1-3 seconds, the executor thread is saturated. Subsequent transcription requests queue up, introducing latency that grows over time.

The default executor is `ThreadPoolExecutor` with a limited number of workers. If all workers are busy with Whisper inference, other executor-based operations (file I/O, subprocess communication) also stall.

**Prevention:**
- Use a dedicated `ThreadPoolExecutor(max_workers=1)` for Whisper inference, separate from the default executor. This prevents Whisper from blocking other executor tasks.
- Implement a "skip if busy" pattern: if a Whisper inference is already running, buffer the new audio segment instead of queuing another inference. Transcribe the combined buffer when the current inference completes.
- For continuous monitoring mode, use the `small` model (~4x faster inference) instead of `large-v3`.

**Phase to address:** Phase 1 (STT architecture).

**Severity:** MODERATE

---

### Pitfall 12: Faster-Whisper Memory Leak on Long-Running Processes

**What goes wrong:**
A documented issue with faster-whisper shows memory utilization gradually growing during long transcription sessions, eventually hitting OOM. The issue was reported for a 5.5-hour audio file where memory grew from ~10% to 100% over 2 hours. In always-on mode, the process runs for hours or days. Even a small per-inference leak (e.g., unreleased tensor buffers, accumulating log data) compounds into a crash.

**Prevention:**
- Monitor process RSS memory every 5 minutes. Log it to the event bus. Alert if memory grows by more than 20% from session start.
- Consider periodic Whisper model reload: every 2 hours, unload and reload the model to release any accumulated memory. This adds a 2-3 second gap in STT coverage but prevents OOM.
- Test with extended sessions (4+ hours) before shipping. The existing test suite runs short clips -- add a soak test.
- Set Python garbage collection to be more aggressive during idle periods: `gc.collect()` after each transcription completes.

**Phase to address:** Phase 3 (reliability/long-running). Not a launch blocker but must be addressed before production use.

**Severity:** MODERATE

---

### Pitfall 13: Response Backend Selection Heuristic Is Hard to Get Right

**What goes wrong:**
The system must choose between Claude CLI (deep, slow, costs API calls) and Ollama (fast, shallow, free) for each response. The naive approach: "use Ollama for simple questions, Claude for complex ones" requires classifying question complexity, which is itself an LLM task. If the monitoring LLM (Ollama 3B) misclassifies a complex question as simple, Ollama generates a shallow/wrong answer. If it misclassifies a simple question as complex, the user waits 5 seconds for Claude to say "It's 3pm."

The backend selection must also handle failure modes: Claude CLI is offline (no network), Ollama is overloaded (another model loaded), or one backend is consistently producing bad results.

**Prevention:**
- Start with a simple rule: **name-triggered responses use Claude CLI, ambient/proactive responses use Ollama.** If the user explicitly addresses the AI, they expect a good answer and will tolerate latency. If the AI is interjecting proactively, speed matters more than depth.
- The existing `CircuitBreaker` class (line 148) handles failure cascading. Extend it to cover both backends: if Claude CLI fails 3 times, fall back to Ollama; if Ollama fails 3 times, fall back to Claude CLI.
- Don't try to auto-detect "complexity" initially. Let the user set a preference in config, and auto-fallback handles the rest.
- Log which backend was selected for each response, along with user satisfaction signals (did the user ask a follow-up? did they interrupt? did they say "that's wrong"?). Use this data to refine selection heuristics over time.

**Phase to address:** Phase 3 (backend routing). Get one backend working end-to-end first, then add the second with routing.

**Severity:** MODERATE

---

### Pitfall 14: Barge-In System Breaks Under Always-On Semantics

**What goes wrong:**
The current barge-in system (lines 2699-2809) assumes a clean turn-taking model: AI speaks, user interrupts, AI stops and listens. In always-on mode, "interruption" is ambiguous:
- User coughs while AI is speaking -- is this a barge-in or throat clearing?
- Another person in the room speaks -- barge-in or ambient noise?
- User says "Russel, stop" -- barge-in (should stop) or name mention (should listen)?
- User laughs at something on TV while AI is explaining -- barge-in or reaction?

The VAD-based barge-in (6 consecutive speech chunks, line 2281) will fire on any sustained audio during AI speech. In a room with ambient noise or other people, this creates frequent false barge-ins, constantly interrupting the AI mid-sentence.

**Prevention:**
- Increase barge-in sensitivity during always-on mode. Require higher sustained speech threshold (10+ chunks instead of 6) or higher VAD probability threshold (0.7 instead of 0.5).
- Use PipeWire AEC output for barge-in detection. After echo cancellation, the residual audio should only contain non-AI speech. If the AEC-processed audio has speech energy, it's genuinely someone else speaking.
- Add a "confidence cooldown" -- if barge-in fired but the subsequent STT produced no meaningful transcript (hallucination or very short), increase the barge-in threshold temporarily. This adapts to noisy environments.
- Consider: in always-on mode, "barge-in" during proactive AI speech should be easier (lower threshold) than during user-requested AI speech. If the user asked a question, they probably want to hear the answer. If the AI interjected proactively, the user should be able to shut it up easily.

**Phase to address:** Phase 3 (barge-in refinement). The existing system works for PTT mode; always-on refinements come after basic always-on is working.

**Severity:** MODERATE

---

### Pitfall 15: Ollama Concurrency -- Monitoring and Response Can't Run Simultaneously

**What goes wrong:**
Ollama's default `OLLAMA_NUM_PARALLEL=1` means it processes one request at a time. If the monitoring LLM is running inference (deciding whether to respond) and simultaneously a response is being generated via Ollama, the second request queues. In the worst case: monitoring decides "respond now" but the response request waits for the monitoring inference to finish -- adding the monitoring LLM's full inference time to the response latency.

If both monitoring and response use the same model (Llama 3.2 3B), this is a single-model concurrency issue. If they use different models, Ollama may need to swap models (evicting one to load the other), which is even slower.

**Prevention:**
- Use the same model for both monitoring and response via Ollama. Set `OLLAMA_NUM_PARALLEL=2` to allow concurrent requests to the same model. This doubles the context memory requirement (2x 2048 = 4096 tokens extra VRAM) but enables parallelism.
- Better: make monitoring and response sequential by design. The monitoring LLM decides first (fast, ~200ms), then the response pipeline starts. Don't run them concurrently. This is simpler and avoids the concurrency issue entirely.
- If using different backends for monitoring (Ollama) and response (Claude CLI), there's no Ollama concurrency issue -- they're different processes.

**Phase to address:** Phase 2 (monitoring architecture).

**Severity:** MODERATE

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

---

### Pitfall 16: Configuration Complexity Explosion

**What goes wrong:**
Always-on mode introduces many new configuration options: listening mode (PTT vs always-on), monitoring model, response backend, name variants, confidence thresholds, echo cancellation settings, VRAM budget, context window size, keepalive duration, barge-in sensitivity, privacy retention period, etc. The current `config.json` has 16 keys. Adding 10-15 more makes configuration overwhelming and error-prone. Users misconfigure one setting and the whole system behaves unexpectedly with no clear error message.

**Prevention:**
- Use a "profile" system: `"mode": "ptt"` (current behavior, all existing defaults) vs `"mode": "always_on"` (new defaults tuned for always-on). Individual settings can still be overridden, but the profile provides sensible defaults.
- Validate configuration at startup. If `mode: always_on` but PipeWire AEC is not configured, warn. If Ollama is not running, warn. If VRAM is insufficient, warn.
- Keep the Settings UI (in `indicator.py`) focused on the most important options. Advanced settings stay config.json-only.

**Phase to address:** Phase 1 (config structure). Design the config schema before building features.

**Severity:** MINOR

---

### Pitfall 17: Conversation and Interview Modes Break Under Always-On Refactoring

**What goes wrong:**
The codebase supports multiple modes: live, dictate, conversation, interview. The pipeline refactoring for always-on mode changes fundamental assumptions (continuous STT, monitoring layer, response routing). If these changes aren't properly isolated behind the `listening_mode` flag, they leak into other modes -- breaking dictation accuracy, interview flow, or conversation session management.

**Prevention:**
- All always-on behavior must be gated on a clear `listening_mode == "always_on"` check. The existing modes use the same pipeline entry points -- changes to `_stt_stage` or `_llm_stage` must preserve existing behavior when always-on is disabled.
- Run the full existing test suite after every always-on change. The 96KB `test_live_session.py` covers the existing pipeline extensively -- any regression will be caught if tests are run.
- Consider: the `InterviewSession` and `ConversationSession` classes (referenced in CLAUDE.md) have their own lifecycle management. Always-on should be a separate session type, not a modification of existing session types.

**Phase to address:** Every phase (continuous regression testing).

**Severity:** MINOR

---

### Pitfall 18: Dashboard and Event Bus Overwhelmed by Continuous Events

**What goes wrong:**
The event bus (JSONL file) and SSE dashboard currently log discrete events: STT start/complete, LLM send/receive, barge-in, etc. In always-on mode, STT events fire continuously (every 1-3 seconds even during silence due to hallucination filter rejections). Audio RMS events already fire per-chunk (~12 per second). Adding monitoring LLM decision events (every few seconds), the JSONL file grows rapidly and the SSE dashboard receives a firehose of events that's impossible to read.

**Prevention:**
- Rate-limit event bus writes for continuous monitoring events. Log monitoring decisions only when they change state (from "don't respond" to "respond"), not every evaluation.
- Add an event type category: "ephemeral" events (RMS, per-chunk status) vs "significant" events (STT complete, monitoring decision, response start). Dashboard shows significant events by default.
- Rotate JSONL files by size or time (every hour or every 10MB). Don't let a single session file grow to gigabytes.

**Phase to address:** Phase 2 (event bus extension).

**Severity:** MINOR

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Audio architecture (Phase 1) | Feedback loop (#1) | PipeWire AEC + transcript fingerprinting. Must be first thing built. |
| Audio architecture (Phase 1) | Hallucination explosion (#3) | Aggressive VAD, expanded filter list, confidence gating. |
| Resource budgeting (Phase 1) | GPU memory exhaustion (#2) | Measure actual VRAM with both models loaded. Consider smaller Whisper model. |
| Pipeline refactor (Phase 1) | PTT architecture assumptions (#10) | Transcript dispatcher pattern. Don't break existing modes. |
| Config/privacy (Phase 1) | Privacy boundary (#9) | Persistent indicator, data retention policy, mute zones. |
| Config/privacy (Phase 1) | Config explosion (#16) | Profile-based defaults (PTT vs always-on). |
| Monitoring LLM (Phase 2) | Context window growth (#4) | Two-tier context: rolling summary + recent buffer. |
| Monitoring LLM (Phase 2) | False positive responses (#5) | Default to NOT responding. Name-based activation first. |
| Monitoring LLM (Phase 2) | Ollama cold start (#7) | keep_alive=-1, heartbeat keepalive, pre-load on start. |
| Name detection (Phase 2) | STT-based name detection accuracy (#6) | Fuzzy matching, initial_prompt bias, consider openWakeWord. |
| Latency optimization (Phase 3) | End-to-end response latency (#8) | Quick response bridge, speculative pre-processing, parallel stages. |
| Backend routing (Phase 3) | Selection heuristic (#13) | Simple rule first (name=Claude, ambient=Ollama). Refine with data. |
| Barge-in refinement (Phase 3) | Always-on barge-in semantics (#14) | Use AEC output, increase threshold, context-aware sensitivity. |
| Reliability (Phase 3) | Memory leak (#12) | Monitor RSS, periodic model reload, soak testing. |
| Integration (All phases) | Regression in existing modes (#17) | Run full test suite on every change. |

---

## Sources

- [OpenAI Community: Realtime API starts to answer itself](https://community.openai.com/t/realtime-api-starts-to-answer-itself-with-mic-speaker-setup/977801) -- feedback loop documentation
- [PipeWire: Echo Cancel Module](https://docs.pipewire.org/page_module_echo_cancel.html) -- official AEC configuration
- [arXiv 2501.11378: Whisper ASR Hallucinations Induced by Non-Speech Audio](https://arxiv.org/abs/2501.11378) -- 40.3% hallucination rate on non-speech
- [arXiv 2505.12969: Calm-Whisper](https://arxiv.org/html/2505.12969v1) -- Whisper hallucination analysis
- [GitHub: openai/whisper Discussion #679](https://github.com/openai/whisper/discussions/679) -- hallucination mitigations
- [faster-whisper: High Memory Use Issue #249](https://github.com/guillaumekln/faster-whisper/issues/249) -- memory leak in long sessions
- [Ollama FAQ](https://docs.ollama.com/faq) -- keep_alive, context window, parallel requests
- [Ollama VRAM Requirements Guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) -- model memory sizing
- [GitHub: ollama/ollama Issue #9410](https://github.com/ollama/ollama/issues/9410) -- keep_alive GPU loading bug
- [RTX2060 Ollama Benchmark](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx2060) -- Llama 3.2 3B at 50 tokens/s
- [Stanford HAI: Teaching a Voice Assistant When to Speak](https://hai.stanford.edu/news/it-my-turn-yet-teaching-voice-assistant-when-speak) -- turn-taking research
- [Proactive behavior in voice assistants (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2451958824000447) -- proactive response research
- [Picovoice: Benchmarking Wake Word Detection](https://picovoice.ai/blog/benchmarking-a-wake-word-detection-engine/) -- <5% FRR at 1 FAR/10h
- [openWakeWord](https://github.com/dscripka/openWakeWord) -- open-source wake word framework
- [Context Window Management Strategies (getmaxim.ai)](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/) -- sliding window + summary patterns
- [Deepgram: Voice Agent Echo Cancellation](https://developers.deepgram.com/docs/voice-agent-echo-cancellation) -- AEC approaches
- Existing codebase: `live_session.py`, `push-to-talk.py`, `event_bus.py`, `stream_composer.py` (line references throughout)
