# Domain Pitfalls: Adaptive Quick Response Library

**Domain:** AI-driven quick response selection for voice conversation system
**Researched:** 2026-02-18
**Confidence:** HIGH (verified against existing codebase, academic research, industry benchmarks)

---

## Critical Pitfalls

Mistakes that cause rewrites or fundamentally break the feature.

---

### Pitfall 1: Classification Latency Exceeds the Filler Window

**What goes wrong:**
The entire value proposition of quick responses is speed -- they play *before* the LLM responds. The current `_filler_manager` has a 500ms gate (`asyncio.wait_for(cancel_event.wait(), timeout=0.5)`), meaning the filler must be selected and begin playing within that 500ms window. If classification takes 200ms+ on top of the ~100ms already spent on STT finalization + transcript delivery, the classification result arrives too late. The filler either doesn't play (LLM response already cancelled it) or plays simultaneously with the LLM's first TTS chunk, creating audio collision.

Industry benchmarks confirm: the human-perceivable response window is 200-300ms. A "quick" response that arrives at 400ms feels the same as no response at all -- the user has already noticed the silence gap.

**Why it happens:**
Developers naturally reach for LLM-based classification ("ask Claude to classify the intent"). Even a fast model like Haiku adds 300-800ms of API latency (network round-trip + inference). Local transformer models (DistilBERT, TinyBERT) need ~50-200ms depending on GPU availability. On a CPU-only path, even small models can take 100-300ms. Meanwhile, the Whisper transcription that produces the input text already consumed the latency budget.

**How to avoid:**
Classification must complete in under 50ms. This rules out any network call and most ML model inference. Use one of these approaches in order of preference:

1. **Pattern matching / keyword lookup** (~1ms): A dictionary mapping keywords/phrases to response categories. "What's your name" -> greeting, "thanks" -> acknowledgment, "wow" -> reaction. This is the only approach that reliably fits the timing budget.

2. **Lightweight local classifier** (~5-20ms): A pre-trained scikit-learn model (TF-IDF + logistic regression or naive Bayes) loaded at startup. Inference on short text is microseconds. The model file is tiny (<1MB).

3. **Embedding similarity** (~10-30ms): Pre-compute embeddings for each response category. At runtime, embed the user text with a tiny model (e.g., sentence-transformers/all-MiniLM-L6-v2 quantized) and do cosine similarity. Only viable if the embedding model is already loaded in memory.

Never use: Claude API, any HTTP call, loading a model from disk at classification time, or any approach that requires GPU warmup.

**Warning signs:**
- `[timing]` logs showing filler played after first LLM text token
- Filler cancel event fires before clip starts playing
- Users report fillers never playing or always playing the same generic one

**Detection metric:**
Log `classification_ms` alongside existing `[timing]` data. Alert if p95 exceeds 30ms.

**Phase to address:** Phase 1 -- this is the core design decision. Getting classification speed wrong means the entire quick response system is useless.

**Severity:** CRITICAL

---

### Pitfall 2: Quick Response Collides with LLM's First Audio Chunk

**What goes wrong:**
A quick response clip starts playing, but the LLM responds faster than expected. The current code cancels fillers via `self._filler_cancel.set()` when the first `text_delta` arrives (line 1182). But the clip is already partially queued in `_audio_out_q` as `FrameType.FILLER` frames. The playback stage (line 1715) treats FILLER and TTS_AUDIO the same way -- it writes them sequentially to the PyAudio stream. Result: the tail end of the quick response clip overlaps with the beginning of the LLM's actual response. The user hears "Sure, let me--Hello! My name is..."

This is worse than the current random filler problem because contextual responses set up stronger expectations. Hearing "That's really funny--" cut off by "The weather today is..." sounds broken, not just awkward.

**Why it happens:**
The `_play_filler_audio` method checks `cancel_event.is_set()` per-chunk (4096 bytes = ~85ms of audio at 24kHz/16-bit), but chunks already pushed to `_audio_out_q` are not retracted. The playback stage doesn't check for cancellation -- it just plays whatever's in the queue. With a quick response clip of ~1-2 seconds, multiple chunks may already be queued before cancellation fires.

**How to avoid:**
Two-layer cancellation:

1. **Source-side** (existing): `_play_filler_audio` stops pushing new chunks when cancel fires. Already works.

2. **Sink-side** (missing): The playback stage must drain FILLER frames from `_audio_out_q` when cancellation occurs. Add a drain step in the same place that handles `_filler_cancel.set()` -- after line 1183, drain all pending FILLER frames:
   ```python
   # Drain queued filler frames
   drained = []
   while not self._audio_out_q.empty():
       try:
           f = self._audio_out_q.get_nowait()
           if f.type != FrameType.FILLER:
               drained.append(f)  # Keep non-filler frames
       except asyncio.QueueEmpty:
           break
   for f in drained:
       await self._audio_out_q.put(f)
   ```

3. **Fade-out on cancel**: When the playback stage receives a FILLER frame but `_filler_cancel` is set, apply a fast fade-out (20ms ramp to zero) to the current chunk and discard remaining FILLER frames. This prevents the hard cut that sounds like a glitch.

**Warning signs:**
- Users hear partial quick responses followed immediately by LLM speech
- Audio artifacts (clicks/pops) at the transition point
- The overlapping audio sounds like two people talking at once

**Phase to address:** Phase 1 -- must be solved as part of the core quick response playback mechanism.

**Severity:** CRITICAL

---

### Pitfall 3: Over-Classifying Intent Creates a Brittle Taxonomy

**What goes wrong:**
The developer creates 30+ response categories: greeting, farewell, thanks, apology, agreement, disagreement, surprise, confusion, humor, sympathy, encouragement, clarification-request, topic-change, self-reference, meta-question, emotional-expression, and so on. Each category has 3-5 clips. The classifier struggles to distinguish "surprise" from "confusion" or "agreement" from "acknowledgment." Result: responses feel *almost right* but slightly off, which is worse than random -- it suggests the AI heard and understood the user but chose to respond weirdly.

**Why it happens:**
Developers think more categories = more nuanced responses = better UX. But intent classification accuracy degrades roughly logarithmically with category count. A 5-category classifier might hit 90%+ accuracy. A 30-category classifier on short informal speech transcripts might hit 60%. And 60% accuracy on "contextually appropriate" responses means 40% of the time the response feels *wrong*, not just *generic*.

**How to avoid:**
Start with 5-7 broad categories maximum:

| Category | Trigger Examples | Response Examples |
|----------|-----------------|-------------------|
| acknowledgment | (default/fallback) | "Sure" "Got it" "Let me think" |
| greeting | "hi" "hello" "hey" "what's up" | "Hey!" "Hi there" |
| reaction | "wow" "cool" "nice" "whoa" | "Right?" "I know!" |
| thanks | "thank you" "thanks" "appreciate" | "Of course" "Happy to help" |
| farewell | "bye" "goodbye" "see you" "later" | "See you!" "Take care" |

Use "acknowledgment" as the fallback -- if classification confidence is low, play an acknowledgment. This is safe because acknowledgments are socially appropriate in any context (the current behavior, just contextually selected).

Add new categories only when you have evidence (from usage logs) that a common situation lacks a good response. Never add a category without at least 5 distinct clips and a clear discrimination signal.

**Warning signs:**
- Classification confidence consistently below 0.7
- Users commenting that responses feel "off" or "weird" more than before
- Multiple categories with overlapping trigger words
- Adding a new category degrades accuracy on existing categories

**Phase to address:** Phase 1 design, but revisitable. Start minimal, expand based on data.

**Severity:** CRITICAL

---

### Pitfall 4: Non-Speech Event Detection Misclassifies Speech

**What goes wrong:**
The system tries to detect coughs, sighs, and laughter to play appropriate responses ("bless you", empathetic hmm, laughing along). But the non-speech detector triggers on actual speech that happens to have similar acoustic properties -- breathy speech gets classified as a sigh, a chuckle mid-sentence gets classified as laughter, throat-clearing before speaking gets classified as a cough. The AI responds to non-existent events, breaking conversational flow.

Research confirms this is a known hard problem: Whisper's `no_speech_prob` has a 40.3% hallucination rate on non-speech audio. Dedicated non-speech classifiers achieve 93.1% F1 at best, meaning ~7% of events are misclassified. In a conversation with dozens of utterances per session, 7% error rate means multiple false triggers per session.

**Why it happens:**
Non-speech sounds and speech exist on a continuum. A laugh can transition into speech mid-sound. A cough can precede or interrupt a word. Short breathy utterances ("huh", "hmm") are acoustically indistinguishable from sighs. The existing Whisper segment filtering (lines 782-817) already rejects segments with `no_speech_prob >= 0.6` or `avg_logprob < -1.0`, but these thresholds were tuned to reject hallucinations, not to positively detect non-speech events.

**How to avoid:**

1. **Defer non-speech detection to a later phase.** The quick response library provides immediate value with text-based classification alone. Adding acoustic classification multiplies the error surface.

2. **If implemented, require high confidence + low speech probability.** Don't fire on marginal signals. Require `no_speech_prob >= 0.85` AND a dedicated non-speech classifier confidence >= 0.9.

3. **Gate on conversation state.** Only attempt non-speech detection when the system is in "listening" mode and no speech has been detected in the current utterance. If Whisper produces any text at all, suppress non-speech classification for that segment.

4. **Make non-speech responses safe.** If the system does respond to a detected cough/laugh, use a response that's also appropriate if the detection was wrong -- a brief empathetic "hmm" rather than "bless you!" which would be confusing if the user didn't cough.

**Warning signs:**
- Quick responses firing in the middle of user speech
- Users asking "why did you say that?" after a false trigger
- Non-speech events detected on almost every utterance

**Phase to address:** Phase 3 or later. Text-based classification is the safe starting point.

**Severity:** CRITICAL (if implemented early; drops to MINOR if properly deferred)

---

## Major Pitfalls

Mistakes that cause significant rework or degrade UX.

---

### Pitfall 5: Uncanny Valley -- Contextually "Correct" but Emotionally Wrong

**What goes wrong:**
The classifier correctly identifies "my dog died" as a sad statement and plays a sympathetic response clip. But the Piper TTS voice that generated the clip sounds flat and robotic. The *content* is right ("I'm sorry to hear that") but the *delivery* is emotionally tone-deaf. Research shows this inconsistency between intelligence (correct classification) and voice quality (robotic delivery) triggers the uncanny valley more strongly than a consistently robotic response.

Similarly, the quick response clip might have different prosody than the LLM's subsequent full response -- the "I'm so sorry" clip sounds one way, then the LLM's continuation "That must be really hard" sounds slightly different in pacing, pitch, and energy. The voice appears to change personality mid-sentence.

**Why it happens:**
Piper TTS (en_US-lessac-medium model) produces intelligible but not emotionally expressive speech. It doesn't convey genuine sympathy, excitement, or humor through prosody. The `clip_factory.py` randomizes `length_scale`, `noise_w_scale`, and `noise_scale` for variety, but these parameters control speaking rate and vocal texture, not emotion.

Pre-generated clips also have a fixed acoustic signature from their generation parameters. The LLM's response, also generated by Piper TTS but with potentially different parameters (or different text length affecting prosody), will sound subtly different.

**How to avoid:**

1. **Avoid emotionally loaded categories entirely.** Don't try to sound sympathetic, excited, or humorous through pre-generated clips. Stick to neutral/functional categories: acknowledgments ("got it"), greetings ("hey"), brief reactions ("right"). These don't require emotional nuance.

2. **Normalize generation parameters.** Quick response clips and LLM TTS should use identical Piper parameters (`length_scale`, `noise_w_scale`, `noise_scale`). The `clip_factory.py` currently randomizes these (lines 73-79). For quick response clips, fix them to the exact same values used by `_tts_to_pcm` in `live_session.py` (which uses Piper defaults -- no custom parameters passed on line 580).

3. **Match audio characteristics.** Both quick response clips and live TTS use Piper at 22050Hz resampled to 24000Hz via `_resample_22050_to_24000`. Ensure the same resampling is applied to quick response clips at load time (the current `_load_filler_clips` on line 516 already does this for acknowledgment clips).

4. **Test transitions.** Play a quick response clip followed immediately by a Piper TTS sentence. Listen for jarring quality/prosody shifts. If audible, the quick response value is negative.

**Warning signs:**
- Quick response sounds noticeably different from the LLM's subsequent speech
- Users describe the AI as "two different voices"
- Emotional responses feel hollow or creepy

**Phase to address:** Phase 1 (parameter normalization), Phase 2 (category selection strategy).

**Severity:** MAJOR

---

### Pitfall 6: Library Bloat -- Too Many Clips Slow Startup and Complicate Selection

**What goes wrong:**
The developer generates 20+ clips per category across 10+ categories = 200+ WAV files. Loading all clips into memory at startup (as `_load_filler_clips` does -- line 516 converts each file to raw PCM bytes in a dict) consumes significant memory and slows initialization. More critically, having many similar clips within a category makes the "no-repeat guard" (`_pick_filler` on line 545, which only prevents consecutive repeats) insufficient -- users still hear the same few clips cycling because short-term memory spans ~5 items. And the selection logic (random choice with no-repeat) doesn't account for clip appropriateness within a category.

The `clip_factory.py` currently caps at `ACK_POOL_SIZE_CAP = 15` clips. If the quick response system has 7 categories x 15 clips = 105 clips, each ~1-2 seconds of 24kHz 16-bit PCM = ~48-96KB per clip. Total: ~5-10MB in memory. Not catastrophic, but wasted if most clips never play.

**Why it happens:**
The intuition "more variety = more natural" is correct up to a point, but the current random selection means adding more clips doesn't improve perceived variety -- it just increases memory usage and makes quality control harder. With 15 clips per category, the chance of hearing a repeat within 5 plays is still ~33% (birthday paradox).

**How to avoid:**

1. **Cap at 5-8 clips per category.** This is the sweet spot for perceived variety. With 7 categories x 7 clips = 49 clips, memory is ~2.5MB.

2. **Use round-robin, not random.** Shuffle the clips for a category at startup, then cycle through them sequentially. Reset shuffle on each full cycle. This guarantees no repeats until all clips have played. The current `_last_filler` approach (prevent consecutive repeats) is not sufficient.

3. **Lazy loading.** Don't load all categories at startup. Load "acknowledgment" (the default/fallback) eagerly. Load other categories on first use. Most sessions may only ever trigger 2-3 categories.

4. **Separate pool sizes from clip factory.** The `clip_factory.py` manages clip generation and quality. The live session manages clip selection. Keep these concerns separate -- the factory can generate up to 15 per category (to have reserves), but the session only loads the best 7.

**Warning signs:**
- Session startup takes noticeably longer after adding categories
- Memory usage climbs with each new category
- Users report hearing the same clip repeatedly despite a large library
- Quality variation within a category (some clips sound great, some sound robotic)

**Phase to address:** Phase 1 (initial pool sizing), Phase 2 (selection algorithm upgrade).

**Severity:** MAJOR

---

### Pitfall 7: Cold Start -- Empty Library Produces Silence or Crashes

**What goes wrong:**
On first run, or after a library reset/corruption, no clips exist for any category. The classifier selects "greeting" but there are no greeting clips. The code falls through to the generic acknowledgment pool, which is also empty. Result: either silence (no filler plays at all, which is the current behavior when `fillers_enabled = False`), or a crash if the code doesn't handle `None` from clip selection.

A subtler version: the clip factory daemon generates clips asynchronously, but the user starts a conversation before generation completes. The factory has generated 3 of 10 clips. The system works, but with minimal variety for potentially several minutes.

**Why it happens:**
The current `clip_factory.py` runs as a one-shot top-up (`top_up_ack_pool`) or daemon. The existing acknowledgment pool takes ~30-60 seconds to generate 10 clips (each Piper call takes ~1-3 seconds plus quality evaluation). A quick response library with 7 categories needs 7x that time = 3.5-7 minutes for initial population. If each category is generated by an independent factory call, there's a long window where some categories are empty.

**How to avoid:**

1. **Graceful fallback chain.** If the specific category (e.g., "greeting") has no clips, fall back to "acknowledgment". If "acknowledgment" has no clips, fall back to no filler (current behavior -- not ideal but not broken). Never crash.

   ```python
   def _pick_quick_response(self, category: str) -> bytes | None:
       clip = self._pick_filler(category)
       if clip is None and category != "acknowledgment":
           clip = self._pick_filler("acknowledgment")
       return clip  # None is acceptable -- means no quick response
   ```

2. **Ship a seed library.** Include 2-3 pre-generated clips per category in the git repo under `audio/fillers/`. These are the baseline that always exists. The clip factory adds variety over time but the system is functional from first launch.

3. **Prioritize generation.** When generating clips for the first time, generate 1 clip per category before generating the remaining clips for any category. This ensures broad coverage quickly rather than deep coverage of one category while others are empty.

4. **Signal readiness.** Track which categories have at least 1 clip. Log a warning if a session starts with empty categories. Don't enable contextual classification for categories without clips.

**Warning signs:**
- First-time users never hearing quick responses
- Categories that "work sometimes but not always" (race condition with factory)
- Error logs showing `None` clip selection with no fallback

**Phase to address:** Phase 1 -- fallback chain and seed library. Phase 2 -- factory prioritization.

**Severity:** MAJOR

---

### Pitfall 8: Pipeline Timing -- Classification Runs After the Filler Window Closes

**What goes wrong:**
The current filler system triggers in `_filler_manager`, which is spawned as an `asyncio.create_task` on line 1532, immediately after the transcript is available. The 500ms gate starts from this spawn point. But if classification is added *inside* the filler manager, the sequence becomes:

1. Transcript arrives (t=0ms)
2. Filler task spawned (t=0ms)
3. Classification runs (t=0ms to t=50ms)
4. Clip selected (t=50ms)
5. 500ms gate starts *after* classification? Or gate starts at spawn?

If the gate starts at spawn and classification runs in parallel with the gate, the timing works. But if classification must complete before the gate starts (to know which clip to load), the effective window shrinks from 500ms to 450ms. Worse: if classification depends on the transcript text, and the transcript arrives just before the gate expires, there's a race condition.

**Why it happens:**
The current system is simple: spawn filler task -> wait 500ms -> play random clip. Adding classification creates a dependency: classify -> select clip -> wait remaining gate time -> play. The "wait remaining gate time" must account for classification duration, or the clip plays too early (before the gate) or too late (after the LLM response).

**How to avoid:**
Classification and gate waiting must be parallel, not sequential:

```python
async def _filler_manager(self, user_text: str, cancel_event: asyncio.Event):
    # Start classification immediately (non-blocking, < 50ms)
    category = self._classify_quick_response(user_text)
    clip = self._pick_filler(category)

    if clip is None:
        clip = self._pick_filler("acknowledgment")
    if clip is None:
        return

    # Gate: wait remaining time of 500ms window
    # Classification already consumed some of that time
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=0.5)
        return  # LLM responded fast, skip filler
    except asyncio.TimeoutError:
        pass

    if cancel_event.is_set():
        return

    await self._play_filler_audio(clip, cancel_event)
```

The key insight: classification (1-50ms) happens *before* the 500ms gate, consuming negligible time from the budget. The gate is the dominant delay. This works because the classification is so fast it doesn't affect the overall timing. If classification were slow (200ms+), you'd need to run it in parallel with the gate and use `asyncio.gather` or `asyncio.wait`.

**Warning signs:**
- Fillers playing at inconsistent delays (sometimes fast, sometimes slow)
- Classification logging shows high variance in execution time
- Gate timeout fires before clip selection completes

**Phase to address:** Phase 1 -- this is the core timing architecture.

**Severity:** MAJOR

---

### Pitfall 9: Barge-In Interrupts a Quick Response, Creating an Awkward Trailing Clip

**What goes wrong:**
User says "hey". AI plays greeting quick response "Hi there!" (1.2 seconds of audio). User immediately barges in with a follow-up question at 400ms into the clip. `_trigger_barge_in` fires (line 1787): increments `generation_id`, drains queues, plays a trailing acknowledgment clip (150ms fade-out, lines 1833-1845).

But the trailing clip is *another* acknowledgment filler -- it doesn't know a *greeting* clip was just interrupted. The user hears: "Hi th--" [cut] "mmhmm" [acknowledgment trail]. This sounds wrong -- the greeting was cut and replaced with an unrelated sound.

Worse: the barge-in annotation (line 1854) says "They heard up to: 'Hi there!'" -- but the AI's quick response wasn't a sentence the LLM produced. The LLM gets a confusing annotation about speech it never generated.

**Why it happens:**
The barge-in system was designed for LLM responses, not quick response clips. It tracks `_spoken_sentences` (what the LLM said) and builds annotations based on that. Quick response clips exist outside this tracking -- they're `FrameType.FILLER`, not `FrameType.TTS_AUDIO`, and they don't appear in `_spoken_sentences`.

**How to avoid:**

1. **Track quick response clips in barge-in state.** When a quick response starts playing, record what's playing:
   ```python
   self._current_quick_response = "Hi there!"  # The text of the playing clip
   ```

2. **Customize barge-in behavior for quick responses.** If the user barges in during a quick response (not during LLM TTS), skip the trailing acknowledgment -- the quick response *was* the acknowledgment. Just cut to silence and start listening.

3. **Don't include quick response text in barge-in annotations.** The LLM didn't say it. The annotation should reflect that:
   ```python
   if self._current_quick_response:
       self._barge_in_annotation = "[The user interrupted a brief filler response. They are speaking now.]"
   else:
       # Existing annotation for LLM responses
   ```

4. **Consider making quick response clips shorter.** A 0.5-second clip is harder to barge into than a 1.5-second clip. The current acknowledgment clips range from 0.3-4.0 seconds (clip_factory.py line 145). Cap quick response clips at 1.0 second.

**Warning signs:**
- Barge-in annotations mentioning text the LLM never said
- Trailing acknowledgment clips playing after quick responses are interrupted
- Users hearing two filler-type clips in rapid succession

**Phase to address:** Phase 2 -- after basic quick responses work, refine barge-in integration.

**Severity:** MAJOR

---

## Moderate Pitfalls

Mistakes that cause delays, confusion, or technical debt.

---

### Pitfall 10: Audio Quality Discontinuity Between Pre-Generated and Live TTS

**What goes wrong:**
Pre-generated clips in the library were created with specific Piper parameters at a specific time. Live TTS responses are generated on the fly with potentially different parameters. Even with the same model and parameters, Piper's output can vary slightly based on input text length (short phrases vs. long sentences produce different prosodic patterns). The transition from a quick response clip to the LLM's live TTS response has an audible "seam" -- volume difference, tonal shift, or pacing change.

**Prevention:**
- Use identical Piper model, version, and default parameters for both quick response generation and live TTS.
- Apply the same resampling pipeline (22050Hz -> 24000Hz) to both.
- Normalize volume (RMS-based) across all clips and live TTS output to a target level.
- Test by playing clip + TTS sentence back-to-back and listening for discontinuity.
- Store Piper version and parameters in clip metadata (already partially done in `clip_factory.py` line 291) and reject clips generated with different parameters on load.

**Phase to address:** Phase 1 (parameter normalization), Phase 3 (volume normalization).

**Severity:** MODERATE

---

### Pitfall 11: Personality Drift -- Library Evolves Away from AI's Character

**What goes wrong:**
If the library growth mechanism uses an LLM to generate new response phrases, the phrasing can drift from the AI's established personality over time. The personality is defined in `personality/*.md` files and loaded as system prompt. But the clip factory generates phrases from a hardcoded list (`ACKNOWLEDGMENT_PROMPTS`, line 44). If a new "smart growth" feature generates phrases dynamically, those phrases may not match the personality's speaking style, vocabulary, or tone.

Example: the personality is casual and uses slang ("yo, on it"), but the clip factory generates formal responses ("Certainly, I'll look into that immediately").

**Prevention:**
- When generating new response phrases, include the personality prompt as context so the LLM generates phrases consistent with the character.
- Validate new phrases against the existing clip library for stylistic consistency (ask the LLM: "Does this phrase match the speaking style of these existing phrases?").
- Cap automatic library growth at a rate that allows human review (e.g., max 2 new clips per session, logged for review).
- Include a `personality_hash` in clip metadata. When the personality changes, flag clips generated under the old personality for review/regeneration.

**Phase to address:** Phase 3 (library growth/pruning features).

**Severity:** MODERATE

---

### Pitfall 12: Classification Overfits to Transcript Artifacts

**What goes wrong:**
Whisper transcription of informal speech produces artifacts: dropped words, hallucinated punctuation, inconsistent capitalization, filler words ("um", "uh") included or excluded unpredictably. A classifier trained on clean text ("What is your name?") fails on real transcripts ("what's your uh name"). Keyword-based matching misses contractions ("what's" vs "what is"), misspellings Whisper produces on unusual words, or transcription of names/proper nouns.

The existing Whisper configuration uses `condition_on_previous_text=False` (line 779), which reduces hallucination but also means each utterance is transcribed independently without context -- leading to more ambiguous short transcripts.

**Prevention:**
- Normalize text before classification: lowercase, strip punctuation, remove common fillers ("um", "uh", "like", "you know").
- Use fuzzy matching or stemming rather than exact string matching.
- Test classification on *actual Whisper transcripts* from real sessions, not on clean typed text.
- Log classification inputs alongside results to build a corpus of real-world transcripts for testing.
- Handle the empty/very-short transcript case: utterances like "hmm" or "uh huh" should map to acknowledgment, not trigger a classification error.

**Phase to address:** Phase 1 (text normalization), Phase 2 (testing against real transcripts).

**Severity:** MODERATE

---

### Pitfall 13: Classification Runs But No Quick Response Is Warranted

**What goes wrong:**
Not every user utterance warrants a quick response. For long, complex questions, a quick response is appropriate (buys time while the LLM thinks). But for very short exchanges in rapid conversation, quick responses add noise. Example: user says "yeah" in response to the AI -- the AI should just process this as input, not play "Got it!" as a quick response before its actual reply. The user hears: "Got it! ... Yes, exactly, so as I was saying..."

The current 500ms gate handles this somewhat -- if the LLM responds within 500ms, the filler is skipped. But classification happens before the gate, meaning the system does classification work that's wasted 30-50% of the time (for fast LLM responses).

**Prevention:**
- Make classification lazy: only classify if the gate is about to expire (close to the 500ms mark). If the LLM responds fast, skip classification entirely.
- Alternatively, classify eagerly but make the gate category-aware: some categories (like acknowledgment for long questions) should have a shorter gate (300ms), while others (like greeting) should play immediately (0ms gate -- greetings should be instant).
- Track a "conversation cadence" metric. In rapid back-and-forth, suppress quick responses. In slow exchanges (user pauses before speaking), enable them.
- Never play a quick response for utterances shorter than 3 words -- these are likely conversational fragments, not standalone inputs.

**Phase to address:** Phase 2 (conversation-aware gating).

**Severity:** MODERATE

---

### Pitfall 14: Resampling Artifacts From Sample Rate Mismatch

**What goes wrong:**
Piper TTS outputs at 22050Hz. The pipeline operates at 24000Hz. The current `_resample_22050_to_24000` method (line 1568) uses linear interpolation -- a simple but lossy algorithm that introduces high-frequency artifacts (aliasing) and slightly changes the perceived pitch/timbre. When a quick response clip (resampled at clip generation time by `save_clip_to` at 22050Hz, then loaded and resampled at runtime) transitions to live TTS audio (resampled once from 22050Hz to 24000Hz), any difference in resampling quality is audible.

The `clip_factory.py` saves clips at 22050Hz (line 36, `SAMPLE_RATE = 22050`), and `_load_filler_clips` reads them and applies `_resample_22050_to_24000` at load time (line 532). This means clips go through the same resampling pipeline as live TTS. But if the clip factory ever changes to pre-resample and save at 24000Hz, or if a different TTS engine is introduced at a different sample rate, the mismatch surfaces.

**Prevention:**
- Keep the single source of truth for resampling in `live_session.py`. Never pre-resample clips in the factory.
- If resampling quality becomes an issue, upgrade to a proper resampler (scipy.signal.resample or soxr) instead of the linear interpolation on line 1568. The CPU cost is negligible for short clips.
- Document the audio pipeline's sample rate contract: Piper outputs 22050Hz, pipeline operates at 24000Hz, all resampling happens at load time or TTS time, never at generation time.

**Phase to address:** Phase 1 (documentation), Phase 3 (resampling quality upgrade if artifacts are audible).

**Severity:** MODERATE

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

---

### Pitfall 15: Clip Metadata Gets Out of Sync With Clip Files

**What goes wrong:**
The quick response library uses a metadata file (like `ack_pool.json`) to track clip properties, generation parameters, and quality scores. If clips are manually added/removed, or if the metadata file is corrupted/deleted, the system's view of available clips diverges from reality. The `clip_factory.py` already handles this reconciliation (lines 243-248 reconcile metadata with disk), but a new per-category metadata system needs the same resilience.

**Prevention:**
- Apply the same reconciliation pattern from `clip_factory.py` to the quick response library: on load, scan the directory for WAV files and cross-reference with metadata. Remove orphaned metadata entries, load orphaned WAV files with default metadata.
- Use the filesystem as the source of truth, not the metadata file. A clip exists if and only if its WAV file exists. Metadata is supplementary.

**Phase to address:** Phase 1 (metadata design).

**Severity:** MINOR

---

### Pitfall 16: Logging and Debugging Quick Response Decisions

**What goes wrong:**
Without logging, it's impossible to tell why a particular quick response was selected or whether the classification was accurate. When a user reports "the AI said something weird", there's no trail to investigate. The current filler system has minimal logging -- just which clip played, not why.

**Prevention:**
- Log every classification decision: input text, classified category, confidence score, selected clip filename, time elapsed.
- Log when classification is skipped (LLM responded before gate expired).
- Log when fallback is used (classified category had no clips, fell back to acknowledgment).
- Store classification logs alongside the existing conversation JSONL log for post-session analysis.
- Format: `{"type": "quick_response", "text": "...", "category": "greeting", "confidence": 0.92, "clip": "hi_there_001.wav", "classification_ms": 12, "played": true}`

**Phase to address:** Phase 1 (essential for iteration).

**Severity:** MINOR

---

### Pitfall 17: Testing Quick Responses Requires Audio Playback

**What goes wrong:**
Unit tests can verify classification logic and clip selection, but can't verify the most important quality: does the response *sound right*? The uncanny valley, audio seams, and timing issues are perceptual -- they require listening. But the test suite (existing `test_live_session.py`) doesn't have audio playback infrastructure.

**Prevention:**
- Build a command-line test harness that takes a transcript and plays the selected quick response clip followed by a sample TTS sentence. This lets a developer listen to the full sequence in seconds.
- Record "golden" test sequences: transcript -> quick response clip -> TTS continuation. Use these for regression testing (automated comparison of audio waveforms, not perceptual quality, to catch unintended changes).
- Keep a `test_phrases.txt` with sample transcripts covering each category + edge cases. Run classification against all of them and verify categories match expectations.

**Phase to address:** Phase 1 (classification test harness), Phase 2 (audio sequence test tool).

**Severity:** MINOR

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Core classification + playback (Phase 1) | Classification latency exceeds budget (#1) | Use keyword/pattern matching, not ML. Benchmark early. |
| Core classification + playback (Phase 1) | Audio collision on cancel (#2) | Implement sink-side drain of FILLER frames. |
| Core classification + playback (Phase 1) | Over-classification (#3) | Start with 5 categories max. Expand only with data. |
| Core classification + playback (Phase 1) | Cold start (#7) | Ship seed clips. Implement fallback chain. |
| Core classification + playback (Phase 1) | Pipeline timing (#8) | Classify before gate, not during. Keep under 50ms. |
| Barge-in integration (Phase 2) | Quick response barge-in confusion (#9) | Track quick response state separately from LLM state. |
| Barge-in integration (Phase 2) | Unnecessary quick responses (#13) | Category-aware gating, suppress for short utterances. |
| Non-speech detection (Phase 3+) | False positive misclassification (#4) | Require very high confidence. Gate on conversation state. |
| Library growth/pruning (Phase 3+) | Personality drift (#11) | Include personality prompt in generation context. |
| Library growth/pruning (Phase 3+) | Library bloat (#6) | Cap at 7 clips/category. Use round-robin selection. |

---

## Sources

- [AssemblyAI: The 300ms Rule](https://www.assemblyai.com/blog/low-latency-voice-ai) -- latency thresholds for voice AI
- [Trillet: Voice AI Latency Benchmarks 2026](https://www.trillet.ai/blogs/voice-ai-latency-benchmarks) -- classification tiers
- [Talkdesk: How to Avoid the Uncanny Valley in Voice Design](https://www.talkdesk.com/blog/voice-design/) -- consistency between voice and intelligence
- [Sesame: Crossing the Uncanny Valley of Conversational Voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) -- strategic imperfection
- [Wayline: The Uncanny Valley of AI Voice](https://www.wayline.io/blog/ai-voice-uncanny-valley-imperfection) -- signal mismatch triggers
- [arXiv 2501.11378: Whisper ASR Hallucinations Induced by Non-Speech Audio](https://arxiv.org/abs/2501.11378) -- 40.3% hallucination rate on non-speech
- [arXiv 2505.12969: Calm-Whisper](https://arxiv.org/html/2505.12969v1) -- 3 of 20 decoder heads cause 75% of hallucinations
- [GitHub: openai/whisper Discussion #679](https://github.com/openai/whisper/discussions/679) -- community hallucination mitigations
- [GitHub: FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) -- non-speech event detection labels
- [Nonspeech7k Dataset (Rashid et al.)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/sil2.12233) -- non-speech sound classification
- [Medium: 8 ONNX Runtime Tricks for Low-Latency Python Inference](https://medium.com/@Modexa/8-onnx-runtime-tricks-for-low-latency-python-inference-baee6e535445) -- sub-200ms local inference
- Existing codebase: `live_session.py`, `clip_factory.py`, `pipeline_frames.py` (line references throughout)
