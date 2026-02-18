# Architecture: Adaptive Quick Responses Integration

**Domain:** Adaptive quick response system for existing voice pipeline
**Researched:** 2026-02-18
**Overall Confidence:** HIGH (codebase fully read, integration points verified in source)

## Executive Summary

The adaptive quick response system replaces the current random filler selection in `live_session.py` with a context-aware response library. The architecture adds three new components (input classifier, response library, library curator) while modifying two existing ones (filler manager in `live_session.py`, clip factory). The key insight from reading the codebase: the integration point is narrow and well-defined. The `_filler_manager()` method and `_llm_stage()` are the only places that need modification in the hot path. Library growth/pruning runs entirely outside the pipeline as a post-session process.

## Current Pipeline Architecture

```
                  +--------------+
                  | Audio Capture |  Stage 1: pw-record -> AUDIO_RAW frames
                  +------+-------+
                         | _audio_in_q (Queue, maxsize=100)
                         v
                  +--------------+
                  |   STT Stage  |  Stage 2: Whisper in executor -> TRANSCRIPT frames
                  | (+ VAD for   |  Also: silence detection, hallucination filtering,
                  |  barge-in)   |        STT gating during playback
                  +------+-------+
                         | _stt_out_q (Queue, maxsize=50)
                         v
                  +--------------+
                  |  LLM Stage   |  Stage 3: Claude CLI subprocess -> TEXT_DELTA frames
                  | (+ filler    |  Also: filler_manager starts here, barge-in annotation,
                  |  manager)    |        tool intent display, post-tool buffering
                  +------+-------+
                         | _llm_out_q (Queue, maxsize=50)
                         v
                  +--------------+
                  |  TTS Stage   |  Stage 4: Piper TTS -> TTS_AUDIO frames
                  |              |  22050Hz -> 24000Hz resampling
                  +------+-------+
                         | _audio_out_q (Queue, maxsize=200)
                         v
                  +--------------+
                  | Playback     |  Stage 5: PyAudio output
                  | Stage        |  Also: STT gating, sentence counting, delayed unmute
                  +--------------+
```

### Current Filler Flow (What Gets Replaced)

Reading `_llm_stage()` lines 1528-1559, the flow is:

1. TRANSCRIPT frame arrives from STT
2. `_filler_cancel = asyncio.Event()` created
3. `_filler_manager(transcript, cancel_event)` started as asyncio task
4. User message sent to CLI with `_send_to_cli()`
5. `_read_cli_response()` streams back text deltas, setting `_filler_cancel` on first text
6. After CLI response complete, `_filler_cancel.set()` forces cancellation

Inside `_filler_manager()` (lines 559-574):
1. Wait 500ms gate -- if LLM responds within 500ms, skip filler entirely
2. If gate expires, pick random acknowledgment clip via `_pick_filler("acknowledgment")`
3. Play clip via `_play_filler_audio()` which sends FILLER frames to `_audio_out_q`

The `_pick_filler()` method (lines 545-557) is purely random with a no-repeat guard. It has zero awareness of what the user said.

### Key Timing Constraint

The filler system has a 500ms gate. This means the input classifier must produce a result in well under 500ms. Whisper transcription already takes 200-600ms (it runs in a thread executor -- see line 1471). So by the time the TRANSCRIPT arrives at `_llm_stage`, the user has already been waiting 200-600ms. The filler gate adds another 500ms. Total silence before filler: 700-1100ms.

The classifier must run within that 500ms gate window, ideally in under 100ms to leave room for clip lookup and any pre-play processing. This rules out any LLM-based classification in the hot path.

## Recommended Architecture

### Component Diagram

```
  EXISTING                          NEW
  --------                          ---

  +---------------+
  | STT Stage     |
  | (Whisper)     |------TRANSCRIPT frame----+
  +---------------+                          |
                                             v
                              +-----------------------------+
                              | InputClassifier             |  NEW
                              | - keyword/pattern matching  |
                              | - non-speech event flags    |
                              | - situation categorization  |
                              +-------------+---------------+
                                            |
                                   ClassifiedInput
                                   {category, confidence,
                                    original_text, tags}
                                            |
  +---------------+                         v
  | LLM Stage     |          +-----------------------------+
  | (Claude CLI)  |          | ResponseLibrary             |  NEW
  | _filler_mgr   |<------  | - lookup(classification)    |
  | (MODIFIED)    |          | - returns best clip or None |
  +---------------+          | - in-memory index + disk    |
                              +-----------------------------+
                                            ^
                                            | loads/saves
                                            v
                              +-----------------------------+
                              | LibraryCurator              |  NEW
                              | - post-session analysis     |  (daemon, like learner.py)
                              | - gap identification        |
                              | - clip generation           |
                              | - quality pruning           |
                              +-----------------------------+
```

### New Components

#### 1. InputClassifier (in-process, synchronous)

**Location:** New file `input_classifier.py`, instantiated in `LiveSession.__init__()`

**Why synchronous:** Must complete in <100ms. This is a pure-Python pattern matcher, not an LLM call. The current STT already provides the text; classification is string analysis.

```python
@dataclass
class ClassifiedInput:
    """Result of classifying user input for quick response selection."""
    category: str          # "question", "command", "emotional", "social",
                           # "acknowledgment", "non_speech", "unknown"
    subcategory: str       # "greeting", "farewell", "thanks", "cough",
                           # "sigh", "laugh", "task_request", "info_query", etc.
    confidence: float      # 0.0 - 1.0
    original_text: str     # The raw transcript
    tags: list[str]        # Additional context tags: ["short", "urgent", etc.]

class InputClassifier:
    """Fast, rule-based input classification for quick response selection.

    NOT an LLM call. Uses keyword patterns, regex, and heuristics.
    Designed to complete in <10ms for typical inputs.
    """

    def classify(self, transcript: str, stt_metadata: dict | None = None) -> ClassifiedInput:
        """Classify user input into a situation category.

        Args:
            transcript: The STT-produced text
            stt_metadata: Optional metadata from STT (no_speech_prob, etc.)

        Returns:
            ClassifiedInput with category and confidence
        """
        ...
```

**Classification categories (initial set):**

| Category | Subcategory | Pattern | Example Input |
|----------|-------------|---------|---------------|
| social | greeting | starts with "hey", "hi", "hello", "what's up" | "Hey, how's it going?" |
| social | farewell | "bye", "see you", "later", "goodnight" | "Alright, talk to you later" |
| social | thanks | "thanks", "thank you", "appreciate it" | "Thanks for that" |
| question | info_query | ends with "?", starts with wh-word | "What time is it?" |
| question | opinion | "what do you think", "should I" | "Should I use SQLite or JSON?" |
| command | task_request | "can you", "please", "do X", imperative | "Run the tests" |
| command | tool_use | references files, code, projects | "Check the logs in /var/log" |
| emotional | frustration | "ugh", "damn", "this sucks" | "Ugh, it broke again" |
| emotional | excitement | "awesome", "yes!", "nice" | "That's awesome!" |
| acknowledgment | affirmative | "yeah", "okay", "sure", "got it" | "Yeah, makes sense" |
| acknowledgment | negative | "no", "nah", "not really" | "Nah, that's not right" |
| non_speech | cough | STT rejected + audio characteristics | (cough detected by STT filter) |
| non_speech | sigh | STT low confidence + duration | (sigh) |
| non_speech | laugh | STT low confidence + pattern | (laugh) |
| unknown | unknown | low confidence on all patterns | (ambiguous input) |

**Non-speech detection integration:** The existing STT stage already filters non-speech via `_whisper_transcribe()` (lines 784-817). When all segments are rejected, it calls `self._set_status("stt_rejected")`. Currently this data is discarded. The new architecture captures these rejections:

- When Whisper rejects all segments with high `no_speech_prob`, the STT stage can emit a new frame type (see pipeline frame changes below) carrying the rejection metadata
- The classifier uses `no_speech_prob`, `avg_logprob`, and the raw text (even rejected text like "hmm" or "uh") to categorize the non-speech event
- This allows "excuse you" for a cough, empathetic acknowledgment for a sigh, etc.

#### 2. ResponseLibrary (in-process, lookup)

**Location:** New file `response_library.py`, instantiated in `LiveSession.__init__()`

**Storage format: JSON** (not SQLite). Rationale:
- The library will have 50-200 entries at most (each is a situation + clip reference)
- Entire library loads into memory at startup (same pattern as `_load_filler_clips()`)
- No concurrent write access needed during a session (writes happen post-session)
- JSON is human-readable and consistent with `ack_pool.json` pattern
- SQLite adds a dependency for no benefit at this scale

**Storage location:** `audio/responses/` (parallel to `audio/fillers/`)

```
audio/
  fillers/
    acknowledgment/          # existing ack clips (to be migrated)
    ack_pool.json            # existing metadata
  responses/
    library.json             # response library index
    clips/                   # pre-generated response audio clips
      greeting_hey_001.wav
      greeting_hey_002.wav
      thanks_welcome_001.wav
      cough_excuse_001.wav
      ...
```

**library.json schema:**

```json
{
  "version": 1,
  "entries": [
    {
      "id": "greeting_hey_001",
      "situation": {
        "category": "social",
        "subcategory": "greeting"
      },
      "response_text": "Hey! What's up?",
      "clip_filename": "greeting_hey_001.wav",
      "created_at": 1771418989.27,
      "use_count": 14,
      "last_used": 1771500000.0,
      "effectiveness": 0.85,
      "tags": ["casual", "short"]
    }
  ]
}
```

**Lookup algorithm:**

```python
class ResponseLibrary:
    def __init__(self):
        self._entries: list[ResponseEntry] = []
        self._index: dict[str, list[ResponseEntry]] = {}  # category -> entries
        self._clips: dict[str, bytes] = {}  # id -> pcm_data
        self._usage_log: list[dict] = []    # tracks what was played

    def lookup(self, classification: ClassifiedInput) -> ResponseEntry | None:
        """Find the best matching response for a classified input.

        Returns None if no good match exists (falls back to generic ack).
        """
        key = f"{classification.category}:{classification.subcategory}"
        candidates = self._index.get(key, [])

        if not candidates:
            # Fall back to category-level match
            candidates = self._index.get(classification.category, [])

        if not candidates:
            return None  # No match -- _filler_manager falls back to random ack

        # Score candidates: prefer variety (avoid recently used), high effectiveness
        return self._score_and_pick(candidates)

    def get_clip(self, entry_id: str) -> bytes | None:
        """Get pre-loaded PCM audio for a response entry."""
        return self._clips.get(entry_id)
```

**Lookup must be fast:** The `_index` is a dict keyed by `"category:subcategory"`. Lookup is O(1) to find candidates, then O(n) to score a small list (typically 2-5 candidates per subcategory). Total: <1ms.

#### 3. LibraryCurator (post-session daemon)

**Location:** New file `library_curator.py`, spawned like `learner.py`

**Pattern:** Follows the exact same pattern as `learner.py`:
- Spawned as subprocess at session start via `_spawn_curator()`
- Reads conversation JSONL log
- Runs post-session analysis via `claude -p`
- Writes new entries to `library.json` and generates clips via Piper
- Communicates back via signal file (like `learner_notify`)

**Why a separate daemon (not inline):**
- Library growth involves LLM calls (deciding what phrases to generate) -- too slow for real-time
- Clip generation via Piper takes 1-3s per clip -- would block the pipeline
- Same architectural pattern as learner: watch log, extract insights, write to disk
- Next session picks up new clips automatically via `_load_library()` at startup

**Curator responsibilities:**

1. **Gap identification:** After session ends, analyze conversation log for situations where:
   - No response was found (logged as `response_miss` events)
   - A generic acknowledgment was used where a specific response would be better
   - New interaction patterns appeared (user greeting style, frequent question types)

2. **Clip generation:** For each identified gap:
   - Use Claude to generate 2-3 natural response phrases for the situation
   - Generate audio clips via Piper with varied TTS parameters (same as clip_factory.py)
   - Evaluate clip quality (same scoring as existing `evaluate_clip()`)
   - Add passing clips to `library.json`

3. **Quality pruning:** Remove entries that:
   - Were used but followed by the user interrupting (barge-in after quick response = bad match)
   - Have very low `effectiveness` scores
   - Are duplicative (multiple entries for same situation with similar text)
   - Exceed per-category caps (keep top N per subcategory)

### Modified Components

#### 4. Modified: `_filler_manager()` in live_session.py

The current `_filler_manager()` is the surgical modification point. It currently:
1. Waits 500ms gate
2. Picks random clip
3. Plays it

The new version:
1. Classifies input via `InputClassifier.classify(transcript)`
2. Looks up response via `ResponseLibrary.lookup(classification)`
3. Waits gate (keep the gate but may reduce to 300ms for known-good matches)
4. If response found: play it
5. If no response: fall back to random acknowledgment clip (existing behavior)

```python
async def _filler_manager(self, user_text: str, cancel_event: asyncio.Event):
    """Play a context-appropriate quick response while waiting for LLM."""
    # Step 1: Classify input (sync, <10ms)
    classification = self._classifier.classify(user_text)

    # Step 2: Lookup response (sync, <1ms)
    response = self._response_library.lookup(classification)

    # Step 3: Gate -- skip if LLM responds fast
    gate_ms = 0.3 if response and response.confidence > 0.8 else 0.5
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=gate_ms)
        return  # LLM responded fast, skip
    except asyncio.TimeoutError:
        pass

    if cancel_event.is_set():
        return

    # Step 4: Play response or fall back
    if response:
        clip = self._response_library.get_clip(response.id)
        if clip:
            self._response_library.log_usage(response.id, classification)
            await self._play_filler_audio(clip, cancel_event)
            return

    # Fallback: existing random acknowledgment
    clip = self._pick_filler("acknowledgment")
    if clip:
        await self._play_filler_audio(clip, cancel_event)
```

**What does NOT change:**
- `_play_filler_audio()` -- unchanged, sends FILLER frames to `_audio_out_q`
- `_play_gated_ack()` -- unchanged, used for tool-use acknowledgments
- `_pick_filler()` -- kept as fallback
- All playback stage logic -- unchanged
- Barge-in logic -- unchanged
- STT gating -- unchanged

#### 5. Modified: STT stage (non-speech event forwarding)

Currently when Whisper rejects all segments (lines 813-817), it calls `self._set_status("stt_rejected")` and discards the data. The modification:

```python
# In _stt_stage, after all segments rejected:
if not kept and segments:
    self._set_status("stt_rejected")
    rejected_text = "".join(s.get("text", "") for s in segments).strip()

    # NEW: Forward non-speech event for quick response handling
    await self._stt_out_q.put(PipelineFrame(
        type=FrameType.NON_SPEECH,
        generation_id=self.generation_id,
        data=rejected_text,
        metadata={
            "no_speech_prob": max(s.get("no_speech_prob", 0) for s in segments),
            "avg_logprob": min(s.get("avg_logprob", 0) for s in segments),
        }
    ))
```

Then in `_llm_stage`, handle NON_SPEECH frames by running classification and playing a quick response WITHOUT sending to the LLM:

```python
if frame.type == FrameType.NON_SPEECH:
    # Classify non-speech event
    classification = self._classifier.classify(
        frame.data,
        stt_metadata=frame.metadata
    )
    response = self._response_library.lookup(classification)
    if response:
        clip = self._response_library.get_clip(response.id)
        if clip:
            await self._play_filler_audio(clip, asyncio.Event())  # no cancel
    continue  # Do NOT send to LLM
```

### Pipeline Frame Changes

Add one new frame type to `pipeline_frames.py`:

```python
class FrameType(Enum):
    AUDIO_RAW = auto()
    TRANSCRIPT = auto()
    TEXT_DELTA = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    TTS_AUDIO = auto()
    END_OF_TURN = auto()
    END_OF_UTTERANCE = auto()
    FILLER = auto()
    BARGE_IN = auto()
    CONTROL = auto()
    NON_SPEECH = auto()       # NEW: STT-rejected audio event (cough, sigh, etc.)
```

The `metadata` field on `PipelineFrame` already exists and supports arbitrary dicts, so no structural change to the dataclass is needed. The NON_SPEECH frame carries:
- `data`: the rejected transcript text (may be useful for classification: "hmm", "uh")
- `metadata`: `{"no_speech_prob": float, "avg_logprob": float}` from Whisper segments

### Data Flow: Complete Turn Lifecycle

**Normal speech turn (with quick response):**

```
1. User speaks into mic
2. Audio Capture -> AUDIO_RAW frames -> audio_in_q
3. STT Stage: Whisper transcribes -> "What's the weather?"
4. STT Stage -> END_OF_UTTERANCE + TRANSCRIPT frames -> stt_out_q
5. LLM Stage receives TRANSCRIPT:
   a. InputClassifier.classify("What's the weather?")
      -> ClassifiedInput(category="question", subcategory="info_query")
   b. ResponseLibrary.lookup(classification)
      -> ResponseEntry(text="Let me check on that.", clip="info_query_check_001.wav")
   c. Start _filler_manager with classification + response pre-loaded
   d. Send to Claude CLI
   e. _filler_manager waits 300ms gate
   f. Gate expires -> play "Let me check on that." clip as FILLER frames
   g. Claude CLI responds -> cancel_event.set() -> filler stops
   h. TEXT_DELTA frames flow to TTS
6. TTS Stage: Piper synthesizes -> TTS_AUDIO frames -> audio_out_q
7. Playback Stage: plays audio
```

**Non-speech event turn:**

```
1. User coughs into mic
2. Audio Capture -> AUDIO_RAW frames -> audio_in_q
3. STT Stage: Whisper transcribes -> rejects all segments (no_speech_prob=0.85)
4. STT Stage -> NON_SPEECH frame -> stt_out_q
   (data="", metadata={"no_speech_prob": 0.85, "avg_logprob": -1.5})
5. LLM Stage receives NON_SPEECH:
   a. InputClassifier.classify("", stt_metadata={...})
      -> ClassifiedInput(category="non_speech", subcategory="cough")
   b. ResponseLibrary.lookup(classification)
      -> ResponseEntry(text="Excuse you!", clip="cough_excuse_001.wav")
   c. Play clip directly (no LLM call, no gate delay)
6. Playback Stage: plays "Excuse you!" clip
7. Return to listening (no LLM processing needed)
```

**Session end (library growth):**

```
1. Session ends -> session_end event logged to JSONL
2. LibraryCurator (daemon) detects session_end
3. Curator reads conversation log + response_miss events
4. Curator calls claude -p to analyze gaps:
   "User said 'morning!' 3 times but got generic 'let me check' -- need morning greeting"
5. Curator generates clips via Piper: "Good morning!", "Morning!", "Hey, good morning!"
6. Curator writes new entries to library.json
7. Next session: LiveSession.__init__() loads updated library
```

### Integration with Existing Systems

#### Integration with Learner Daemon

The learner and curator are complementary but independent:
- **Learner:** Watches conversation, writes to `personality/memories/` (who the user is)
- **Curator:** Watches conversation, writes to `audio/responses/library.json` (how to respond quickly)
- Both follow the same pattern: subprocess, JSONL tailing, claude -p evaluation
- They run concurrently, no conflicts (different output files)

The learner does NOT help grow the library directly. They have different concerns:
- Learner extracts durable facts ("user likes Python, has a cat named Luna")
- Curator identifies response patterns ("user says 'morning' at session start -> need greeting clip")

#### Integration with Clip Factory

The existing `clip_factory.py` manages the acknowledgment clip pool. With adaptive responses:
- The clip factory continues to maintain the acknowledgment fallback pool
- The curator generates situation-specific clips using the same Piper TTS and `evaluate_clip()` function
- The curator can import `generate_clip` and `evaluate_clip` from `clip_factory.py` directly
- Over time, as the response library grows, the acknowledgment pool becomes less used (but never removed -- it is the safety net)

#### Integration with Barge-in

No changes to barge-in logic. The playback stage already handles FILLER frames identically to TTS_AUDIO for barge-in purposes. When the user interrupts a quick response:
1. Barge-in triggers as normal (VAD detection, generation_id increment)
2. Quick response audio stops (filler frame generation checks `cancel_event` and `generation_id`)
3. Annotation is built with whatever was spoken
4. The curator logs this as a negative signal for that response entry (barge-in after quick response = bad match)

#### Integration with Tool-Use Flow

The existing tool-use flow in `_read_cli_response()` has its own acknowledgment path (`_play_gated_ack`). This remains unchanged. The adaptive response system only affects the initial filler played before the LLM starts responding. Tool-use acknowledgments ("checking now", "one moment") are already contextually appropriate since they play after a tool_use content_block_start event.

### Storage and Persistence

| Data | Location | Format | Lifecycle |
|------|----------|--------|-----------|
| Response library index | `audio/responses/library.json` | JSON | Grows across sessions, loaded at startup |
| Response audio clips | `audio/responses/clips/*.wav` | WAV (22050Hz) | Generated by curator, loaded at startup |
| Usage log (per session) | Session dir `response_usage.jsonl` | JSONL | Written during session, read by curator |
| Response miss log | Session dir `response_misses.jsonl` | JSONL | Written during session, read by curator |
| Acknowledgment fallback | `audio/fillers/acknowledgment/*.wav` | WAV (22050Hz) | Existing, maintained by clip_factory |

### Concurrency and Thread Safety

- `InputClassifier.classify()`: Pure function, no state mutation, safe to call from asyncio
- `ResponseLibrary.lookup()`: Read-only during session, safe from asyncio event loop
- `ResponseLibrary.log_usage()`: Appends to in-memory list, flushed to JSONL at session end
- `LibraryCurator`: Separate subprocess, writes to `library.json` only after session ends, no concurrent access
- Clip loading at startup: synchronous in `__init__`, same pattern as existing `_load_filler_clips()`

## Build Order

Components can be built in phases with clear independence:

### Phase 1: Input Classification (can be built and tested independently)

Build `input_classifier.py` with `ClassifiedInput` dataclass and `InputClassifier` class. This has zero dependencies on the rest of the system. Can be tested with unit tests against sample transcripts.

**Deliverable:** `input_classifier.py` with tests
**Dependencies:** None
**Risk:** Low -- pattern matching is well-understood

### Phase 2: Response Library (can be built and tested independently)

Build `response_library.py` with `ResponseEntry` dataclass, `ResponseLibrary` class, JSON storage, and lookup algorithm. Seed with initial entries (migrate existing acknowledgment phrases + add social/emotional responses).

**Deliverable:** `response_library.py` with initial `library.json`, tests
**Dependencies:** None (uses existing Piper for initial clip generation)
**Risk:** Low -- data structure and lookup

### Phase 3: Pipeline Integration (depends on Phase 1 + 2)

Modify `_filler_manager()` in `live_session.py` to use classifier + library. Add `NON_SPEECH` frame type. Modify STT stage to emit non-speech events. Wire up logging of usage and misses.

**Deliverable:** Modified `live_session.py`, `pipeline_frames.py`
**Dependencies:** Phase 1, Phase 2
**Risk:** Medium -- touching the hot path, needs careful testing to avoid regression

### Phase 4: Library Curator (can be built after Phase 2)

Build `library_curator.py` daemon. Follows learner.py pattern exactly. Post-session gap analysis, clip generation, quality pruning.

**Deliverable:** `library_curator.py`
**Dependencies:** Phase 2 (needs library.json schema), `clip_factory.py` (reuses generate/evaluate)
**Risk:** Medium -- LLM-based analysis quality depends on prompt engineering

## Anti-Patterns to Avoid

### Anti-Pattern 1: LLM Classification in the Hot Path

**What:** Using Claude to classify input before selecting a quick response.
**Why bad:** Claude CLI takes 1-3 seconds for even simple prompts. This would eliminate the entire benefit of quick responses (which exist to fill that wait time).
**Instead:** Use pure-Python pattern matching. The classification does not need to be perfect -- it needs to be fast. A wrong quick response is better than silence, and the LLM response arrives within seconds anyway.

### Anti-Pattern 2: Shared Mutable State Between Session and Curator

**What:** Having the curator modify the library while a session is running.
**Why bad:** Race conditions on `library.json` reads/writes. The in-memory index would go stale.
**Instead:** Curator only writes when no session is active (post-session). Next session loads fresh state at startup. Same pattern as learner.py.

### Anti-Pattern 3: Over-Engineering the Classifier

**What:** Building an ML-based classifier for input categorization.
**Why bad:** Adds model loading latency, dependency complexity, and the categories are simple enough for rules. The 15 categories above cover 90%+ of conversational inputs with regex + keyword matching.
**Instead:** Start with rules. If classification quality is insufficient, add ML later (can swap the classifier implementation without changing the interface).

### Anti-Pattern 4: Replacing the Filler System Entirely

**What:** Removing the existing `_pick_filler("acknowledgment")` fallback.
**Why bad:** The response library starts empty/small. Without fallback, there would be silence for uncovered situations.
**Instead:** Keep the existing acknowledgment pool as fallback. Quick responses are an upgrade layer, not a replacement.

### Anti-Pattern 5: Pre-Generating All Possible Responses

**What:** Trying to generate clips for every possible situation up front.
**Why bad:** Combinatorial explosion. The strength of this system is that it learns which situations actually occur and generates clips for those.
**Instead:** Start with a minimal seed library (greetings, thanks, common question acknowledgments). Let the curator grow it based on actual usage.

## Summary of Changes by File

| File | Change Type | What Changes |
|------|-------------|--------------|
| `pipeline_frames.py` | MODIFY | Add `NON_SPEECH` to FrameType enum |
| `live_session.py` | MODIFY | `_filler_manager()` uses classifier + library; `_llm_stage()` handles NON_SPEECH frames; `__init__()` instantiates classifier + library; STT stage emits NON_SPEECH frames; `_spawn_curator()` added; logging additions |
| `input_classifier.py` | NEW | InputClassifier class, ClassifiedInput dataclass |
| `response_library.py` | NEW | ResponseLibrary class, ResponseEntry dataclass, JSON persistence |
| `library_curator.py` | NEW | Post-session daemon for library growth/pruning |
| `audio/responses/library.json` | NEW | Response library index (seed + grows over time) |
| `audio/responses/clips/` | NEW | Directory for response audio clips |
| `clip_factory.py` | UNCHANGED | Continues managing acknowledgment fallback pool |
| `learner.py` | UNCHANGED | Continues managing personality memories |

## Sources

- **live_session.py** (read in full): Pipeline architecture, filler system, STT gating, barge-in logic, tool-use flow. Lines 559-574 (filler_manager), 1496-1564 (llm_stage), 1325-1492 (stt_stage).
- **pipeline_frames.py** (read in full): Frame types and PipelineFrame dataclass.
- **learner.py** (read in full): Daemon pattern for post-session processing.
- **clip_factory.py** (read in full): Clip generation, quality evaluation, pool management.
- **task_manager.py** (read in full): Subprocess management patterns.
- **ack_pool.json** (read in full): Current clip metadata schema.
