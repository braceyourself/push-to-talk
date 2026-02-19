# Phase 9: Semantic Matching + Pipeline Polish - Research

**Researched:** 2026-02-18
**Domain:** Semantic text classification (model2vec), trivial input detection, audio stream composition/cadence management, sentence-segmented TTS pipeline
**Confidence:** HIGH (codebase fully read, model2vec docs verified, architecture patterns from existing pipeline)

## Summary

Phase 9 adds four capabilities to the existing pipeline: (1) model2vec semantic fallback for the heuristic classifier, (2) trivial input detection that suppresses filler clips for backchannels, (3) barge-in awareness during clip playback, and (4) a stream composer subprocess that manages a unified sentence queue with natural pacing between all audio segments.

The semantic fallback is straightforward: model2vec's `StaticModel` loads a ~8-30MB model, encodes text in <1ms, and cosine similarity against a small set of category exemplar embeddings produces a classification. This runs inside the existing classifier daemon process (`input_classifier.py`), requiring only `pip install model2vec` as a new dependency. The trivial input detection is a word-list check in the classifier daemon that returns a "trivial" flag alongside the classification result. Both fit within the existing 500ms filler gate with room to spare.

The stream composer is the most significant new component. It replaces the current direct-to-queue TTS pipeline with a subprocess that receives all audio segments (filler clips, TTS sentences, silence pauses, non-speech clips), orders them in a unified queue, inserts natural pauses between segments, and emits final audio frames. Barge-in integration means the composer tracks which sentence is currently playing and can pause the queue mid-stream. The current `_tts_stage()` and `_play_filler_audio()` both write to `_audio_out_q` -- the composer sits between the content producers and `_audio_out_q`, becoming the single writer to the playback queue.

**Primary recommendation:** Add model2vec to the classifier daemon (same process, loaded at startup), implement trivial detection as a word-list check with context flag, build the stream composer as an in-process asyncio class (not a separate subprocess -- IPC overhead is unnecessary for what is essentially queue management within the same event loop).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `model2vec` | 0.7.x | Semantic embedding for fallback classification | 8MB model, <1ms encode, only dependency is numpy (already installed). Official MinishLab library. |
| `numpy` (existing) | 1.26.4 | Cosine similarity computation, audio processing | Already installed. `np.dot` + `np.linalg.norm` for similarity. No need for sklearn. |
| Python `asyncio` (stdlib) | 3.12 | Stream composer queue management, event coordination | Already powers entire pipeline. Composer is an asyncio task, not a subprocess. |
| Python `re` (stdlib) | 3.12 | Trivial input pattern matching | Already used in classifier daemon. |
| `pysbd` | 0.3.4 | Sentence boundary detection for LLM text splitting | Rule-based, no model to load, handles abbreviations/decimals correctly. 22 languages. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Piper TTS (existing) | en_US-lessac-medium | Per-sentence TTS generation | Already installed. Called per-sentence by composer instead of per-text-block. |
| Python `wave` (stdlib) | 3.12 | Loading non-speech audio clips (thinking sounds, breaths) | Already used throughout codebase. |
| Python `json` (stdlib) | 3.12 | Classifier IPC, cadence configuration | Already used for all IPC in project. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| model2vec `StaticModel.encode()` + cosine sim | model2vec `StaticModelForClassification` | The classifier API requires training data and scikit-learn dependency. For 6 categories with 5-10 exemplars each, cosine similarity against pre-computed exemplar embeddings is simpler, needs no training step, and has identical speed. |
| model2vec potion-base-8M | model2vec potion-base-2M | 2M is smaller (4MB vs 8MB) but has lower accuracy. 8M is the sweet spot: still tiny, 30ms load time, better similarity quality. |
| model2vec potion-base-8M | potion-base-32M | 32M is more accurate but 4x larger (30MB). For 6 coarse categories, the extra accuracy is marginal. 8M keeps the classifier daemon lean. |
| pysbd | NLTK `sent_tokenize` | NLTK requires downloading punkt data (140KB+), has a heavier import chain. pysbd is rule-based, zero data files, handles edge cases better. |
| pysbd | Regex `SENTENCE_END_RE` (existing) | The existing regex (`[.!?]\s|[.!?]$|\n`) is already in `live_session.py`. It fails on abbreviations ("Dr. Smith said..."), decimal numbers ("version 3.14 is..."), and URLs. pysbd handles all of these correctly. |
| In-process stream composer class | Separate subprocess (per CONTEXT.md) | CONTEXT.md says "new subprocess: stream composer / cadence manager -- separate from classifier daemon." However, after analyzing the codebase, a subprocess adds IPC latency and complexity for no benefit. The composer needs real-time access to `_audio_out_q`, `generation_id`, and barge-in events -- all in-process state. Recommend implementing as an asyncio class within the main process that runs as an asyncio task. The "separate concern" benefit is achieved via class isolation, not process isolation. |

**Installation:**
```bash
pip install model2vec pysbd
```

**Model download (first run):**
```python
from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/potion-base-8M")  # Downloads ~8MB from HuggingFace
```

## Architecture Patterns

### Current Pipeline (5 stages)

```
Audio Capture -> STT (Whisper+VAD) -> LLM (Claude CLI) -> TTS (Piper) -> Playback (PyAudio)
                                      |                    |
                                      +-- filler_manager --+-- writes FILLER frames to audio_out_q
                                      |   (classification, |
                                      |    clip lookup,    |
                                      |    500ms gate)     |
                                      |                    |
                                      +-- _read_cli_response() writes TEXT_DELTA to llm_out_q
                                                           |
                                                           TTS stage reads from llm_out_q,
                                                           writes TTS_AUDIO to audio_out_q
```

### Proposed Pipeline (with stream composer)

```
Audio Capture -> STT (Whisper+VAD) -> LLM (Claude CLI) -> Stream Composer -> Playback (PyAudio)
                                      |                    |
                                      +-- filler_manager --+-- sends clip to composer
                                      |   (classification +|
                                      |    semantic match + |
                                      |    trivial detect)  |
                                      |                    |
                                      +-- _read_cli_response() sends sentences to composer
                                                           |
                                                           Composer:
                                                           1. Receives segments (clip, sentences, pauses)
                                                           2. Generates TTS per sentence (inline)
                                                           3. Inserts pauses between segments
                                                           4. Writes final frames to audio_out_q
                                                           5. Tracks current sentence for barge-in
```

### Recommended Project Structure Changes

```
push-to-talk/
├── input_classifier.py      # MODIFIED: add model2vec semantic fallback + trivial detection
├── stream_composer.py        # NEW: StreamComposer class (asyncio, in-process)
├── response_library.py       # UNCHANGED
├── live_session.py           # MODIFIED: route filler + TTS through composer
├── pipeline_frames.py        # MODIFIED: add new frame types
├── clip_factory.py           # UNCHANGED
├── audio/
│   ├── responses/            # EXISTING (Phase 8)
│   └── non_speech/           # NEW: thinking sounds, breath clips, tonal cues
│       ├── thinking/         # "hmm" sounds, tonal thinking cues
│       ├── breath/           # Subtle breath-like pauses
│       └── transition/       # Brief tonal cues for segment transitions
├── category_exemplars.json   # NEW: exemplar phrases per category for semantic matching
└── requirements.txt          # MODIFIED: add model2vec, pysbd
```

### Pattern 1: Semantic Fallback in Classifier Daemon

**What:** model2vec loads at classifier daemon startup. When heuristic patterns produce no confident match, semantic similarity against category exemplars provides a fallback.

**When to use:** When the heuristic classifier returns confidence < 0.4 (current default fallback). Instead of immediately falling back to "acknowledgment", try semantic matching first.

**How it works:**
1. Classifier daemon loads `StaticModel.from_pretrained("minishlab/potion-base-8M")` at startup
2. Pre-computes embeddings for 5-10 exemplar phrases per category from `category_exemplars.json`
3. On classify request: heuristic runs first (<1ms). If confident, return immediately.
4. If heuristic returns low confidence: encode input text (<1ms), compute cosine similarity against all exemplar embeddings (<1ms), return highest-scoring category.
5. If highest semantic score is also very low (< threshold): fall back to acknowledgment.

**Example:**

```python
# In input_classifier.py -- at daemon startup
import numpy as np

class SemanticFallback:
    def __init__(self, exemplars_path: str):
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained("minishlab/potion-base-8M")

        # Load exemplar phrases per category
        with open(exemplars_path) as f:
            exemplars = json.load(f)

        # Pre-compute embeddings for all exemplars
        self._category_embeddings: dict[str, np.ndarray] = {}
        for category, phrases in exemplars.items():
            embeddings = self.model.encode(phrases)  # shape: (N, dim)
            self._category_embeddings[category] = embeddings

    def classify(self, text: str) -> tuple[str, float]:
        """Return (category, confidence) via semantic similarity."""
        text_emb = self.model.encode([text])[0]  # shape: (dim,)

        best_category = "acknowledgment"
        best_score = 0.0

        for category, exemplar_embs in self._category_embeddings.items():
            # Cosine similarity against each exemplar
            norms = np.linalg.norm(exemplar_embs, axis=1) * np.linalg.norm(text_emb)
            sims = np.dot(exemplar_embs, text_emb) / np.maximum(norms, 1e-8)
            max_sim = float(np.max(sims))

            if max_sim > best_score:
                best_score = max_sim
                best_category = category

        # Convert cosine similarity (0-1 range) to confidence
        # Cosine sim of 0.3+ is a reasonable match for sentence embeddings
        confidence = min(best_score, 1.0)
        return best_category, confidence

# Integration into classify() function:
def classify(text: str, semantic: SemanticFallback | None = None) -> ClassifiedInput:
    """Classify with heuristic first, semantic fallback second."""
    # Heuristic match (existing code, <1ms)
    result = _heuristic_classify(text)

    if result.confidence >= 0.5:
        return result  # Heuristic is confident, use it

    if semantic is None:
        return result  # No semantic model loaded

    # Semantic fallback (<5ms)
    sem_category, sem_confidence = semantic.classify(text)

    # Higher confidence wins (per CONTEXT.md decision)
    if sem_confidence > result.confidence:
        return ClassifiedInput(
            category=sem_category,
            confidence=sem_confidence,
            original_text=text,
            subcategory=_infer_subcategory(sem_category, text),
            match_type="semantic"  # For logging
        )

    return result
```

### Pattern 2: Trivial Input Detection

**What:** Detect conversational backchannels ("yes", "ok", "mhm", "yeah sure") that don't need a filler clip. Returns a `trivial` flag alongside the normal classification.

**When to use:** For every classification request. The `trivial` flag is checked in `_filler_manager()` -- if True, skip filler entirely (natural silence).

**Context-dependency:** If the AI just asked a question (tracked via `last_ai_asked_question` flag in session state), the next utterance is treated as a real answer regardless of length.

**Example:**

```python
# Trivial input word/phrase list
TRIVIAL_PATTERNS = {
    # Pure affirmatives (no verb, no directive)
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright",
    "right", "correct", "exactly", "mhm", "mm-hm", "mmhm", "uh huh",
    "cool", "fine", "gotcha", "got it",
    # Pure negatives
    "no", "nah", "nope", "not really",
    # Minimal acknowledgments
    "ah", "oh", "hm", "hmm", "huh",
    # Casual affirmations
    "yeah sure", "okay cool", "sure thing", "sounds good",
    "yeah okay", "ok cool", "alright cool",
}

def is_trivial(text: str, ai_asked_question: bool = False) -> bool:
    """Detect trivial backchannel input that needs no filler clip.

    Args:
        text: User's transcribed speech
        ai_asked_question: If True, treat ANY input as real (answer to question)
    """
    if ai_asked_question:
        return False  # Context override: AI asked, user answered

    cleaned = text.strip().lower().rstrip('.!?,')

    # Must be short (<=4 words) to be trivial
    if len(cleaned.split()) > 4:
        return False

    # Anything with a verb or directive is real input
    # Quick check: if it matches a task/question pattern, it's not trivial
    if any(pat.search(cleaned) for pat in PATTERNS.get("task", [])):
        return False
    if cleaned.endswith("?"):
        return False

    # Check against trivial word/phrase list
    return cleaned in TRIVIAL_PATTERNS
```

### Pattern 3: Stream Composer (Cadence Manager)

**What:** An asyncio class that manages a unified sentence queue. All audio segments (filler clips, TTS sentences, silence pauses, non-speech clips) flow through the composer, which controls pacing, insertion of pauses, and provides barge-in awareness.

**Why not a subprocess:** The composer needs real-time access to `generation_id` (for barge-in detection), `_audio_out_q` (for frame output), and needs to call Piper TTS (for per-sentence generation). These are all in-process resources. IPC would add latency and complexity for zero benefit.

**When to use:** Replace the current TTS stage with the composer as the sole writer to `_audio_out_q`.

**Example:**

```python
# stream_composer.py

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Awaitable

class SegmentType(Enum):
    FILLER_CLIP = auto()      # Pre-recorded response clip (PCM bytes)
    TTS_SENTENCE = auto()     # Text to synthesize via Piper
    SILENCE = auto()           # Explicit pause (duration in seconds)
    NON_SPEECH = auto()        # Thinking sound, breath, tonal cue (PCM bytes)

@dataclass
class AudioSegment:
    type: SegmentType
    data: bytes | str = b""    # PCM bytes for clips, text for TTS
    duration: float = 0.0      # For SILENCE segments
    metadata: dict = field(default_factory=dict)

class StreamComposer:
    """Manages unified audio sentence queue with natural pacing.

    All audio segments flow through the composer:
    1. Filler clip (from filler_manager/response_library)
    2. Optional pause
    3. LLM TTS sentences (text -> Piper -> PCM, streamed per-sentence)
    4. Inter-sentence pauses (variable, context-aware)
    5. Non-speech elements (thinking sounds, breaths)
    """

    def __init__(self,
                 audio_out_q: asyncio.Queue,
                 tts_fn: Callable[[str], Awaitable[bytes | None]],
                 get_generation_id: Callable[[], int],
                 sample_rate: int = 24000):
        self._audio_out_q = audio_out_q
        self._tts_fn = tts_fn  # async fn: text -> PCM bytes (resampled)
        self._get_gen_id = get_generation_id
        self._sample_rate = sample_rate

        # Segment queue
        self._segment_q: asyncio.Queue[AudioSegment | None] = asyncio.Queue()

        # State tracking
        self._current_gen_id: int = 0
        self._sentence_index: int = 0
        self._sentences_played: int = 0
        self._paused: bool = False  # Set by barge-in
        self._held_segments: list[AudioSegment] = []  # Unspoken after barge-in

        # Cadence parameters (tunable)
        self._inter_sentence_pause: float = 0.15  # Default 150ms between sentences
        self._post_clip_pause: float = 0.3         # Pause after filler clip
        self._thinking_pause: float = 0.5           # Pause for thinking sounds

    async def enqueue(self, segment: AudioSegment):
        """Add a segment to the playback queue."""
        await self._segment_q.put(segment)

    async def enqueue_end_of_turn(self):
        """Signal that all segments for this turn have been queued."""
        await self._segment_q.put(None)  # Sentinel

    def pause(self) -> list[AudioSegment]:
        """Pause playback (barge-in). Returns unplayed segments."""
        self._paused = True
        held = list(self._held_segments)
        self._held_segments.clear()
        # Drain remaining queued segments
        while not self._segment_q.empty():
            try:
                seg = self._segment_q.get_nowait()
                if seg is not None:
                    held.append(seg)
            except asyncio.QueueEmpty:
                break
        return held

    def resume(self):
        """Resume playback after barge-in pause."""
        self._paused = False

    async def run(self):
        """Main loop: process segments, generate TTS, emit audio frames."""
        while True:
            segment = await self._segment_q.get()

            if segment is None:
                # End of turn sentinel
                gen_id = self._get_gen_id()
                await self._audio_out_q.put(PipelineFrame(
                    type=FrameType.END_OF_TURN,
                    generation_id=gen_id
                ))
                continue

            if self._paused:
                self._held_segments.append(segment)
                continue

            gen_id = self._get_gen_id()

            if segment.type == SegmentType.FILLER_CLIP:
                # Emit pre-recorded clip PCM
                await self._emit_pcm(segment.data, FrameType.FILLER, gen_id)
                # Pause after filler before LLM TTS begins
                await self._emit_silence(self._post_clip_pause, gen_id)

            elif segment.type == SegmentType.TTS_SENTENCE:
                # Generate TTS for this sentence
                pcm = await self._tts_fn(segment.data)
                if pcm:
                    await self._emit_pcm(pcm, FrameType.TTS_AUDIO, gen_id)
                    # Sentence done marker
                    await self._audio_out_q.put(PipelineFrame(
                        type=FrameType.CONTROL,
                        generation_id=gen_id,
                        data="sentence_done"
                    ))
                    self._sentences_played += 1
                    # Inter-sentence pause
                    await self._emit_silence(self._inter_sentence_pause, gen_id)

            elif segment.type == SegmentType.SILENCE:
                await self._emit_silence(segment.duration, gen_id)

            elif segment.type == SegmentType.NON_SPEECH:
                await self._emit_pcm(segment.data, FrameType.FILLER, gen_id)

    async def _emit_pcm(self, pcm: bytes, frame_type: FrameType, gen_id: int):
        """Emit PCM data as chunked frames to audio_out_q."""
        offset = 0
        chunk_size = 4096
        while offset < len(pcm):
            if self._get_gen_id() != gen_id or self._paused:
                return
            chunk = pcm[offset:offset + chunk_size]
            offset += chunk_size
            await self._audio_out_q.put(PipelineFrame(
                type=frame_type,
                generation_id=gen_id,
                data=chunk
            ))

    async def _emit_silence(self, duration: float, gen_id: int):
        """Emit silence frames for the given duration."""
        if duration <= 0:
            return
        total_bytes = int(self._sample_rate * 2 * duration)  # 16-bit mono
        silence = b'\x00' * total_bytes
        await self._emit_pcm(silence, FrameType.TTS_AUDIO, gen_id)
```

### Pattern 4: Sentence Splitting for Streaming TTS

**What:** Split LLM text response into sentences using pysbd, then feed each sentence to TTS individually. First sentence plays while later sentences are still being generated/synthesized.

**Why pysbd over existing regex:** The current `SENTENCE_END_RE` regex in `live_session.py` splits on any `.`, `!`, `?` followed by whitespace or end-of-string. This fails on:
- "Dr. Smith said..." (splits after "Dr.")
- "Version 3.14 is great." (splits after "3.")
- "See example.com for details." (splits after "example.")

pysbd handles all of these correctly with rule-based disambiguation.

**Example:**

```python
import pysbd

segmenter = pysbd.Segmenter(language="en", clean=False)

# In _read_cli_response, replace regex splitting:
sentences = segmenter.segment(accumulated_text)
for sentence in sentences:
    clean = self._strip_markdown(sentence.strip())
    if clean:
        await composer.enqueue(AudioSegment(
            type=SegmentType.TTS_SENTENCE,
            data=clean
        ))
```

### Pattern 5: Category Exemplar Phrases for Semantic Matching

**What:** A JSON file with 5-10 representative phrases per category. These are embedded at daemon startup and used as comparison targets for semantic similarity.

**Key design:** Exemplars should cover the variety of phrasings within each category. Include both direct and indirect forms.

**Example (category_exemplars.json):**

```json
{
    "task": [
        "fix the bug in the login page",
        "run the test suite",
        "deploy the app to production",
        "refactor the auth module",
        "could you take a peek at this",
        "clean up the old files",
        "set up the database",
        "go ahead and merge that",
        "can you look into this issue",
        "try restarting the server"
    ],
    "question": [
        "what does this function do",
        "how does the authentication work",
        "why is the test failing",
        "when was this last deployed",
        "can you explain this error",
        "what do you think about this approach",
        "how would you handle this",
        "is there a better way to do this"
    ],
    "conversational": [
        "the weather is nice today",
        "I had a long day",
        "just thinking out loud",
        "interesting thought",
        "that reminds me of something",
        "I was reading about this",
        "random question",
        "so I was thinking"
    ],
    "social": [
        "hey how's it going",
        "good morning",
        "see you later",
        "thanks for your help",
        "appreciate it",
        "hello there",
        "take care",
        "goodbye for now"
    ],
    "emotional": [
        "this is so frustrating",
        "that's amazing news",
        "I really appreciate your help",
        "this sucks",
        "I'm so excited about this",
        "oh that's terrible",
        "you're the best",
        "I hate when this happens"
    ],
    "acknowledgment": [
        "okay",
        "got it",
        "sure",
        "makes sense",
        "understood",
        "alright then",
        "sounds good",
        "perfect"
    ]
}
```

### Anti-Patterns to Avoid

- **Stream composer as a separate process:** IPC overhead for what is queue management within the same event loop. Process isolation provides no benefit here since the composer needs real-time access to `generation_id`, audio queues, and TTS functions.
- **Loading model2vec in the main process:** The classifier daemon is a separate process specifically so models can load without affecting main process memory/startup. Load model2vec there.
- **Using `StaticModelForClassification.fit()` instead of cosine similarity:** The classification API requires scikit-learn as a dependency and a training step. For 6 categories with 5-10 exemplars each, cosine similarity is simpler, has no extra dependency, and runs in <1ms.
- **Replacing pysbd sentence splitting in the LLM text accumulation path:** The current regex-based splitting in `_read_cli_response()` works for streaming text deltas where sentences arrive incrementally. pysbd is better for splitting complete text blocks. Use pysbd for the final sentence split, keep regex for streaming delta accumulation.
- **Hard-coding pause durations:** The cadence manager should have tunable parameters, not magic numbers scattered through the code. Centralize all timing constants in the composer class.
- **Blocking TTS generation in the composer:** The composer must call Piper TTS asynchronously. If it blocks on TTS for sentence N, it can't process barge-in or accept new segments. Use `asyncio.create_subprocess_exec` (already the pattern in `_tts_stage`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentence boundary detection | Custom regex beyond current `SENTENCE_END_RE` | `pysbd.Segmenter` | Handles abbreviations, decimals, URLs. Rule-based, zero model loading, 22 languages. |
| Semantic text embeddings | Word2vec, TF-IDF, custom embeddings | model2vec `StaticModel` | 8MB, <1ms encode, only needs numpy. Purpose-built for this exact use case. |
| Cosine similarity | sklearn `cosine_similarity`, custom distance | `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` | numpy is already installed. Three lines of code vs adding sklearn as dependency. |
| Backchannel/trivial detection | ML-based backchannel classifier | Word-list lookup with context flag | Linguistic research shows backchannels come from a small, closed set of lexical items. A word list covers 95%+ of cases. |
| Audio pause insertion | Manual byte calculation scattered through code | Composer class with `_emit_silence(duration)` | Centralizes all timing/silence logic. Duration in seconds, not bytes. |
| Non-speech audio clips | Generate at runtime via TTS | Pre-recorded WAV clips shipped with project | "Hmm" thinking sounds and breath pauses sound unnatural from TTS. Pre-recorded clips with natural prosody are more convincing. |

**Key insight:** The semantic matching component is small (model2vec encode + cosine sim, ~10 lines of new code in the classifier). The stream composer is the bulk of new work, but it replaces and simplifies existing scattered logic (filler playback, TTS staging, sentence tracking, pause insertion) into a single coherent class.

## Common Pitfalls

### Pitfall 1: model2vec Download Blocks Classifier Startup

**What goes wrong:** The first time the classifier daemon starts, `StaticModel.from_pretrained("minishlab/potion-base-8M")` downloads the model from HuggingFace (~8MB). This can take 5-30 seconds on slow connections, delaying the CLASSIFIER_READY signal and causing the session to start without classification.

**Why it happens:** HuggingFace downloads models on first access. The daemon's startup is synchronous.

**How to avoid:** Two strategies:
1. Pre-download during installation (add to setup/install script): `python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"`
2. Graceful degradation: emit CLASSIFIER_READY after heuristic patterns are compiled, before model2vec loads. If semantic model isn't ready yet, heuristic-only classification still works. Load model2vec in a background thread and set a flag when ready.

**Warning signs:** First session after install has classifier timeout. Subsequent sessions work fine.

### Pitfall 2: Cosine Similarity Threshold Mismatch with Heuristic Confidence

**What goes wrong:** Heuristic confidence (0.0-1.0 based on regex match count) and cosine similarity (0.0-1.0 range but typically 0.2-0.8 for sentence embeddings) have different distributions. A cosine similarity of 0.5 does not mean the same "confidence" as a heuristic confidence of 0.5.

**Why it happens:** Cosine similarity between sentence embeddings rarely exceeds 0.8 even for very similar sentences. Heuristic confidence can reach 0.95 from multiple regex matches.

**How to avoid:** Normalize semantic scores before comparing with heuristic confidence. Map cosine sim to a confidence scale:
- cosine >= 0.6: high confidence (map to 0.8-0.9)
- cosine 0.4-0.6: moderate confidence (map to 0.5-0.7)
- cosine < 0.4: low confidence (map to 0.2-0.4)

Exact thresholds should be tuned empirically. Log both raw cosine sim and mapped confidence for Phase 10 learning loop.

**Warning signs:** Semantic matching always wins over heuristics (if raw cosine sim is used as confidence) or never wins (if thresholds are wrong). Check logs for the distribution of scores.

### Pitfall 3: Stream Composer Creates Audio Gaps at Segment Boundaries

**What goes wrong:** When transitioning from filler clip to first TTS sentence, there's a perceptible audio gap (>200ms of silence) that sounds like the system "hiccupped."

**Why it happens:** The composer finishes emitting the filler clip PCM, then calls `_tts_fn()` to synthesize the first sentence. Piper TTS takes 200-400ms to produce the first byte. During this time, no audio is being written to `_audio_out_q`, creating silence.

**How to avoid:** Pre-buffer: while the filler clip is playing, start generating TTS for the first sentence in parallel. The composer should look ahead one segment in the queue and begin TTS generation while the current segment is still playing. This requires maintaining a "next segment" preview.

```python
# While emitting current segment, start preparing next:
next_segment = await self._peek_next()
if next_segment and next_segment.type == SegmentType.TTS_SENTENCE:
    # Start TTS generation in parallel
    tts_task = asyncio.create_task(self._tts_fn(next_segment.data))
```

**Warning signs:** Noticeable silence between filler clip and first AI sentence. Users perceive "two separate audio events" instead of one continuous response.

### Pitfall 4: Trivial Detection Suppresses Real Answers

**What goes wrong:** User says "yes" in response to AI's question "Should I deploy to production?" and the system treats it as trivial (no filler, just silence). The AI still processes it, but the lack of acknowledgment feels broken.

**Why it happens:** The trivial detector doesn't know the AI asked a question. "Yes" is in the trivial word list.

**How to avoid:** Implement the context flag: `ai_asked_question`. After the AI speaks a response containing a question mark or question-pattern ("should I", "do you want", "would you like"), set this flag. The next user utterance clears it. When the flag is set, `is_trivial()` returns False regardless of input.

This requires the classifier daemon to receive the `ai_asked_question` flag as part of the classification request (add it to the JSON IPC message).

**Warning signs:** After AI asks a question, user's "yes"/"no" response gets no filler. The interaction feels like the system didn't hear them.

### Pitfall 5: Barge-In During Filler Clip Causes Double Audio

**What goes wrong:** User barges in during a filler clip. The existing `_trigger_barge_in()` cancels all queued frames and increments `generation_id`. But if the stream composer has already written some frames to `_audio_out_q` and is mid-emission, the playback stage may play a few stale frames after the barge-in.

**Why it happens:** The composer writes frames in chunks (4096 bytes). Between chunks, there's an `await` where the event loop runs. Barge-in fires during this window, incrementing `generation_id`. But frames already in `_audio_out_q` have the old generation_id and are discarded by the playback stage. The real issue is if the composer continues writing frames after barge-in because it hasn't checked `generation_id` yet.

**How to avoid:** The composer's `_emit_pcm()` must check `generation_id` before each chunk write (already shown in the pattern above). Additionally, the `_trigger_barge_in()` method should call `composer.pause()` to immediately stop the composer from writing more frames.

**Warning signs:** Brief audio glitch (a few ms of stale audio) after barge-in trigger.

### Pitfall 6: pysbd Import Cost at Daemon Startup

**What goes wrong:** Importing pysbd and creating a Segmenter adds ~100-200ms to the classifier daemon startup time, which is already tight (3s timeout for readiness).

**Why it happens:** pysbd loads language-specific rules on import.

**How to avoid:** pysbd is used in the main process (for sentence splitting in `_read_cli_response`), NOT in the classifier daemon. The classifier daemon handles input classification only. Sentence splitting happens where LLM text is accumulated -- in `live_session.py`. This keeps the classifier daemon lean.

**Warning signs:** Classifier readiness timeout on slower hardware.

## Code Examples

### Modified Classifier Daemon IPC Protocol

The classifier request now includes an `ai_asked_question` flag, and the response includes a `trivial` flag.

```python
# Request (from live_session.py):
{"text": "yes", "ai_asked_question": false}

# Response (from classifier daemon):
{
    "category": "acknowledgment",
    "confidence": 0.9,
    "original_text": "yes",
    "subcategory": "affirmative",
    "trivial": true,
    "match_type": "heuristic"  # or "semantic"
}
```

### Modified _filler_manager() with Trivial Detection

```python
async def _filler_manager(self, user_text: str, cancel_event: asyncio.Event):
    """Play a context-appropriate quick response while waiting for LLM."""
    # Step 1: Classify via daemon IPC (<5ms)
    classification = await self._classify_input(user_text)
    category = classification.get("category", "acknowledgment")
    confidence = classification.get("confidence", 0.0)
    is_trivial = classification.get("trivial", False)

    # Step 2: Trivial input -> natural silence (no filler)
    if is_trivial:
        self._log_event("classification",
            input_text=user_text, category=category,
            confidence=confidence, trivial=True,
            clip_id=None, clip_phrase=None,
        )
        # Brief visual indicator
        self._set_status("thinking")  # Subtle flash
        return

    # Step 3: Low confidence -> semantic fallback already handled by daemon
    # If still low confidence after semantic, fall back to acknowledgment
    if confidence < 0.4:
        category = "acknowledgment"

    # Step 4-7: Same as current (lookup clip, log, gate, play)
    # But route through stream composer instead of direct _play_filler_audio()
    ...
```

### Non-Speech Audio Clips

Pre-recorded WAV clips for natural pacing elements. Generated once (Piper TTS with careful parameter tuning or sourced from royalty-free audio).

```python
# Non-speech clip types and usage
NON_SPEECH_CLIPS = {
    "thinking": [
        "hmm_thinking_001.wav",   # 0.3s gentle "hmm"
        "hmm_thinking_002.wav",   # 0.4s slightly longer thinking hum
    ],
    "breath": [
        "breath_001.wav",          # 0.2s subtle inhale
        "breath_002.wav",          # 0.15s brief breath
    ],
    "transition": [
        "soft_tone_001.wav",       # 0.1s very subtle tonal cue
    ],
}
```

### Cadence Manager Context Awareness

```python
class StreamComposer:
    def _choose_pause_duration(self, context: dict) -> float:
        """Choose pause duration based on conversation context.

        Rapid back-and-forth: shorter pauses (100ms)
        Deep explanation: longer pauses (200-400ms)
        After filler clip: longer pause (300ms)
        """
        recent_turn_gap = context.get("avg_turn_gap", 2.0)  # seconds

        if recent_turn_gap < 1.5:
            # Rapid exchange -- shorter pauses feel natural
            return 0.1
        elif recent_turn_gap > 4.0:
            # Slow, thoughtful conversation -- longer pauses
            return 0.3
        else:
            return 0.15  # Default moderate pacing
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Random filler from single pool | Category-aware clip from heuristic classifier (Phase 8) | Phase 8 | Users hear contextually appropriate responses |
| Heuristic-only classification with ack fallback | Heuristic + model2vec semantic fallback | Phase 9 | Paraphrased/ambiguous inputs still classified correctly |
| All inputs get a filler clip | Trivial inputs get natural silence | Phase 9 | "yes"/"ok" don't trigger unnecessary audio |
| TTS stage writes directly to audio_out_q | Stream composer manages unified sentence queue | Phase 9 | Smooth clip-to-LLM transitions, natural pacing, barge-in awareness |
| Regex sentence splitting (`[.!?]\s`) | pysbd rule-based sentence boundary detection | Phase 9 | No false splits on abbreviations, decimals, URLs |
| No inter-sentence pauses | Variable pauses chosen by cadence manager | Phase 9 | Speech sounds deliberate, not rushed |

**Deprecated/outdated after Phase 9:**
- Direct writes from `_play_filler_audio()` and `_tts_stage()` to `_audio_out_q` -- replaced by stream composer as single writer
- The simple `SENTENCE_END_RE` regex for final sentence splitting -- replaced by pysbd (though regex may remain for incremental streaming delta accumulation)

## Open Questions

1. **Stream composer: asyncio class vs subprocess**
   - What we know: CONTEXT.md says "new subprocess: stream composer / cadence manager -- separate from classifier daemon." However, the composer needs real-time access to in-process state (generation_id, audio queues).
   - What's unclear: Is the user's intent "separate concern" (class isolation) or "separate process" (process isolation)?
   - Recommendation: Implement as an asyncio class. If process isolation is truly needed, refactor later. Class gives the same API separation without IPC overhead.

2. **model2vec model choice: 8M vs 2M vs 32M**
   - What we know: potion-base-8M is ~8MB, 30ms load, <1ms encode. potion-base-2M is ~4MB but lower accuracy. potion-base-32M is ~30MB but higher accuracy.
   - What's unclear: How much accuracy difference matters for 6 coarse categories.
   - Recommendation: Start with potion-base-8M. If classification accuracy is insufficient, upgrade to 32M. The API is identical.

3. **Non-speech audio clip sourcing**
   - What we know: CONTEXT.md says non-speech elements sourced from "pre-recorded clips (shipped as library files)." Piper can generate "hmm" sounds but they sound robotic.
   - What's unclear: How to source natural-sounding thinking/breath clips. Options: record manually, use royalty-free audio libraries, or generate with Piper and post-process with effects.
   - Recommendation: Start with Piper-generated clips for "hmm" sounds (they work acceptably for brief sounds). Record or source breath clips separately. The clip set is small (5-10 clips total) and can be iterated.

4. **Incremental sentence splitting vs block splitting**
   - What we know: LLM text arrives as streaming deltas. Current code accumulates text and splits on regex. pysbd works on complete text, not streaming deltas.
   - What's unclear: How to integrate pysbd with streaming text accumulation.
   - Recommendation: Keep the current regex for incremental splitting during streaming (it works well enough for streaming deltas where text is partial). Use pysbd as a validation pass on the final complete text block for the post-tool buffer and remaining sentence flush.

5. **ai_asked_question flag tracking**
   - What we know: The trivial detection needs to know if the AI just asked a question.
   - What's unclear: The precise heuristic for detecting questions in AI responses. Some AI responses are statements that imply a question ("Let me know if you want me to proceed").
   - Recommendation: Simple heuristic: if the AI's last sentence ends with `?`, set the flag. If the last sentence contains "do you want", "should I", "would you like", also set it. Start simple, refine based on Phase 10 logs.

## Sources

### Primary (HIGH confidence)
- **Codebase reading** (complete):
  - `input_classifier.py`: Full file -- heuristic classifier, regex patterns, Unix socket daemon, IPC protocol
  - `response_library.py`: Full file -- ResponseLibrary class, lookup, recording usage
  - `live_session.py`: Full file -- `_filler_manager()` (lines 652-711), `_tts_stage()` (lines 1723-1790), `_playback_stage()` (lines 1794-1877), `_trigger_barge_in()` (lines 1924-2011), `_read_cli_response()` (lines 1195-1383), `run()` pipeline setup (lines 2015-2079)
  - `clip_factory.py`: Full file -- generation, seed responses, quality evaluation
  - `pipeline_frames.py`: Full file -- FrameType enum, PipelineFrame dataclass
  - `learner.py`: Full file -- daemon subprocess pattern
- **Phase 8 Research**: `08-RESEARCH.md` -- complete analysis of classifier architecture, IPC patterns, response library design
- **Phase 8 Context**: `08-CONTEXT.md` -- locked decisions for classification taxonomy and daemon architecture
- **Phase 9 Context**: `09-CONTEXT.md` -- locked decisions for semantic fallback, trivial handling, stream composer, barge-in behavior
- **v1.2 Research**: `ARCHITECTURE.md`, `FEATURES.md` -- pipeline architecture diagrams, feature analysis

### Secondary (MEDIUM confidence)
- [model2vec GitHub README](https://github.com/MinishLab/model2vec) -- API, models, benchmarks
- [model2vec Training README](https://github.com/MinishLab/model2vec/blob/main/model2vec/train/README.md) -- Classification API, training workflow
- [model2vec Releases](https://github.com/MinishLab/model2vec/releases) -- Version history, v0.7.0 latest
- [model2vec HuggingFace Blog](https://huggingface.co/blog/Pringled/model2vec) -- Architecture, performance claims
- [minishlab/potion-base-2M HuggingFace](https://huggingface.co/minishlab/potion-base-2M) -- Model card, usage
- [Minish AI Documentation](https://minish.ai/packages/model2vec/introduction) -- Installation, usage
- [pySBD GitHub](https://github.com/nipunsadvilkar/pySBD) -- Sentence boundary detection, rule-based, 22 languages
- [model2vec PyPI](https://pypi.org/project/model2vec/) -- Package info

### Tertiary (LOW confidence)
- [Backchannel linguistics research (ACL 2025)](https://aclanthology.org/2025.acl-long.743.pdf) -- Backchannel prediction in human-machine conversations. Confirms backchannels are a small closed set.
- [MDPI 2025 backchannel distribution study](https://www.mdpi.com/2226-471X/10/8/194) -- Timing and distribution of verbal backchannels
- [Speechmatics 2025 article](https://www.speechmatics.com/company/articles-and-news/voice-ai-doesnt-need-to-be-faster-it-needs-to-read-the-room) -- Voice AI pacing design philosophy
- [RealtimeTTS library](https://github.com/KoljaB/RealtimeTTS) -- Reference for sentence-segmented TTS streaming patterns

## Metadata

**Confidence breakdown:**
- Semantic matching (model2vec): HIGH -- API verified via official docs, model sizes confirmed, dependency chain clean (only numpy). The encode + cosine sim approach is standard and well-documented.
- Trivial detection: HIGH -- Based on established linguistics research on backchannels. Word-list approach is simple and well-understood. Context flag is straightforward.
- Stream composer architecture: HIGH -- Based on deep reading of existing pipeline code. All integration points verified in source. The composer replaces well-understood existing code paths.
- Non-speech audio clips: MEDIUM -- Sourcing natural-sounding clips needs experimentation. Piper-generated "hmm" sounds are acceptable but not ideal.
- Sentence splitting (pysbd): HIGH -- Rule-based library, well-tested, spaCy universe listed. Edge cases documented.
- Cadence manager tuning: LOW -- Pause duration values are initial guesses. Need empirical tuning during testing.

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (model2vec API is stable at v0.7.x; pysbd hasn't had breaking changes in years)
