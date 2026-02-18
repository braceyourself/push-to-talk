# Stack Research: Adaptive Quick Response Library

**Domain:** Context-aware audio response selection for voice conversation app
**Researched:** 2026-02-18
**Confidence:** HIGH

## Executive Summary

The adaptive quick response system replaces random acknowledgment clip selection with context-aware, AI-driven phrase matching. The existing stack (Piper TTS, Whisper STT, Silero VAD, asyncio pipeline) provides nearly everything needed. Only two small additions are recommended: **model2vec** for ultra-fast semantic matching of user input to response phrases, and **SQLite** (stdlib) for the response library metadata store. Non-speech event detection should piggyback on Whisper's existing output rather than introducing a separate audio classifier -- Whisper already produces bracketed annotations like `[Laughter]` and the existing `no_speech_prob` / `avg_logprob` filters already detect non-speech. The key architectural insight: the response library is small (dozens to low hundreds of phrases), so the matching problem is trivial and should not be over-engineered.

## Recommended Stack Additions

### Input Classification & Phrase Matching

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `model2vec` | 0.4.x (latest) | Semantic embedding of user input and response phrases for similarity matching | Only dependency is numpy (already installed). Encodes 20,000+ sentences/sec on CPU. ~8MB model file. Sub-millisecond per query against a library of hundreds of phrases. Far faster than sentence-transformers (50 sent/sec) and far lighter (8MB vs 100MB+). No GPU needed. |
| `numpy` | 1.26.4 (installed) | Cosine similarity computation on embedding vectors | Already installed. `np.dot()` on normalized vectors gives cosine similarity in microseconds for a few hundred phrases. No additional library needed for the similarity math. |

**Confidence: HIGH** -- model2vec is actively maintained (last release May 2025), only needs numpy, verified API via GitHub docs. The `potion-base-8M` model is 8MB on disk and retains ~90% of MiniLM accuracy while being 500x faster on CPU.

### Response Library Storage

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python `sqlite3` (stdlib) | 3.45.1 (verified on system) | Response library metadata: phrases, categories, audio file paths, usage stats, embeddings | Zero-dependency. FTS5 verified available. Atomic reads/writes for concurrent access from pipeline stages. Single file on disk, trivially backed up. Already the standard for local-first desktop apps. |
| JSON metadata files | N/A | Clip pool metadata (extends existing `ack_pool.json` pattern) | Existing pattern from clip_factory.py. Use for backward compatibility with the existing acknowledgment system during migration. |

**Confidence: HIGH** -- sqlite3 is stdlib, FTS5 confirmed available via `PRAGMA compile_options`.

### Audio Clip Pre-generation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Piper TTS | installed (en_US-lessac-medium.onnx) | Generate response audio clips from text phrases | Already integrated. The existing `clip_factory.py` demonstrates the exact pattern needed: subprocess Piper call, quality gating with RMS/duration/clipping checks, WAV storage. Extend this pattern rather than replacing it. |
| `asyncio` subprocess | stdlib (3.12) | Async batch generation of clips during idle periods | Already used throughout the pipeline. Use `asyncio.create_subprocess_exec()` for non-blocking Piper invocations. Can generate clips in background without blocking the voice pipeline. |

**Confidence: HIGH** -- Existing, working code in clip_factory.py.

### Non-Speech Event Detection

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Whisper (existing) | 20250625 (installed) | Detect non-speech events via transcript text and segment metadata | Whisper already outputs bracketed non-speech annotations like `[Laughter]`, `[Coughing]`, `[Music]` from its training data. The existing `_whisper_transcribe()` method returns these as text. Combined with existing `no_speech_prob` (>0.6 = non-speech) and `avg_logprob` (<-1.0 = low confidence transcription), we can classify audio into: speech, non-speech-with-annotation, and rejected-non-speech. No new model needed. |
| Silero VAD (existing) | silero_vad.onnx (installed) | Voice activity detection for distinguishing silence from non-speech sounds | Already loaded and running during playback for barge-in. Can be reused during input classification: VAD says "voice activity" but Whisper says "not speech" = non-speech vocalization (cough, laugh, sigh). |

**Confidence: MEDIUM** -- Whisper's bracketed annotations are inconsistent (trained on YouTube subtitles, not explicitly designed for this). The existing multi-layer filtering already catches most non-speech via `no_speech_prob` and `avg_logprob`. For initial implementation, combining these signals is sufficient. Dedicated audio classifiers (Whisper-AT, SenseVoice) can be added later if detection quality is insufficient.

## Alternatives Considered

### For Semantic Matching

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `model2vec` (potion-base-8M) | `sentence-transformers` (all-MiniLM-L6-v2) | 100MB model, ~50 sent/sec on CPU, pulls in PyTorch (~2GB). Massive overkill for matching against a few hundred phrases. Adds 10-20ms per encode vs sub-millisecond with model2vec. |
| `model2vec` (potion-base-8M) | TF-IDF + cosine similarity (scikit-learn) | Would need scikit-learn (~30MB dependency). TF-IDF is lexical, not semantic -- "I appreciate that" would not match "thanks" or "that's great." For a response library where semantic similarity matters, embeddings win. |
| `model2vec` (potion-base-8M) | `rapidfuzz` (fuzzy string matching) | Edit-distance based, not semantic. "Can you help me with this?" would not match "I need assistance." Good for typo correction, wrong tool for intent matching. |
| `model2vec` (potion-base-8M) | Claude API call for classification | Adds 500-2000ms latency per classification. The entire point of quick responses is sub-100ms selection. Cloud dependency defeats local-first design. |
| `model2vec` (potion-base-8M) | Rule-based keyword matching | Brittle, requires manual maintenance, misses paraphrases. Would need growing list of regex patterns per response category. |
| `model2vec` (potion-base-8M) | No matching at all (random selection like current system) | The current system is the baseline we're replacing. Context-aware selection is the entire feature. |

### For Non-Speech Detection

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Whisper existing output + heuristics | `whisper-at` (Whisper Audio Tagger) | Only 152 weekly PyPI downloads. Replaces the whisper import entirely (uses `whisper_at` instead of `whisper`). Adds 527-class AudioSet tagging but we only need ~5 categories (laugh, cough, sigh, clear, silence). The existing Whisper already gives us what we need via bracket annotations + confidence scores. |
| Whisper existing output + heuristics | SenseVoice (FunAudioLLM) | Requires `funasr` framework (~large dependency tree). Would replace Whisper entirely. 70ms for 10s of audio is fast, but we already have Whisper loaded and running. Adding a second speech model doubles memory usage for marginal non-speech detection improvement. |
| Whisper existing output + heuristics | librosa + manual feature classification | Adds ~20MB dependency (librosa). Would need to build custom classifiers from MFCC/spectral features. Training data needed. Significant engineering effort for something Whisper already handles adequately. |
| Whisper existing output + heuristics | MediaPipe Audio Classifier | Google's solution. Adds TensorFlow Lite dependency. Broad categories, not tuned for conversational non-speech sounds. |

### For Storage

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| SQLite (stdlib) | JSON files (current pattern) | JSON works for the current 10-15 clip pool. Will not scale to hundreds of phrases with usage statistics, embeddings, category metadata, and lookup queries. JSON requires loading entire file for any query. SQLite handles concurrent reads from multiple pipeline stages. |
| SQLite (stdlib) | Redis / any external DB | External service dependency for a desktop app. Must be running before the app starts. Overkill for a single-user local database of a few hundred records. |
| SQLite (stdlib) | TinyDB | Unnecessary dependency when sqlite3 is stdlib and more capable. |
| SQLite (stdlib) | pickle files | Security risk, not human-readable, no query capability. |

## What NOT to Add

| Avoid | Why | Impact if Added |
|-------|-----|-----------------|
| PyTorch | model2vec uses numpy only. Sentence-transformers would pull in PyTorch (~2GB). The existing Whisper already has torch, but model2vec avoids needing it for the matching layer. | +2GB disk, +500MB RAM, slower startup |
| scikit-learn | TF-IDF is lexical, not semantic. Would add ~30MB for worse matching quality than model2vec. | +30MB, still needs embeddings for good semantic matching |
| FAISS / Annoy / HNSWlib | Vector search indices for millions of vectors. The response library has at most a few hundred entries. Brute-force `np.dot()` on 100 vectors of 256 dimensions takes microseconds. | Unnecessary complexity, additional C++ dependency |
| A separate audio classifier model | SenseVoice, Whisper-AT, or custom CNN would add a second loaded model. Whisper already detects non-speech adequately for this use case. | +100MB-1GB model, doubled audio processing time |
| LangChain / LlamaIndex | RAG frameworks for retrieval-augmented generation. The response library is not a document corpus -- it's a small lookup table of pre-generated audio clips. | Massive dependency tree, wrong abstraction |
| Cloud embedding APIs (OpenAI, Cohere) | Adds latency (100-500ms per call), requires API key, breaks offline operation. Model2vec runs locally in sub-millisecond. | Network dependency, latency, cost |

## Installation

```bash
# New dependency (only numpy as transitive dep, already installed)
pip install model2vec

# Download the model (8MB, one-time)
python3 -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"
```

Add to `requirements.txt`:
```
# Semantic matching for quick response library
model2vec
```

No other new packages needed. SQLite is stdlib. Piper, Whisper, numpy, onnxruntime are already installed.

## Integration Points with Existing Stack

### 1. Clip Factory Extension (clip_factory.py)

The existing `clip_factory.py` generates acknowledgment clips with quality gating. Extend it to:
- Accept a phrase list (not just hardcoded `ACKNOWLEDGMENT_PROMPTS`)
- Store metadata in SQLite instead of JSON
- Compute and store model2vec embeddings alongside each phrase
- Support multiple categories (acknowledgment, empathy, agreement, humor, deflection)

```python
# Existing pattern (preserve):
pcm = generate_clip(prompt, length_scale, noise_w, noise_scale)
scores = evaluate_clip(pcm)
if scores["pass"]:
    save_clip_to(pcm, filename, clip_dir)

# New: also store embedding
from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/potion-base-8M")
embedding = model.encode([prompt])[0]  # 256-dim vector
# Store embedding in SQLite alongside clip metadata
```

### 2. Filler Selection (live_session.py)

Replace `_pick_filler()` (random selection with no-repeat guard) with context-aware selection:

```python
# Current (random):
def _pick_filler(self, category: str) -> bytes | None:
    clips = self._filler_clips.get(category)
    idx = random.choice([i for i in range(len(clips)) if i != last])
    return clips[idx]

# New (semantic match):
def _pick_response(self, user_text: str) -> bytes | None:
    user_embedding = self._model.encode([user_text])[0]
    scores = np.dot(self._phrase_embeddings, user_embedding)
    best_idx = np.argmax(scores)
    return self._response_clips[best_idx]
```

### 3. Non-Speech Detection (live_session.py)

Extend `_whisper_transcribe()` to classify input type before returning:

```python
# Existing multi-layer filtering already rejects non-speech.
# Add: when ALL segments rejected, check WHY they were rejected
# to distinguish cough/laugh from silence.

# Heuristic: VAD detected activity + Whisper rejected = non-speech vocalization
# Heuristic: Whisper text contains brackets = annotated non-speech event
# Heuristic: no_speech_prob > 0.8 + low RMS = silence
# Heuristic: no_speech_prob 0.3-0.6 + specific text patterns = uncertain
```

### 4. Pipeline Frames (pipeline_frames.py)

Add a new frame type for classified input:

```python
class FrameType(Enum):
    # ... existing types ...
    QUICK_RESPONSE = auto()  # Pre-generated clip selected by context
    INPUT_CLASSIFIED = auto()  # Classification result (speech/non-speech/event type)
```

### 5. SQLite Library Schema

```sql
CREATE TABLE phrases (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    category TEXT NOT NULL,  -- 'acknowledgment', 'empathy', 'agreement', etc.
    embedding BLOB,          -- model2vec 256-dim float32 vector (1KB)
    created_at REAL,
    last_used_at REAL,
    use_count INTEGER DEFAULT 0
);

CREATE TABLE clips (
    id INTEGER PRIMARY KEY,
    phrase_id INTEGER REFERENCES phrases(id),
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    duration REAL,
    rms REAL,
    quality_pass BOOLEAN,
    piper_params TEXT,  -- JSON: length_scale, noise_w, noise_scale
    created_at REAL
);

CREATE TABLE usage_log (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER REFERENCES clips(id),
    user_input TEXT,
    input_type TEXT,  -- 'speech', 'laughter', 'cough', etc.
    similarity_score REAL,
    timestamp REAL
);

-- FTS5 for text search across phrases
CREATE VIRTUAL TABLE phrases_fts USING fts5(text, category, content=phrases, content_rowid=id);
```

## Performance Budget

| Operation | Target | Actual (estimated) | Notes |
|-----------|--------|-------------------|-------|
| Embed user text | <5ms | <1ms | model2vec: ~0.05ms per sentence on CPU |
| Match against library | <5ms | <0.1ms | np.dot() on 100x256 matrix: microseconds |
| Total classification + selection | <20ms | <5ms | Well within the 500ms filler gate |
| Model load (startup) | <500ms | ~200ms | model2vec loads 8MB file, one-time at session start |
| Clip factory batch (10 clips) | <30s | ~10s | Piper generates ~1 clip/sec, quality gating adds negligible overhead |
| SQLite query | <5ms | <1ms | Single-file local DB, simple indexed queries |

Context: The existing filler system has a 500ms gate before playing any acknowledgment clip. The entire classification + selection pipeline must complete well within this gate to avoid adding perceptible latency. At <5ms total, this leaves 495ms of headroom.

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| Python 3.12 | All recommended additions | model2vec requires Python 3.8+. sqlite3 is stdlib. |
| model2vec 0.4.x | numpy 1.26.4 (installed) | Only dependency is numpy. No torch/tensorflow needed. |
| numpy 1.26.4 | model2vec, existing whisper, existing onnxruntime | Already installed, no version conflicts. |
| SQLite 3.45.1 | FTS5 enabled | Verified via PRAGMA compile_options. |
| Piper TTS | en_US-lessac-medium.onnx | Same model and invocation pattern as existing clip_factory.py. |

## Data Footprint

| Component | Size | Notes |
|-----------|------|-------|
| model2vec model (potion-base-8M) | ~8MB | Downloaded to HuggingFace cache on first use |
| SQLite database (100 phrases) | <1MB | Embeddings are 1KB each (256 x float32), metadata is small |
| Audio clips (100 phrases, ~2 clips each) | ~10MB | WAV files at 22050Hz, 1-3 seconds each |
| Total new disk usage | ~20MB | Negligible for a desktop app |

## Migration Path from Current System

The current system uses:
- `audio/fillers/acknowledgment/` directory with WAV files
- `audio/fillers/ack_pool.json` for metadata
- `_pick_filler()` for random selection

Migration approach:
1. **Phase 1:** Keep existing clips and metadata. Add SQLite alongside JSON. Import existing clips into SQLite.
2. **Phase 2:** Add model2vec embeddings to SQLite entries. Implement context-aware `_pick_response()` alongside `_pick_filler()`.
3. **Phase 3:** Expand phrase library beyond acknowledgments. Add new categories. Retire JSON metadata in favor of SQLite.

This allows the existing random filler system to keep working as a fallback while the new system is built incrementally.

## Sources

- [model2vec GitHub](https://github.com/MinishLab/model2vec) -- Verified API, model sizes, dependencies, inference speed (HIGH confidence)
- [model2vec PyPI](https://pypi.org/project/model2vec/) -- Latest version, release history (HIGH confidence)
- [minishlab/potion-base-8M on HuggingFace](https://huggingface.co/minishlab/potion-base-8M) -- Model specs, 8M params, 256 dimensions (HIGH confidence)
- [sentence-transformers/all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) -- Comparison baseline: 80MB, 50 sent/sec CPU (HIGH confidence)
- [Whisper-AT paper and PyPI](https://pypi.org/project/whisper-at/) -- 152 weekly downloads, replaces whisper import (MEDIUM confidence)
- [SenseVoice GitHub](https://github.com/FunAudioLLM/SenseVoice) -- Non-speech event labels, requires funasr framework (MEDIUM confidence)
- [RapidFuzz PyPI](https://pypi.org/project/RapidFuzz/) -- v3.14.3, edit-distance based, not semantic (HIGH confidence)
- [SQLite FTS5 docs](https://sqlite.org/fts5.html) -- Full-text search extension, BM25 ranking (HIGH confidence)
- [Python sqlite3 stdlib](https://docs.python.org/3.12/library/sqlite3.html) -- Verified FTS5 available on system (HIGH confidence)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper) -- no_speech_prob, avg_logprob, compression_ratio segment fields (HIGH confidence)
- Existing codebase: `clip_factory.py`, `live_session.py`, `pipeline_frames.py` -- Direct code inspection (HIGH confidence)

---
*Stack research for: Adaptive Quick Response Library*
*Researched: 2026-02-18*
