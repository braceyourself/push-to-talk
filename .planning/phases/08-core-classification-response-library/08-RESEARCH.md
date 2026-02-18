# Phase 8: Core Classification + Response Library - Research

**Researched:** 2026-02-18
**Domain:** Heuristic text classification + categorized audio response library for voice pipeline
**Confidence:** HIGH

## Summary

Phase 8 replaces the random filler selection in `_filler_manager()` with a heuristic classifier and categorized clip library. The classifier runs as its own daemon process (per CONTEXT.md decision), receives STT text via IPC, and returns a category classification. The response library organizes clips by 6 categories (task, question, conversational, social, emotional, acknowledgment) in a JSON file following the existing `ack_pool.json` pattern. A seed phrase list ships in the repo; clips are generated on first launch via Piper TTS.

The primary technical challenge is IPC latency: the classifier daemon must receive text, classify it, and return a result well within the 500ms filler gate. Unix domain sockets (already used in the codebase for MCP tool proxying) provide sub-millisecond round-trip latency for small JSON messages. Combined with the sub-millisecond heuristic classification itself, total overhead is under 5ms -- well within budget.

**Primary recommendation:** Use Unix domain sockets for classifier daemon IPC (matching the existing `_start_tool_ipc_server()` pattern), heuristic pattern matching for classification (regex + keyword lookup), and JSON-based response library with clips organized per-category under `audio/responses/`.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `re` (stdlib) | 3.12 | Regex patterns for heuristic classification | Zero dependency, sub-ms execution on short strings |
| Python `json` (stdlib) | 3.12 | Response library storage + IPC message serialization | Already used throughout codebase (`ack_pool.json`, session logs) |
| Python `asyncio` (stdlib) | 3.12 | Unix socket server/client for classifier IPC | Already powers entire pipeline, `asyncio.start_unix_server` used for tool IPC |
| Piper TTS (existing) | en_US-lessac-medium | Seed clip generation | Already installed, `clip_factory.py` provides generation + quality gating |
| `numpy` (existing) | 1.26.4 | Clip quality evaluation (RMS, clipping, silence ratio) | Already installed, used by `clip_factory.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python `wave` (stdlib) | 3.12 | WAV file read/write for clips | Already used in `_load_filler_clips()` and `clip_factory.py` |
| Python `subprocess` (stdlib) | 3.12 | Spawning classifier daemon process | Already used for learner and clip factory daemons |
| Python `dataclasses` (stdlib) | 3.12 | ClassifiedInput and ResponseEntry typed structures | Already used for PipelineFrame |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Unix domain socket IPC | stdin/stdout pipes | Pipes require maintaining a streaming protocol; Unix sockets allow request-response pattern matching the existing tool IPC code |
| Unix domain socket IPC | Shared memory (`multiprocessing.shared_memory`) | Lower latency but requires manual synchronization (semaphores/events); overkill when socket round-trip is already <1ms |
| Unix domain socket IPC | Named pipes (FIFO) | No asyncio-native support; would need thread-based IO |
| Regex pattern matching | scikit-learn TF-IDF classifier | Adds 30MB dependency for marginal accuracy gain on 6 coarse categories |
| JSON response library | SQLite | Unnecessary for 50-200 entries; JSON matches existing `ack_pool.json` pattern |

**Installation:**
```bash
# No new dependencies. Everything is stdlib or already installed.
```

## Architecture Patterns

### Recommended Project Structure

```
push-to-talk/
├── input_classifier.py      # NEW: Classifier daemon process (standalone script)
├── response_library.py      # NEW: ResponseLibrary class + ResponseEntry dataclass
├── clip_factory.py           # MODIFIED: Add per-category generation functions
├── live_session.py           # MODIFIED: _filler_manager uses classifier + library
├── pipeline_frames.py        # UNCHANGED (no NON_SPEECH in Phase 8)
├── audio/
│   ├── fillers/
│   │   ├── acknowledgment/   # EXISTING: old ack clips (untouched, becomes unused)
│   │   └── ack_pool.json     # EXISTING: old metadata (untouched, becomes unused)
│   └── responses/            # NEW: categorized response library
│       ├── library.json      # NEW: response library index
│       ├── task/             # NEW: task clips (on_it_001.wav, etc.)
│       ├── question/         # NEW: question clips
│       ├── conversational/   # NEW: conversational clips
│       ├── social/           # NEW: social clips (greeting, farewell)
│       ├── emotional/        # NEW: emotional clips (sub-pools inside)
│       └── acknowledgment/   # NEW: acknowledgment clips (fallback)
└── seed_phrases.json         # NEW: phrase list committed to repo (not WAV files)
```

### Pattern 1: Classifier Daemon with Unix Socket IPC

**What:** The classifier runs as a separate Python process that starts at session launch, listens on a Unix domain socket, receives text, returns classification results as JSON.

**When to use:** When classification needs to run outside the main asyncio event loop as a daemon process (per CONTEXT.md decision).

**Why a daemon (not inline):** Isolates classification logic into its own process. If the classifier ever grows more complex (model2vec in Phase 9), the daemon can load models without affecting main process memory or startup time. Follows the same subprocess pattern as `learner.py` and `clip_factory.py`.

**Example:**

```python
# input_classifier.py -- runs as daemon process

import asyncio
import json
import re
import sys
from dataclasses import dataclass, asdict

@dataclass
class ClassifiedInput:
    category: str       # "task", "question", "conversational", "social", "emotional", "acknowledgment"
    confidence: float   # 0.0 - 1.0
    original_text: str
    subcategory: str = ""  # e.g., "greeting", "farewell", "frustration"

# Compiled regex patterns for classification
PATTERNS = {
    "question": [
        re.compile(r"^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did|will|has|have|was|were)\b", re.I),
        re.compile(r"\?\s*$"),
    ],
    "task": [
        re.compile(r"^(please|can you|could you|would you|go ahead|just|try)\b", re.I),
        re.compile(r"\b(run|check|find|fix|build|deploy|create|update|delete|refactor|test|look at|pull up|open|close|restart|install|show me)\b", re.I),
    ],
    "social": [
        re.compile(r"^(hey|hi|hello|howdy|yo|sup|what's up|good morning|good afternoon|good evening|good night)\b", re.I),
        re.compile(r"\b(bye|goodbye|see you|later|take care|good night|gotta go|peace)\b", re.I),
        re.compile(r"^(thanks|thank you|appreciate|cheers)\b", re.I),
    ],
    "emotional": [
        re.compile(r"\b(ugh|damn|crap|shit|fuck|dammit|argh)\b", re.I),
        re.compile(r"\b(awesome|amazing|incredible|fantastic|love it|yes!|nice|sick|dope|hell yeah)\b", re.I),
        re.compile(r"\b(thank you so much|really appreciate|you're the best|means a lot)\b", re.I),
        re.compile(r"\b(sucks|terrible|horrible|frustrated|annoying|hate)\b", re.I),
    ],
    "acknowledgment": [
        re.compile(r"^(yes|yeah|yep|yup|ok|okay|sure|got it|right|exactly|correct|mhm|uh huh|alright|cool)\b", re.I),
        re.compile(r"^(no|nah|nope|not really)\b", re.I),
    ],
}

def classify(text: str) -> ClassifiedInput:
    """Classify user input into a category. Returns in <1ms."""
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    word_count = len(text_stripped.split())

    # Short acknowledgments (< 4 words, no question mark)
    if word_count <= 3 and "?" not in text_stripped:
        for pat in PATTERNS["acknowledgment"]:
            if pat.search(text_lower):
                return ClassifiedInput("acknowledgment", 0.9, text_stripped, "affirmative")

    # Check each category's patterns
    scores = {}
    for category, pats in PATTERNS.items():
        for pat in pats:
            if pat.search(text_lower):
                scores[category] = scores.get(category, 0) + 1

    if scores:
        best = max(scores, key=scores.get)
        confidence = min(0.5 + scores[best] * 0.2, 0.95)
        return ClassifiedInput(best, confidence, text_stripped)

    # Structural fallback: ends with ? -> question
    if text_stripped.endswith("?"):
        return ClassifiedInput("question", 0.6, text_stripped)

    # Default: task (safe fallback -- matches current behavior)
    return ClassifiedInput("task", 0.3, text_stripped)


async def handle_client(reader, writer):
    """Handle a single classification request over Unix socket."""
    try:
        data = await asyncio.wait_for(reader.readline(), timeout=5.0)
        if data:
            request = json.loads(data.decode().strip())
            text = request.get("text", "")
            result = classify(text)
            response = json.dumps(asdict(result))
            writer.write(response.encode() + b"\n")
            await writer.drain()
    except Exception as e:
        error = json.dumps({"category": "acknowledgment", "confidence": 0.0,
                            "original_text": "", "error": str(e)})
        writer.write(error.encode() + b"\n")
        try:
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()


async def run_server(socket_path: str):
    server = await asyncio.start_unix_server(handle_client, socket_path)
    # Signal readiness
    print(f"Classifier: ready at {socket_path}", flush=True)
    async with server:
        await server.serve_forever()


def main():
    if len(sys.argv) < 2:
        print("Usage: input_classifier.py <socket-path>", file=sys.stderr)
        sys.exit(1)
    socket_path = sys.argv[1]
    asyncio.run(run_server(socket_path))


if __name__ == "__main__":
    main()
```

**Client side (in live_session.py):**

```python
async def _classify_input(self, text: str) -> dict:
    """Send text to classifier daemon, get classification result."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(self._classifier_socket_path),
            timeout=0.1  # 100ms connection timeout
        )
        request = json.dumps({"text": text}) + "\n"
        writer.write(request.encode())
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=0.1)
        writer.close()
        return json.loads(response.decode().strip())
    except Exception as e:
        print(f"  [classifier] IPC error: {e}", flush=True)
        return {"category": "acknowledgment", "confidence": 0.0}
```

### Pattern 2: Response Library with JSON Storage

**What:** A JSON file maps categories to clip metadata. Clips are organized in per-category subdirectories. The entire library loads into memory at startup (same pattern as existing `_load_filler_clips()`).

**When to use:** For the response library storage and lookup.

**Example:**

```python
# response_library.py

import json
import random
import wave
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

@dataclass
class ResponseEntry:
    id: str
    category: str
    subcategory: str
    phrase: str
    filename: str
    use_count: int = 0
    barge_in_count: int = 0
    last_used: float = 0.0

RESPONSES_DIR = Path(__file__).parent / "audio" / "responses"
LIBRARY_META = RESPONSES_DIR / "library.json"

class ResponseLibrary:
    def __init__(self):
        self._entries: list[ResponseEntry] = []
        self._index: dict[str, list[ResponseEntry]] = {}  # category -> entries
        self._clips: dict[str, bytes] = {}  # entry_id -> pcm_bytes
        self._recent: dict[str, deque] = {}  # category -> recent entry IDs (for no-repeat)
        self._usage_log: list[dict] = []

    def load(self):
        """Load library from JSON + WAV files. Call at session startup."""
        if not LIBRARY_META.exists():
            return
        meta = json.loads(LIBRARY_META.read_text())
        for entry_data in meta.get("entries", []):
            entry = ResponseEntry(**entry_data)
            self._entries.append(entry)
            self._index.setdefault(entry.category, []).append(entry)

            # Load PCM audio
            clip_path = RESPONSES_DIR / entry.category / entry.filename
            if clip_path.exists():
                with wave.open(str(clip_path), 'rb') as wf:
                    self._clips[entry.id] = wf.readframes(wf.getnframes())

    def lookup(self, category: str, subcategory: str = "") -> ResponseEntry | None:
        """Find best clip for category. Returns None if no clips available."""
        candidates = self._index.get(category, [])
        if not candidates:
            # Fallback to acknowledgment
            if category != "acknowledgment":
                candidates = self._index.get("acknowledgment", [])
            if not candidates:
                return None

        # Filter by subcategory if specified and matches exist
        if subcategory:
            sub_matches = [c for c in candidates if c.subcategory == subcategory]
            if sub_matches:
                candidates = sub_matches

        # Pick with no-recent-repeat (round-robin shuffle)
        recent = self._recent.setdefault(category, deque(maxlen=max(1, len(candidates) - 1)))
        available = [c for c in candidates if c.id not in recent]
        if not available:
            available = candidates  # All recently used, reset

        pick = random.choice(available)
        recent.append(pick.id)
        return pick

    def get_clip_pcm(self, entry_id: str) -> bytes | None:
        return self._clips.get(entry_id)

    def log_usage(self, entry_id: str, category: str, confidence: float,
                  input_text: str, barged_in: bool = False):
        self._usage_log.append({
            "entry_id": entry_id,
            "category": category,
            "confidence": confidence,
            "input_text": input_text,
            "barged_in": barged_in,
        })
```

### Pattern 3: Daemon Spawning (Follows learner.py Pattern)

**What:** Spawn the classifier daemon at session start, clean up at session end.

**Example:**

```python
# In LiveSession

def _spawn_classifier(self):
    """Spawn the classifier daemon process."""
    self._classifier_socket_path = f"/tmp/ptt-classifier-{os.getpid()}.sock"
    # Clean up stale socket
    if os.path.exists(self._classifier_socket_path):
        os.unlink(self._classifier_socket_path)

    classifier_script = Path(__file__).parent / "input_classifier.py"
    cmd = [sys.executable, str(classifier_script), self._classifier_socket_path]
    try:
        self._classifier_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Wait for "ready" signal from daemon
        # (read stdout for "Classifier: ready" line)
        ready_line = self._classifier_process.stdout.readline()
        print(f"Live session: Classifier spawned (PID {self._classifier_process.pid})", flush=True)
    except Exception as e:
        print(f"Live session: Failed to spawn classifier: {e}", flush=True)
        self._classifier_process = None
```

### Pattern 4: Seed Library Generation on First Launch

**What:** On first launch (or when clips are missing), generate seed clips from a committed phrase list using the same Piper TTS and quality gating as `clip_factory.py`.

**Example:**

```python
# In clip_factory.py or a new seed_generator.py

def generate_seed_library(seed_phrases: dict, responses_dir: Path):
    """Generate seed clips for all categories.
    seed_phrases: {"task": ["gotcha", "on it", ...], "question": [...], ...}
    """
    for category, phrases in seed_phrases.items():
        cat_dir = responses_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for phrase in phrases:
            # Generate with default Piper params (match live TTS)
            pcm = generate_clip(phrase, length_scale=1.0, noise_w=0.667, noise_scale=0.667)
            if pcm is None:
                continue
            scores = evaluate_clip(pcm)
            if not scores["pass"]:
                continue
            filename = _next_filename(phrase, {e["filename"] for e in entries})
            save_clip_to(pcm, filename, cat_dir)
            entries.append({
                "id": f"{category}_{filename.replace('.wav', '')}",
                "category": category,
                "subcategory": "",
                "phrase": phrase,
                "filename": filename,
                "use_count": 0,
                "barge_in_count": 0,
                "last_used": 0.0,
            })
    # Write library.json
    # ...
```

### Anti-Patterns to Avoid

- **LLM classification in the hot path:** Claude CLI takes 1-3s. Classification must be <5ms. Never use an LLM call for real-time classification.
- **Replacing the existing filler system entirely:** Keep `_pick_filler("acknowledgment")` and the existing ack pool as a last-resort fallback until the new system is proven.
- **Shared mutable state between session and seed generator:** Seed generation should complete before the session loads the library. Either generate synchronously at startup or check for completion before loading.
- **Over-complex IPC protocol:** Use simple JSON-line protocol (one request per connection). The existing tool IPC server uses exactly this pattern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Clip generation + quality gating | Custom TTS pipeline | Existing `clip_factory.py` functions (`generate_clip`, `evaluate_clip`, `save_clip_to`) | Already handles Piper invocation, RMS/duration/clipping checks, WAV writing |
| Unix socket IPC server | Custom socket handling | `asyncio.start_unix_server` / `asyncio.open_unix_connection` | Already used in codebase (`_start_tool_ipc_server`), handles backpressure, timeouts, cleanup |
| Daemon subprocess management | Custom process lifecycle | Follow `_spawn_learner()` pattern exactly (Popen + start_new_session + cleanup in finally) | Battle-tested pattern that handles PID tracking, termination, timeout |
| WAV file I/O | Custom PCM handling | `wave` stdlib module + existing `_load_filler_clips()` pattern | Already handles sample rate detection, resampling pipeline |
| JSON metadata persistence | Custom serialization | Follow `ack_pool.json` pattern (`_load_meta` / `_save_meta` from `clip_factory.py`) | Handles missing files, corrupt JSON, mkdir -p |

**Key insight:** Nearly every infrastructure piece for Phase 8 already exists in the codebase. The new work is the classification logic, the response library data structure, the seed phrase list, and wiring these together.

## Common Pitfalls

### Pitfall 1: Classifier Daemon Not Ready When First Transcript Arrives

**What goes wrong:** The classifier daemon is spawned at session start but the Unix socket may not be listening yet when the first transcript arrives 1-2 seconds later. The `_classify_input()` call fails with a connection refused error.

**Why it happens:** `subprocess.Popen` returns immediately. The daemon needs time to start Python, import modules, compile regex patterns, and call `asyncio.start_unix_server`. This takes 50-200ms on typical hardware.

**How to avoid:** Wait for a readiness signal before proceeding. The daemon prints a "ready" line to stdout when the socket is listening. Read this line synchronously during `_spawn_classifier()` before returning. Set a 3-second timeout for the readiness check.

**Warning signs:** First transcript of every session falls through to acknowledgment fallback. Classifier IPC errors in logs during first 1-2 seconds.

### Pitfall 2: Classification Falls Through to Wrong Default

**What goes wrong:** When the classifier returns low confidence, the system should use "acknowledgment" as fallback. But if the fallback logic is in the wrong place, it might use whatever low-confidence category the classifier returned (e.g., "emotional" with 0.3 confidence), playing an emotional clip for ambiguous input.

**Why it happens:** The CONTEXT.md says "best-guess matching: use top category even at moderate confidence." This is correct for moderate confidence (0.5-0.7) but there needs to be a floor. Below some threshold, even the best guess is unreliable.

**How to avoid:** Use a two-tier threshold:
- Confidence >= 0.4: Use the classified category (best-guess matching)
- Confidence < 0.4: Fall back to "acknowledgment"

The threshold is deliberately low because the CONTEXT.md says to use top category at moderate confidence, and acknowledgment is the safest fallback. Better to try the classified category than always play generic clips.

**Warning signs:** Users hearing emotional or social clips in response to ambiguous technical statements.

### Pitfall 3: Seed Generation Blocks Session Startup

**What goes wrong:** On first launch, generating 40 clips via Piper takes 40-60 seconds (each clip takes ~1-2 seconds). If seed generation runs synchronously in session startup, the user waits over a minute before they can talk.

**Why it happens:** `generate_clip()` calls Piper as a synchronous subprocess. Generating 40 clips serially is 40 sequential subprocess calls.

**How to avoid:** Run seed generation as a background process (similar to `_spawn_clip_factory()`). The session starts immediately with whatever clips exist (possibly none on first run). If no clips exist yet, fall back to existing acknowledgment pool. The seed generator writes `library.json` progressively -- generate 1 clip per category first (broad coverage), then fill remaining clips. The session can reload the library after seed generation completes via a signal file.

**Warning signs:** First-time users waiting a long time at "listening" state. Session startup logs showing 30-60 second delay.

### Pitfall 4: Audio Quality Mismatch Between Seed Clips and Live TTS

**What goes wrong:** Seed clips are generated with different Piper parameters than live TTS, causing an audible "voice change" when the filler clip transitions to the LLM response.

**Why it happens:** `clip_factory.py` randomizes `length_scale` (0.9-1.3), `noise_w_scale` (0.3-0.8), and `noise_scale` (0.4-0.7). Live TTS in `_tts_to_pcm()` uses Piper defaults (no custom parameters). The two sound different.

**How to avoid:** Generate seed clips with Piper's default parameters (no `--length-scale`, `--noise-w-scale`, `--noise-scale` flags), matching exactly what `_tts_to_pcm()` does. The variety comes from different phrases, not different voice parameters.

**Warning signs:** Users perceiving "two different voices" -- the filler sounds different from the main response.

### Pitfall 5: Library JSON Corruption on Abnormal Exit

**What goes wrong:** The seed generator or the session usage logger writes to `library.json` but the process is killed (SIGKILL, power loss) mid-write, leaving a truncated JSON file. Next session fails to load the library.

**Why it happens:** `json.dumps()` + `file.write()` is not atomic. A partial write produces invalid JSON.

**How to avoid:** Use atomic write: write to a temp file in the same directory, then `os.rename()` (atomic on Linux when same filesystem). The existing `_save_meta()` in `clip_factory.py` does NOT do this (it writes directly), but for a growing library that persists across sessions, atomic writes are important.

```python
def _save_library_atomic(meta: dict, meta_path: Path):
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.rename(str(tmp), str(meta_path))
```

**Warning signs:** `library.json` sometimes contains truncated JSON. Load errors on session startup after a crash.

### Pitfall 6: Emotional Sub-Pools Too Granular

**What goes wrong:** The CONTEXT.md specifies emotional sub-pools (frustration, excitement, gratitude, sadness) with 2-3 clips each. With only 2 clips per sub-pool, the no-repeat guard means you alternate between two clips deterministically. The user quickly notices the pattern.

**Why it happens:** 2 clips provide zero variety -- it's an alternating sequence. Even 3 clips produce noticeable patterns within a few interactions.

**How to avoid:** The emotional category should match based on sub-pool when possible, but fall back to any emotional clip when the sub-pool has few entries. This provides more variety while still maintaining rough relevance. A frustration clip is better than an acknowledgment clip for frustrated input, even if the specific frustration sub-pool is exhausted.

**Warning signs:** Users always hearing the same emotional response in sequence. Emotional clips feeling predictable after a few sessions.

## Code Examples

### Classification Logging (CONTEXT.md Requirement)

Every classification must be logged from day one. Logs accumulate in the session log file.

```python
# In _filler_manager() after classification and clip selection
self._log_event("classification",
    input_text=user_text,
    category=classification["category"],
    confidence=classification["confidence"],
    subcategory=classification.get("subcategory", ""),
    clip_id=response.id if response else None,
    clip_phrase=response.phrase if response else None,
    fallback_used=response is None,
)

# After playback completes or is cancelled, log whether barge-in happened
# (barge_in_count on the entry is updated when barge-in fires)
```

**Log entry format in session JSONL:**
```json
{
  "ts": 1771500000.0,
  "type": "classification",
  "input_text": "hey how's it going",
  "category": "social",
  "confidence": 0.85,
  "subcategory": "greeting",
  "clip_id": "social_hey_001",
  "clip_phrase": "Hey!",
  "fallback_used": false
}
```

### Modified `_filler_manager()` Integration

```python
async def _filler_manager(self, user_text: str, cancel_event: asyncio.Event):
    """Play a context-appropriate quick response while waiting for LLM."""
    # Step 1: Classify via daemon IPC (<5ms)
    classification = await self._classify_input(user_text)
    category = classification.get("category", "acknowledgment")
    confidence = classification.get("confidence", 0.0)

    # Step 2: Low confidence -> fall back to acknowledgment
    if confidence < 0.4:
        category = "acknowledgment"

    # Step 3: Lookup clip from response library (<1ms)
    response = self._response_library.lookup(category,
        subcategory=classification.get("subcategory", ""))

    # Step 4: Log classification trace
    self._log_event("classification",
        input_text=user_text,
        category=category,
        confidence=confidence,
        clip_id=response.id if response else None,
        clip_phrase=response.phrase if response else None,
        fallback_used=response is None,
    )

    # Step 5: Gate -- skip if LLM responds fast (500ms)
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=0.5)
        return
    except asyncio.TimeoutError:
        pass

    if cancel_event.is_set():
        return

    # Step 6: Play clip
    if response:
        clip_pcm = self._response_library.get_clip_pcm(response.id)
        if clip_pcm:
            # Resample if needed (22050 -> 24000)
            clip_pcm = self._resample_22050_to_24000(clip_pcm)
            await self._play_filler_audio(clip_pcm, cancel_event)
            return

    # Step 7: Ultimate fallback -- existing random ack clip
    clip = self._pick_filler("acknowledgment")
    if clip:
        await self._play_filler_audio(clip, cancel_event)
```

### Seed Phrase List (seed_phrases.json)

The phrase list committed to the repo. WAV files are generated on first launch.

```json
{
  "task": [
    "On it.",
    "Sure thing.",
    "Got it.",
    "Gotcha.",
    "Let me take a look.",
    "Working on it.",
    "One sec.",
    "Let me check."
  ],
  "question": [
    "Hmm.",
    "Good question.",
    "Let me think.",
    "Hmm, let me think about that.",
    "Oh, interesting.",
    "Let me see."
  ],
  "conversational": [
    "Hmm.",
    "Right.",
    "Well.",
    "Oh.",
    "Yeah.",
    "Huh."
  ],
  "social": [
    "Hey!",
    "Hey, what's up.",
    "Hi.",
    "See ya.",
    "Later.",
    "Of course.",
    "You bet.",
    "Anytime."
  ],
  "emotional": [
    "Ugh.",
    "Nice!",
    "Wow.",
    "Aw, thanks.",
    "Oh no.",
    "Hell yeah.",
    "That sucks.",
    "Ha."
  ],
  "acknowledgment": [
    "Gotcha.",
    "Sure.",
    "Mm-hm.",
    "Right.",
    "Okay.",
    "Got it.",
    "Yep.",
    "Alright."
  ]
}
```

**Emotional sub-pool mapping (in library.json metadata):**
```
frustration: "Ugh.", "That sucks."
excitement:  "Nice!", "Hell yeah."
gratitude:   "Aw, thanks."
sadness:     "Oh no."
general:     "Wow.", "Ha."
```

### Seed Generation Script Entry Point

```python
# Can be run standalone or imported by clip_factory.py
# python -m clip_factory --seed-responses

def generate_seed_responses():
    """Generate seed response clips from seed_phrases.json."""
    seed_path = Path(__file__).parent / "seed_phrases.json"
    responses_dir = Path(__file__).parent / "audio" / "responses"

    phrases = json.loads(seed_path.read_text())
    entries = []

    for category, phrase_list in phrases.items():
        cat_dir = responses_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in cat_dir.glob("*.wav")}

        for phrase in phrase_list:
            filename = _next_filename(phrase, existing)
            if filename in existing:
                continue  # Already generated

            # Use Piper defaults (no custom params) to match live TTS
            pcm = generate_clip(phrase, length_scale=1.0, noise_w=0.667, noise_scale=0.667)
            if pcm is None:
                continue
            scores = evaluate_clip(pcm)
            if not scores["pass"]:
                continue

            save_clip_to(pcm, filename, cat_dir)
            existing.add(filename)
            entries.append({
                "id": f"{category}_{filename.replace('.wav', '')}",
                "category": category,
                "subcategory": _infer_subcategory(category, phrase),
                "phrase": phrase,
                "filename": filename,
                "use_count": 0,
                "barge_in_count": 0,
                "last_used": 0.0,
            })

    # Merge with existing library.json or create new
    # ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Random ack clip from single pool | Category-aware clip from classified response library | Phase 8 | Users hear contextually appropriate responses instead of random task-oriented fillers |
| Inline filler selection in `_filler_manager` | Daemon-based classifier + in-process library lookup | Phase 8 | Classifier isolated in own process, extensible to model2vec in Phase 9 |
| All ack clips committed as WAV to repo | Phrase list committed, WAVs generated on first launch | Phase 8 | Smaller repo, consistent clip quality from user's Piper installation |
| Single `ack_pool.json` metadata file | Per-category directory structure + `library.json` index | Phase 8 | Organized by category, tracks usage metrics per clip |

**Deprecated/outdated after Phase 8:**
- `audio/fillers/acknowledgment/` and `ack_pool.json`: Still present as ultimate fallback but no longer actively used for new sessions. The new `audio/responses/acknowledgment/` directory replaces it with freshly generated clips.

## Open Questions

1. **Seed generation timing on first launch**
   - What we know: Generating ~46 clips takes 45-90 seconds via Piper
   - What's unclear: Should the session block until seed generation completes, or start with fallback and hot-reload when ready?
   - Recommendation: Start immediately with existing ack pool as fallback. Run seed generation in background. Use a signal file (like `learner_notify`) to tell the session when new clips are available. Session reloads library on next transcript cycle.

2. **Classifier daemon lifecycle across multiple sessions**
   - What we know: Learner and clip factory are spawned per-session and terminated at session end
   - What's unclear: Should the classifier daemon persist across sessions (for faster startup) or restart each time?
   - Recommendation: Per-session (same as learner). Simpler lifecycle, clean state. Sub-200ms startup is negligible.

3. **Emotional sub-pool matching precision**
   - What we know: With heuristic patterns, distinguishing frustration from sadness is unreliable ("this is terrible" could be either)
   - What's unclear: How precise should sub-pool matching be?
   - Recommendation: Match sub-pool when a clear keyword signals it (e.g., "ugh" -> frustration, "thanks" -> gratitude). Otherwise pick any emotional clip. Precision improves in Phase 9 with model2vec.

## Sources

### Primary (HIGH confidence)
- **Codebase reading** (complete):
  - `live_session.py`: `_filler_manager()` (lines 559-574), `_pick_filler()` (lines 545-557), `_load_filler_clips()` (lines 516-543), `_llm_stage()` (lines 1496-1564), `_start_tool_ipc_server()` (lines 862-896), `_spawn_learner()` (lines 243-257), `_spawn_clip_factory()` (lines 259-273), session startup (lines 1885-1905)
  - `clip_factory.py`: Full file -- generation, quality evaluation, pool management, Piper params
  - `learner.py`: Full file -- daemon pattern, JSONL tailing, subprocess spawning
  - `pipeline_frames.py`: Full file -- FrameType enum, PipelineFrame dataclass
  - `config.json`: Session configuration
  - `personality/core.md`, `personality/voice-style.md`: Personality for phrase tone
  - `audio/fillers/ack_pool.json`: Existing clip metadata schema
- **v1.2 Research documents**: `ARCHITECTURE.md`, `FEATURES.md`, `PITFALLS.md`, `STACK.md`, `SUMMARY.md` -- comprehensive domain research already completed

### Secondary (MEDIUM confidence)
- **Python asyncio docs** -- `asyncio.start_unix_server`, `asyncio.open_unix_connection` for Unix domain socket IPC
- **Existing codebase IPC pattern** -- `_start_tool_ipc_server()` demonstrates Unix socket server with JSON-line protocol, already in production

### Tertiary (LOW confidence)
- **Unix socket latency estimates** (<1ms for local, small messages) -- based on general systems knowledge, not measured on target hardware. Should be validated with a quick benchmark during implementation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all stdlib or existing dependencies, zero new packages
- Architecture: HIGH -- follows 3 existing patterns (learner daemon, tool IPC server, clip factory generation), verified against source
- IPC mechanism: HIGH -- Unix domain sockets already used in codebase, JSON-line protocol already proven
- Classification patterns: MEDIUM -- heuristic patterns are straightforward but accuracy on real Whisper transcripts needs testing
- Seed phrases: MEDIUM -- phrase selection matches personality ("chill assistant") but naturalness only verifiable by listening
- Pitfalls: HIGH -- derived from detailed codebase reading + v1.2 research pitfall analysis

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (stable domain, no fast-moving dependencies)
