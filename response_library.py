"""
Categorized audio response library for push-to-talk.

Manages a pool of pre-generated audio clips organized by category (task,
question, conversational, social, emotional, acknowledgment).  Provides
lookup with no-repeat guard, usage tracking, and atomic persistence.

This module is self-contained -- no imports from live_session.py or other
project modules.  It can be imported independently by live_session.py,
clip_factory.py, or test scripts.

Usage:
    from response_library import ResponseLibrary

    lib = ResponseLibrary()
    lib.load()
    entry = lib.lookup("task")
    pcm = lib.get_clip_pcm(entry.id) if entry else None
"""

import json
import os
import random
import time
import wave
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ResponseEntry:
    id: str                    # unique identifier like "task_on_it_001"
    category: str              # one of the 6 categories
    subcategory: str           # e.g. "greeting", "farewell", "frustration", ""
    phrase: str                # the spoken text
    filename: str              # WAV filename within category directory
    use_count: int = 0         # times this clip has been played
    barge_in_count: int = 0    # times user interrupted this clip
    last_used: float = 0.0     # timestamp of last use


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESPONSES_DIR = Path(__file__).parent / "audio" / "responses"
LIBRARY_META = RESPONSES_DIR / "library.json"
CATEGORIES = ["task", "question", "conversational", "social", "emotional", "acknowledgment"]


# ---------------------------------------------------------------------------
# ResponseLibrary
# ---------------------------------------------------------------------------

class ResponseLibrary:
    """Manages categorized audio response clips with no-repeat lookup."""

    def __init__(self):
        self._entries: list[ResponseEntry] = []
        self._index: dict[str, list[ResponseEntry]] = {}  # category -> entries
        self._clips: dict[str, bytes] = {}                 # entry_id -> pcm_bytes
        self._recent: dict[str, deque] = {}                # category -> recent IDs
        self._loaded: bool = False

    def load(self):
        """Load library from library.json + WAV files.

        Call at session startup.  Handles missing library.json gracefully
        (library stays empty).  Handles missing WAV files gracefully (skips
        the entry and prints a warning).
        """
        self._entries.clear()
        self._index.clear()
        self._clips.clear()
        self._recent.clear()
        self._loaded = False

        if not LIBRARY_META.exists():
            return

        try:
            raw = json.loads(LIBRARY_META.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"ResponseLibrary: failed to read {LIBRARY_META}: {e}", flush=True)
            return

        for entry_data in raw.get("entries", []):
            try:
                entry = ResponseEntry(
                    id=entry_data["id"],
                    category=entry_data["category"],
                    subcategory=entry_data.get("subcategory", ""),
                    phrase=entry_data["phrase"],
                    filename=entry_data["filename"],
                    use_count=entry_data.get("use_count", 0),
                    barge_in_count=entry_data.get("barge_in_count", 0),
                    last_used=entry_data.get("last_used", 0.0),
                )
            except (KeyError, TypeError) as e:
                print(f"ResponseLibrary: skipping malformed entry: {e}", flush=True)
                continue

            # Load PCM audio from WAV file
            clip_path = RESPONSES_DIR / entry.category / entry.filename
            if not clip_path.exists():
                print(f"ResponseLibrary: missing clip {clip_path}, skipping", flush=True)
                continue

            try:
                with wave.open(str(clip_path), "rb") as wf:
                    pcm = wf.readframes(wf.getnframes())
                self._clips[entry.id] = pcm
            except Exception as e:
                print(f"ResponseLibrary: error reading {clip_path}: {e}", flush=True)
                continue

            self._entries.append(entry)
            self._index.setdefault(entry.category, []).append(entry)

        if self._entries:
            self._loaded = True
            counts = {cat: len(self._index.get(cat, [])) for cat in CATEGORIES}
            active = {k: v for k, v in counts.items() if v > 0}
            print(f"ResponseLibrary: loaded {len(self._entries)} clips {active}", flush=True)

    def is_loaded(self) -> bool:
        """Return whether library has any clips loaded."""
        return self._loaded

    def lookup(self, category: str, subcategory: str = "") -> ResponseEntry | None:
        """Find a clip for the given category, avoiding recent repeats.

        Falls back to acknowledgment if the requested category has no clips.
        For emotional category with small sub-pools (<= 2), expands candidates
        to all emotional clips to avoid deterministic alternation.

        Returns None if no clips are available at all.
        """
        candidates = self._index.get(category, [])

        # Fallback to acknowledgment if category empty
        if not candidates and category != "acknowledgment":
            candidates = self._index.get("acknowledgment", [])
        if not candidates:
            return None

        # Subcategory filtering
        if subcategory:
            sub_matches = [c for c in candidates if c.subcategory == subcategory]
            if sub_matches:
                # For emotional with small sub-pools, include all emotional clips
                # to avoid deterministic alternation (pitfall 6)
                if category == "emotional" and len(sub_matches) <= 2:
                    candidates = list(candidates)  # use full category pool
                else:
                    candidates = sub_matches

        # No-repeat guard using a deque of recently used IDs
        cache_key = f"{category}:{subcategory}" if subcategory else category
        maxlen = max(1, len(candidates) - 1)
        if cache_key not in self._recent:
            self._recent[cache_key] = deque(maxlen=maxlen)
        else:
            # Resize if candidate pool changed (e.g. after reload)
            recent = self._recent[cache_key]
            if recent.maxlen != maxlen:
                self._recent[cache_key] = deque(recent, maxlen=maxlen)

        recent = self._recent[cache_key]
        available = [c for c in candidates if c.id not in recent]
        if not available:
            recent.clear()
            available = candidates

        pick = random.choice(available)
        recent.append(pick.id)
        return pick

    def get_clip_pcm(self, entry_id: str) -> bytes | None:
        """Return cached PCM bytes for an entry, or None if not loaded."""
        return self._clips.get(entry_id)

    def record_usage(self, entry_id: str, barged_in: bool = False):
        """Record that a clip was played.  Updates use_count, last_used,
        and optionally barge_in_count on the entry."""
        for entry in self._entries:
            if entry.id == entry_id:
                entry.use_count += 1
                entry.last_used = time.time()
                if barged_in:
                    entry.barge_in_count += 1
                return

    def save(self):
        """Atomically write library.json (write .tmp then rename).

        Preserves updated use_count, barge_in_count, last_used for each entry.
        """
        RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "entries": [
                {
                    "id": e.id,
                    "category": e.category,
                    "subcategory": e.subcategory,
                    "phrase": e.phrase,
                    "filename": e.filename,
                    "use_count": e.use_count,
                    "barge_in_count": e.barge_in_count,
                    "last_used": e.last_used,
                }
                for e in self._entries
            ],
        }

        tmp_path = LIBRARY_META.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2) + "\n")
        os.rename(str(tmp_path), str(LIBRARY_META))

    def reload(self):
        """Re-load library from disk.  Picks up new clips added by seed
        generation or the curator daemon.  Clears cached PCM and index."""
        self.load()

    @property
    def entries(self) -> list[ResponseEntry]:
        """Read-only access to all entries."""
        return list(self._entries)

    @property
    def categories(self) -> dict[str, int]:
        """Return category -> count mapping for loaded clips."""
        return {cat: len(self._index.get(cat, [])) for cat in CATEGORIES}
