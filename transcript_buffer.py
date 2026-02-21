"""Transcript buffer and hallucination filter for continuous STT pipeline.

Provides:
- TranscriptSegment: frozen dataclass for individual transcript entries
- TranscriptBuffer: bounded ring buffer with count-based and time-based eviction
- is_hallucination(): multi-layer hallucination detection for Whisper output
- HALLUCINATION_PHRASES: frozenset of known Whisper hallucination strings

No external dependencies beyond stdlib. Importable independently of live_session.py.
"""

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class TranscriptSegment:
    """A single transcript segment from the STT pipeline."""
    text: str
    timestamp: float  # time.time() when captured
    confidence: float = 1.0  # from Whisper logprob
    source: str = "user"  # "user", "ai", "filtered", "other"
    no_speech_prob: float = 0.0


# Expanded hallucination phrase list: merges existing 18 from live_session.py
# with research-backed additions from arXiv 2501.11378.
# Top hallucinated phrases by frequency: "thank you" (24.76%),
# "thanks for watching" (10.32%), "so" (3.80%), "thank you for watching" (2.58%),
# "the" (2.50%)
HALLUCINATION_PHRASES: frozenset = frozenset({
    # Existing phrases (from live_session.py line 2193)
    "thank you", "thanks for watching", "thanks for listening",
    "thank you for watching", "thanks for your time",
    "goodbye", "bye", "you", "the end", "to", "so",
    "please subscribe", "like and subscribe", "i'm sorry",
    "hmm", "uh", "um", "oh",
    # Additional research-backed phrases (arXiv 2501.11378)
    "the", "and", "a", "i", "it", "is",
    "thank you very much", "thanks", "okay",
    "subtitles by", "subtitles made by",
    "transcript emily beynon",
    "music", "applause", "laughter",
    "silence", "inaudible",
    "...", ".", "!", "?",
    "meow", "oh my god",
    "subscribe", "like", "share",
    "amara.org", "amara org community",
})


def is_hallucination(text: str, no_speech_prob: float = 0.0) -> bool:
    """Multi-layer hallucination check for continuous transcription.

    Four detection layers:
    1. Exact phrase match against HALLUCINATION_PHRASES (case-insensitive, punctuation-stripped)
    2. Very short text (<=2 chars after cleaning) -- likely noise
    3. Single word with elevated no_speech_prob (>0.3)
    4. Repetitive content (4+ words with <=2 unique words)

    Args:
        text: Transcribed text from Whisper
        no_speech_prob: Whisper's no_speech_probability for the segment

    Returns:
        True if the text is likely a hallucination
    """
    cleaned = text.lower().strip().rstrip('.!?,')

    # Layer 1: exact phrase match
    if cleaned in HALLUCINATION_PHRASES:
        return True

    # Layer 2: very short text (1-2 chars) -- likely noise
    if len(cleaned) <= 2:
        return True

    # Layer 3: single word with elevated no_speech_prob
    words = cleaned.split()
    if len(words) == 1 and no_speech_prob > 0.3:
        return True

    # Layer 4: repetitive content (e.g., "thank you thank you thank you")
    if len(words) >= 4:
        unique_words = set(words)
        if len(unique_words) <= 2:
            return True

    return False


class TranscriptBuffer:
    """Bounded ring buffer for transcript segments with time-based eviction.

    Uses collections.deque(maxlen=N) for count-based bounding and a
    threading.Lock for safe time-based eviction during concurrent
    append/read operations.

    Args:
        max_segments: Maximum number of segments to hold (default 200)
        max_age_seconds: Maximum age of segments in seconds (default 300.0 = 5 minutes)
    """

    def __init__(self, max_segments: int = 200, max_age_seconds: float = 300.0):
        self._buffer: deque = deque(maxlen=max_segments)
        self._max_age = max_age_seconds
        self._lock = Lock()

    def append(self, segment: TranscriptSegment):
        """Add a segment to the buffer, evicting old entries first."""
        with self._lock:
            self._evict_old()
            self._buffer.append(segment)

    def get_context(self, max_tokens: int = 2048) -> str:
        """Return buffer contents formatted for LLM consumption.

        Iterates from newest to oldest, accumulating lines until the
        character budget (~4 chars per token) is reached. Returns lines
        in chronological order.

        Args:
            max_tokens: Approximate token budget (converted to chars via *4)

        Returns:
            Formatted string with "[source] text" per line
        """
        with self._lock:
            self._evict_old()
            lines = []
            char_budget = max_tokens * 4
            char_count = 0
            for seg in reversed(self._buffer):
                line = f"[{seg.source}] {seg.text}"
                if char_count + len(line) > char_budget and lines:
                    break  # Over budget, but always include at least one line
                lines.append(line)
                char_count += len(line)
            return "\n".join(reversed(lines))

    def get_since(self, timestamp: float) -> list:
        """Return all segments after the given timestamp.

        Args:
            timestamp: Unix timestamp cutoff (exclusive)

        Returns:
            List of TranscriptSegment objects newer than timestamp
        """
        with self._lock:
            return [s for s in self._buffer if s.timestamp > timestamp]

    def clear(self):
        """Empty the buffer."""
        with self._lock:
            self._buffer.clear()

    def _evict_old(self):
        """Remove segments older than max_age_seconds from the front."""
        cutoff = time.time() - self._max_age
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def __len__(self) -> int:
        return len(self._buffer)
