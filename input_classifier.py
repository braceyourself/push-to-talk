#!/usr/bin/env python3
"""
Input classifier daemon for push-to-talk.

Classifies user speech transcripts into 6 categories via heuristic pattern
matching with model2vec semantic fallback.  Runs as a standalone daemon
process, listens on a Unix domain socket, and returns JSON classification
results in <5ms.

Categories:
    task          - imperative/action requests ("fix the bug", "deploy it")
    question      - interrogatives ("what time is it?", "how does this work")
    conversational - casual chat, observations ("the weather is nice")
    social        - greetings, farewells, thanks ("hey!", "see ya")
    emotional     - frustration, excitement, etc. ("ugh", "nice!")
    acknowledgment - short affirmatives/negatives ("yeah", "ok", "nope")

Usage:
    python input_classifier.py /tmp/ptt-classifier.sock
"""

import asyncio
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedInput:
    category: str        # one of the 6 categories
    confidence: float    # 0.0 - 1.0
    original_text: str
    subcategory: str = ""  # e.g. "greeting", "farewell", "frustration"
    match_type: str = "heuristic"  # "heuristic" or "semantic"


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

PATTERNS = {
    "question": [
        re.compile(
            r"^(what|how|why|when|where|who|which|can|could|would|should|"
            r"is|are|do|does|did|will|has|have|was|were)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\?\s*$"),
    ],
    "task": [
        re.compile(
            r"^(please|can you|could you|would you|go ahead|just|try)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(run|check|find|fix|build|deploy|create|update|delete|refactor|"
            r"test|look at|pull up|open|close|restart|install|show me|"
            r"set up|clean up|move|copy|rename|merge|revert|push|commit)\b",
            re.IGNORECASE,
        ),
    ],
    "social": [
        re.compile(
            r"^(hey|hi|hello|howdy|yo|sup|what's up|good morning|"
            r"good afternoon|good evening)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(bye|goodbye|see you|later|take care|good night|gotta go|peace)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(thanks|thank you|appreciate|cheers)\b",
            re.IGNORECASE,
        ),
    ],
    "emotional": [
        re.compile(r"\b(ugh|damn|crap|shit|fuck|dammit|argh|damnit)\b", re.IGNORECASE),
        re.compile(
            r"\b(awesome|amazing|incredible|fantastic|love it|dope|hell yeah)\b",
            re.IGNORECASE,
        ),
        # "nice" and "sick" only when standalone/exclamatory (avoid "nice weather")
        re.compile(r"^(nice|sick)\s*[!.]*$", re.IGNORECASE),
        re.compile(
            r"\b(thank you so much|really appreciate|you're the best|means a lot)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(sucks|terrible|horrible|frustrated|annoying|hate)\b",
            re.IGNORECASE,
        ),
    ],
    "acknowledgment": [
        re.compile(
            r"^(yes|yeah|yep|yup|ok|okay|sure|got it|right|exactly|correct|"
            r"mhm|uh huh|alright|cool|mm-hm|mmhm)\s*$",
            re.IGNORECASE,
        ),
        re.compile(r"^(no|nah|nope|not really)\s*$", re.IGNORECASE),
    ],
}

# Subcategory inference patterns
_SOCIAL_GREETING = re.compile(
    r"^(hey|hi|hello|howdy|yo|sup|what's up|good morning|good afternoon|good evening)\b",
    re.IGNORECASE,
)
_SOCIAL_FAREWELL = re.compile(
    r"\b(bye|goodbye|see you|later|take care|good night|gotta go|peace)\b",
    re.IGNORECASE,
)
_EMO_FRUSTRATION = re.compile(
    r"\b(ugh|damn|crap|shit|fuck|dammit|argh|damnit|sucks|terrible|horrible|frustrated|annoying|hate)\b",
    re.IGNORECASE,
)
_EMO_EXCITEMENT = re.compile(
    r"\b(awesome|amazing|incredible|fantastic|love it|dope|hell yeah)\b|^(nice|sick)\s*[!.]*$",
    re.IGNORECASE,
)
_EMO_GRATITUDE = re.compile(
    r"\b(thank you so much|really appreciate|you're the best|means a lot)\b",
    re.IGNORECASE,
)
_EMO_SADNESS = re.compile(
    r"\b(sad|depressed|bummed|awful|miserable|oh no)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Trivial input detection
# ---------------------------------------------------------------------------

TRIVIAL_PATTERNS = frozenset({
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
})


def is_trivial(text: str, ai_asked_question: bool = False) -> bool:
    """Detect trivial backchannel input that needs no filler clip.

    Args:
        text: User's transcribed speech.
        ai_asked_question: If True, treat ANY input as real (answer to question).
    """
    if ai_asked_question:
        return False  # Context override: AI asked, user answered

    cleaned = text.strip().lower().rstrip(".!?,")

    # Must be short (<=4 words) to be trivial
    if len(cleaned.split()) > 4:
        return False

    # Anything with a question mark is real input
    if text.strip().endswith("?"):
        return False

    # If it matches task patterns, it's real input
    for pat in PATTERNS.get("task", []):
        if pat.search(cleaned):
            return False

    # Check against trivial word/phrase list
    return cleaned in TRIVIAL_PATTERNS


# ---------------------------------------------------------------------------
# Semantic fallback (model2vec)
# ---------------------------------------------------------------------------

class SemanticFallback:
    """Semantic similarity classifier using model2vec embeddings.

    Pre-computes embeddings for category exemplar phrases at init time,
    then classifies new text by cosine similarity against exemplars.
    """

    def __init__(self, exemplars_path: str):
        import numpy as np
        from model2vec import StaticModel

        self._np = np
        self.model = StaticModel.from_pretrained("minishlab/potion-base-8M")

        # Load exemplar phrases per category
        with open(exemplars_path) as f:
            exemplars = json.load(f)

        # Pre-compute embeddings for all exemplars
        self._category_embeddings: dict[str, "np.ndarray"] = {}
        for category, phrases in exemplars.items():
            embeddings = self.model.encode(phrases)  # shape: (N, dim)
            self._category_embeddings[category] = embeddings

    def classify(self, text: str) -> tuple[str, float]:
        """Return (category, normalized_confidence) via semantic similarity."""
        np = self._np
        text_emb = self.model.encode([text])[0]  # shape: (dim,)
        text_norm = np.linalg.norm(text_emb)

        best_category = "acknowledgment"
        best_score = 0.0

        for category, exemplar_embs in self._category_embeddings.items():
            # Cosine similarity against each exemplar
            norms = np.linalg.norm(exemplar_embs, axis=1) * text_norm
            sims = np.dot(exemplar_embs, text_emb) / np.maximum(norms, 1e-8)
            max_sim = float(np.max(sims))

            if max_sim > best_score:
                best_score = max_sim
                best_category = category

        # Normalize cosine similarity to confidence scale
        # (see 09-RESEARCH.md Pitfall 2: cosine sim != heuristic confidence)
        confidence = self._normalize_confidence(best_score)
        return best_category, confidence

    @staticmethod
    def _normalize_confidence(cosine_sim: float) -> float:
        """Map cosine similarity to confidence scale comparable with heuristic.

        cosine >= 0.6  -> 0.8-0.9 (high confidence)
        cosine 0.4-0.6 -> 0.5-0.7 (moderate confidence)
        cosine < 0.4   -> 0.2-0.4 (low confidence)
        """
        if cosine_sim >= 0.6:
            # Linear map [0.6, 1.0] -> [0.8, 0.9]
            return 0.8 + min((cosine_sim - 0.6) / 0.4, 1.0) * 0.1
        elif cosine_sim >= 0.4:
            # Linear map [0.4, 0.6] -> [0.5, 0.7]
            return 0.5 + (cosine_sim - 0.4) / 0.2 * 0.2
        else:
            # Linear map [0.0, 0.4] -> [0.2, 0.4]
            return 0.2 + (cosine_sim / 0.4) * 0.2


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _infer_subcategory(category: str, text: str) -> str:
    """Infer subcategory for social and emotional categories."""
    if category == "social":
        if _SOCIAL_GREETING.search(text):
            return "greeting"
        if _SOCIAL_FAREWELL.search(text):
            return "farewell"
        return "thanks"
    if category == "emotional":
        if _EMO_FRUSTRATION.search(text):
            return "frustration"
        if _EMO_EXCITEMENT.search(text):
            return "excitement"
        if _EMO_GRATITUDE.search(text):
            return "gratitude"
        if _EMO_SADNESS.search(text):
            return "sadness"
        return ""
    return ""


def classify(text: str, semantic: "SemanticFallback | None" = None) -> ClassifiedInput:
    """Classify user input into a category.

    Semantic similarity is the primary classifier (~0.1ms via model2vec).
    Minimal heuristics handle clear-cut cases: short acknowledgments and
    explicit question marks.
    """
    text_stripped = text.strip()
    if not text_stripped:
        return ClassifiedInput("acknowledgment", 0.3, text_stripped)

    text_lower = text_stripped.lower()
    word_count = len(text_stripped.split())

    # Fast path: short inputs (<=3 words, no ?) — check heuristic patterns
    if word_count <= 3 and "?" not in text_stripped:
        for pat in PATTERNS["acknowledgment"]:
            if pat.search(text_lower):
                sub = "negative" if re.match(r"^(no|nah|nope|not really)", text_lower) else "affirmative"
                return ClassifiedInput("acknowledgment", 0.9, text_stripped, sub, "heuristic")
        for pat in PATTERNS["social"]:
            if pat.search(text_lower):
                sub = _infer_subcategory("social", text_stripped)
                return ClassifiedInput("social", 0.9, text_stripped, sub, "heuristic")
        for pat in PATTERNS["emotional"]:
            if pat.search(text_lower):
                sub = _infer_subcategory("emotional", text_stripped)
                return ClassifiedInput("emotional", 0.9, text_stripped, sub, "heuristic")

    # Primary: semantic classification
    if semantic is not None:
        sem_category, sem_confidence = semantic.classify(text_stripped)

        # Boost to question if text ends with ? and semantic didn't pick question
        if text_stripped.endswith("?") and sem_category != "question":
            sem_category = "question"
            sem_confidence = max(sem_confidence, 0.7)

        sem_subcategory = _infer_subcategory(sem_category, text_stripped)
        return ClassifiedInput(
            sem_category, sem_confidence, text_stripped,
            sem_subcategory, "semantic",
        )

    # No semantic model available -- structural fallback
    if text_stripped.endswith("?"):
        return ClassifiedInput("question", 0.6, text_stripped, "", "heuristic")

    # Ultimate default: acknowledgment (safe fallback)
    return ClassifiedInput("acknowledgment", 0.3, text_stripped, "", "heuristic")


# ---------------------------------------------------------------------------
# Unix socket daemon
# ---------------------------------------------------------------------------

# Global semantic fallback instance (loaded in background thread)
_semantic_fallback: "SemanticFallback | None" = None
_semantic_lock = threading.Lock()


def _load_semantic_model(exemplars_path: str):
    """Load semantic model in background thread. Sets global when ready."""
    global _semantic_fallback
    try:
        sf = SemanticFallback(exemplars_path)
        with _semantic_lock:
            _semantic_fallback = sf
        print("SEMANTIC_READY", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Semantic model load failed: {e}", file=sys.stderr, flush=True)


def _get_semantic() -> "SemanticFallback | None":
    """Thread-safe access to the semantic fallback instance."""
    with _semantic_lock:
        return _semantic_fallback


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single classification request over Unix socket."""
    try:
        data = await asyncio.wait_for(reader.readline(), timeout=5.0)
        if data:
            request = json.loads(data.decode().strip())
            text = request.get("text", "")
            ai_asked_question = request.get("ai_asked_question", False)

            # Classify with optional semantic fallback
            semantic = _get_semantic()
            result = classify(text, semantic=semantic)

            # Detect trivial input
            trivial = is_trivial(text, ai_asked_question=ai_asked_question)

            # Build response dict
            response_dict = asdict(result)
            response_dict["trivial"] = trivial

            response = json.dumps(response_dict)
            writer.write(response.encode() + b"\n")
            await writer.drain()
    except Exception as e:
        error_resp = json.dumps({
            "category": "acknowledgment",
            "confidence": 0.0,
            "original_text": "",
            "subcategory": "",
            "match_type": "heuristic",
            "trivial": False,
            "error": str(e),
        })
        writer.write(error_resp.encode() + b"\n")
        try:
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def run_server(socket_path: str):
    """Start the Unix socket server and serve forever."""
    # Clean up stale socket
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = await asyncio.start_unix_server(handle_client, socket_path)
    # Signal readiness to parent process (heuristic classification is ready)
    print("CLASSIFIER_READY", flush=True)

    # Load semantic model in background thread (graceful degradation)
    exemplars_path = str(Path(__file__).parent / "category_exemplars.json")
    if os.path.exists(exemplars_path):
        t = threading.Thread(
            target=_load_semantic_model,
            args=(exemplars_path,),
            daemon=True,
        )
        t.start()
    else:
        print(
            f"Warning: {exemplars_path} not found, semantic fallback disabled",
            file=sys.stderr, flush=True,
        )

    async with server:
        await server.serve_forever()


def main():
    if len(sys.argv) < 2:
        print("Usage: input_classifier.py <socket-path>", file=sys.stderr)
        sys.exit(1)
    socket_path = sys.argv[1]
    try:
        asyncio.run(run_server(socket_path))
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(sys.argv[1]):
            os.unlink(sys.argv[1])


if __name__ == "__main__":
    main()
