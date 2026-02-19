#!/usr/bin/env python3
"""
Input classifier daemon for push-to-talk.

Classifies user speech transcripts into 6 categories via heuristic pattern
matching.  Runs as a standalone daemon process, listens on a Unix domain
socket, and returns JSON classification results in <1ms.

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
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedInput:
    category: str        # one of the 6 categories
    confidence: float    # 0.0 - 1.0
    original_text: str
    subcategory: str = ""  # e.g. "greeting", "farewell", "frustration"


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


def classify(text: str) -> ClassifiedInput:
    """Classify user input into a category. Returns in <1ms."""
    text_stripped = text.strip()
    if not text_stripped:
        return ClassifiedInput("acknowledgment", 0.3, text_stripped)

    text_lower = text_stripped.lower()
    word_count = len(text_stripped.split())

    # Short text (<=3 words, no ?) -- check acknowledgment first
    if word_count <= 3 and "?" not in text_stripped:
        for pat in PATTERNS["acknowledgment"]:
            if pat.search(text_lower):
                sub = "negative" if re.match(r"^(no|nah|nope|not really)", text_lower) else "affirmative"
                return ClassifiedInput("acknowledgment", 0.9, text_stripped, sub)

    # Score each category by counting regex matches
    scores: dict[str, int] = {}
    for category, pats in PATTERNS.items():
        for pat in pats:
            if pat.search(text_stripped):
                scores[category] = scores.get(category, 0) + 1

    if scores:
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = min(0.5 + scores[best] * 0.2, 0.95)
        subcategory = _infer_subcategory(best, text_stripped)
        return ClassifiedInput(best, confidence, text_stripped, subcategory)

    # Structural fallback: ends with ? -> question
    if text_stripped.endswith("?"):
        return ClassifiedInput("question", 0.6, text_stripped)

    # Ultimate default: acknowledgment (safe fallback -- per CONTEXT.md)
    return ClassifiedInput("acknowledgment", 0.3, text_stripped)


# ---------------------------------------------------------------------------
# Unix socket daemon
# ---------------------------------------------------------------------------

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
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
        error_resp = json.dumps({
            "category": "acknowledgment",
            "confidence": 0.0,
            "original_text": "",
            "subcategory": "",
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
    # Signal readiness to parent process
    print("CLASSIFIER_READY", flush=True)

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
