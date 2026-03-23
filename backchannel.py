#!/usr/bin/env python3
"""
Haiku-driven contextual backchannel response generator.

Generates ultra-short (1-5 word) context-aware backchannel responses using
Claude Haiku via the Anthropic Messages API, then synthesizes them to audio
via Piper TTS.

This module is self-contained -- no imports from live_session.py or other
project modules to avoid circular import risks.

Usage:
    from backchannel import generate_backchannel, generate_backchannel_tts

    text = await generate_backchannel("what's the weather like?")
    # -> "Good question"

    result = await generate_backchannel_tts("what's the weather like?")
    # -> ("Good question", b'<raw 22050Hz 16-bit PCM>')
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("backchannel")

# Piper TTS paths (same pattern as clip_factory.py)
PIPER_CMD = str(Path.home() / ".local" / "share" / "push-to-talk" / "venv" / "bin" / "piper")
PIPER_MODEL = str(Path.home() / ".local" / "share" / "push-to-talk" / "piper-voices" / "en_GB-cori-high.onnx")

# Haiku model for fast backchannel generation
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# System prompt kept short to minimize latency
SYSTEM_PROMPT = (
    "You generate ultra-short backchannel responses (1-5 words) to acknowledge "
    "what someone just said before a fuller response arrives. Match the tone: "
    "casual for chat, attentive for questions, energetic for tasks.\n\n"
    "RULES:\n"
    "- Output ONLY the spoken words. Nothing else.\n"
    "- Never output stage directions, descriptions, or meta-commentary "
    "(e.g. \"silent\", \"nods\", \"pauses\", \"thinking\", \"laughs\").\n"
    "- Never use quotes, asterisks, brackets, or parentheses.\n"
    "- Never use filler words like \"um\" or \"uh\".\n"
    "- Only punctuation allowed: ! or ?\n\n"
    "Examples: Oh interesting, Yeah for sure, Good question, On it, Ha nice"
)

# Responses that are meta-descriptions, not actual spoken words
_BLOCKED_RESPONSES = frozenset({
    "silent", "silence", "nods", "nod", "pauses", "pause", "thinking",
    "laughs", "laugh", "smiles", "smile", "listens", "listening",
    "waits", "waiting", "hmm", "...", "considers",
})


# Module-level singleton client (lazy-initialized)
_client = None


def _get_client():
    """Get or create the Anthropic async client singleton."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.AsyncAnthropic()
    return _client


def _validate_response(text: str) -> str | None:
    """Filter out meta-descriptions and stage directions from Haiku output."""
    import re
    # Strip quotes, asterisks, brackets, parentheses
    text = re.sub(r'[*\[\]()"\']', '', text).strip()
    # Check against blocklist (case-insensitive, whole-response match)
    if text.lower() in _BLOCKED_RESPONSES:
        logger.debug("Blocked backchannel response: %r", text)
        return None
    # Reject if it looks like a stage direction: *action* or [action]
    if not text or len(text) > 40:
        return None
    return text


async def generate_backchannel(
    user_text: str,
    category: str = "",
    last_assistant_text: str = "",
) -> str | None:
    """Generate a short backchannel response (1-5 words) for the given user text.

    Args:
        user_text: The user's transcript text.
        category: Optional classification category (unused currently, for future tuning).
        last_assistant_text: Optional last assistant response for context.

    Returns:
        Short backchannel text string, or None on any error.
    """
    try:
        client = _get_client()

        # Build user message with optional context
        user_message = user_text
        if last_assistant_text:
            context_snippet = last_assistant_text[-100:]
            user_message = f'[Context: you just said "{context_snippet}"]\n{user_text}'

        response = await client.messages.create(
            model=HAIKU_MODEL,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=20,
            temperature=0.7,
        )

        if response.content:
            text = response.content[0].text.strip()
            if text:
                cleaned = _validate_response(text)
                if cleaned:
                    return cleaned

        return None
    except Exception as e:
        logger.warning("Backchannel generation failed: %s", e)
        return None


async def generate_backchannel_tts(
    user_text: str,
    category: str = "",
    last_assistant_text: str = "",
) -> tuple[str, bytes] | None:
    """Generate a backchannel response and synthesize it to audio via Piper TTS.

    Args:
        user_text: The user's transcript text.
        category: Optional classification category.
        last_assistant_text: Optional last assistant response for context.

    Returns:
        Tuple of (text, pcm_bytes) where pcm_bytes is raw 22050Hz 16-bit PCM,
        or None if generation or TTS fails.
    """
    text = await generate_backchannel(
        user_text, category=category, last_assistant_text=last_assistant_text
    )
    if text is None:
        return None

    try:
        process = await asyncio.create_subprocess_exec(
            PIPER_CMD, '--model', PIPER_MODEL, '--output-raw',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(input=text.encode()),
            timeout=2.0,
        )
        if stdout:
            return (text, stdout)
    except asyncio.TimeoutError:
        logger.warning("Piper TTS timed out for backchannel: %r", text)
    except Exception as e:
        logger.warning("Piper TTS error for backchannel: %s", e)

    return None
