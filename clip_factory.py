#!/usr/bin/env python3
"""
Background clip factory daemon for non-verbal filler generation.

Generates, evaluates, and manages rotating pools of audio clips via Piper TTS:
- Non-verbal fillers (hums, breaths) for thinking pauses
- Acknowledgment phrases ("let me check that") for pre-tool feedback

Can run as a one-shot pool top-up or as a background daemon that periodically
ensures the pools stay healthy.

Usage:
    python clip_factory.py              # One-shot: top up pools and exit
    python clip_factory.py --daemon     # Daemon: top up every N seconds
    python clip_factory.py --daemon --interval 600
"""

import argparse
import json
import logging
import random
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPER_CMD = str(Path.home() / ".local" / "share" / "push-to-talk" / "venv" / "bin" / "piper")
PIPER_MODEL = str(Path.home() / ".local" / "share" / "push-to-talk" / "piper-voices" / "en_US-lessac-medium.onnx")

CLIP_DIR = Path(__file__).parent / "audio" / "fillers" / "nonverbal"
POOL_META = Path(__file__).parent / "audio" / "fillers" / "pool.json"

POOL_SIZE_CAP = 20   # Maximum clips in the pool
MIN_POOL_SIZE = 10   # Generate until reaching this
SAMPLE_RATE = 22050  # Piper native sample rate

# Non-verbal prompts — must be real speakable words/phrases so Piper
# doesn't spell them out letter-by-letter (e.g. "Hm" → "H-M")
PROMPTS = ["hum", "um", "uh huh", "ah", "mm hm", "oh", "uh", "huh"]

# Acknowledgment clip pool
ACK_CLIP_DIR = Path(__file__).parent / "audio" / "fillers" / "acknowledgment"
ACK_POOL_META = Path(__file__).parent / "audio" / "fillers" / "ack_pool.json"
ACK_POOL_SIZE_CAP = 15
ACK_MIN_POOL_SIZE = 10

ACKNOWLEDGMENT_PROMPTS = [
    "Let me check that.",
    "One sec.",
    "Sure, let me look.",
    "Let me see.",
    "Give me a moment.",
    "Checking now.",
    "On it.",
    "Looking into that.",
    "Let me find out.",
    "Just a moment.",
    "Hang on.",
    "Let me pull that up.",
    "Working on it.",
    "One moment.",
    "Let me take a look.",
]

# Maximum consecutive generation failures before giving up
MAX_CONSECUTIVE_FAILURES = 20

log = logging.getLogger("clip_factory")


# ---------------------------------------------------------------------------
# Synthesis parameter generation
# ---------------------------------------------------------------------------

def random_synthesis_params() -> dict:
    """Return randomized Piper TTS parameters for nonverbal clip diversity."""
    return {
        "prompt": random.choice(PROMPTS),
        "length_scale": round(random.uniform(0.7, 1.8), 2),
        "noise_w_scale": round(random.uniform(0.3, 1.5), 2),
        "noise_scale": round(random.uniform(0.4, 1.0), 2),
    }


def random_ack_params() -> dict:
    """Return randomized Piper TTS parameters for acknowledgment phrases."""
    return {
        "prompt": random.choice(ACKNOWLEDGMENT_PROMPTS),
        "length_scale": round(random.uniform(0.9, 1.3), 2),   # Tighter for natural pace
        "noise_w_scale": round(random.uniform(0.3, 0.8), 2),
        "noise_scale": round(random.uniform(0.4, 0.7), 2),
    }


# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------

def generate_clip(prompt: str, length_scale: float, noise_w: float, noise_scale: float) -> bytes | None:
    """Generate a single clip via Piper TTS. Returns raw PCM bytes or None on failure."""
    try:
        result = subprocess.run(
            [
                PIPER_CMD,
                "--model", PIPER_MODEL,
                "--output-raw",
                "--length-scale", str(length_scale),
                "--noise-w-scale", str(noise_w),
                "--noise-scale", str(noise_scale),
            ],
            input=prompt.encode(),
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            log.warning("Piper failed (exit %d): %s", result.returncode, result.stderr[:200])
            return None
        if not result.stdout:
            log.warning("Piper returned empty output for prompt %r", prompt)
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        log.warning("Piper timed out for prompt %r", prompt)
        return None
    except Exception as e:
        log.warning("Piper error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(pcm_data: bytes, category: str = "nonverbal") -> dict:
    """Evaluate audio quality of raw PCM data. Returns scores dict with pass/fail.

    Category adjusts thresholds:
    - "nonverbal": Short hums/breaths (0.2-2.0s, RMS > 300)
    - "acknowledgment": Full phrases (0.3-4.0s, RMS > 200)
    """
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
    n = len(samples)

    if n == 0:
        return {
            "duration": 0.0,
            "rms": 0.0,
            "peak": 0,
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "pass": False,
        }

    duration = n / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = int(np.max(np.abs(samples)))
    clipping_ratio = float(np.sum(np.abs(samples) >= 32000) / n)
    silence_ratio = float(np.sum(np.abs(samples) < 500) / n)

    if category == "acknowledgment":
        passes = (
            0.3 <= duration <= 4.0    # Longer for full phrases
            and rms > 200             # Slightly lower — phrases may be quieter
            and clipping_ratio < 0.01
            and silence_ratio < 0.5   # Less silence tolerance
        )
    else:
        passes = (
            0.2 <= duration <= 2.0
            and rms > 300
            and clipping_ratio < 0.01
            and silence_ratio < 0.7
        )

    return {
        "duration": round(duration, 3),
        "rms": round(rms, 1),
        "peak": peak,
        "clipping_ratio": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 3),
        "pass": passes,
    }


# ---------------------------------------------------------------------------
# WAV file I/O
# ---------------------------------------------------------------------------

def _next_filename(prompt: str, existing_filenames: set[str]) -> str:
    """Generate the next sequential filename for a prompt."""
    prefix = prompt.lower().rstrip('.')
    # Sanitize: replace spaces with underscores, keep only alnum and underscore
    prefix = "_".join(prefix.split())
    n = 1
    while True:
        name = f"{prefix}_{n:03d}.wav"
        if name not in existing_filenames:
            return name
        n += 1


def save_clip_to(pcm_data: bytes, filename: str, clip_dir: Path) -> Path:
    """Write raw PCM data to a WAV file in the given directory. Returns the full path."""
    filepath = clip_dir / filename
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return filepath


def save_clip(pcm_data: bytes, filename: str) -> Path:
    """Write raw PCM data to a WAV file in the nonverbal directory. Returns the full path."""
    return save_clip_to(pcm_data, filename, CLIP_DIR)


# ---------------------------------------------------------------------------
# Pool metadata persistence
# ---------------------------------------------------------------------------

def _load_meta(meta_path: Path) -> list[dict]:
    """Read pool metadata JSON. Returns empty list if file missing or corrupt."""
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to read %s: %s", meta_path.name, e)
        return []


def _save_meta(meta: list[dict], meta_path: Path) -> None:
    """Write pool metadata to JSON file."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def load_pool_meta() -> list[dict]:
    """Read pool.json. Returns empty list if file missing or corrupt."""
    return _load_meta(POOL_META)


def save_pool_meta(meta: list[dict]) -> None:
    """Write pool metadata to pool.json."""
    _save_meta(meta, POOL_META)


# ---------------------------------------------------------------------------
# Pool rotation
# ---------------------------------------------------------------------------

def _rotate_pool(meta: list[dict], clip_dir: Path, size_cap: int) -> list[dict]:
    """Remove oldest clips when pool exceeds size_cap."""
    if len(meta) <= size_cap:
        return meta

    # Sort by creation time ascending (oldest first)
    meta.sort(key=lambda c: c.get("created_at", 0))

    while len(meta) > size_cap:
        oldest = meta.pop(0)
        clip_path = clip_dir / oldest["filename"]
        if clip_path.exists():
            clip_path.unlink()
            log.info("Rotated out: %s", oldest["filename"])

    return meta


def rotate_pool(meta: list[dict]) -> list[dict]:
    """Remove oldest clips when pool exceeds POOL_SIZE_CAP."""
    return _rotate_pool(meta, CLIP_DIR, POOL_SIZE_CAP)


# ---------------------------------------------------------------------------
# Generic pool top-up
# ---------------------------------------------------------------------------

def _top_up(clip_dir: Path, meta_path: Path, min_size: int, size_cap: int,
            param_fn, category: str) -> None:
    """Generic pool top-up: generate clips until pool reaches min_size."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(meta_path)

    # Reconcile: remove metadata entries for clips that no longer exist on disk
    existing_wavs = {p.name for p in clip_dir.glob("*.wav")}
    before = len(meta)
    meta = [m for m in meta if m["filename"] in existing_wavs]
    if len(meta) < before:
        log.info("Pruned %d orphaned metadata entries (%s)", before - len(meta), category)

    existing_filenames = {m["filename"] for m in meta}
    consecutive_failures = 0

    while len(meta) < min_size:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.error(
                "Too many consecutive failures (%d). %s pool at %d clips.",
                consecutive_failures, category, len(meta),
            )
            break

        params = param_fn()
        pcm = generate_clip(
            params["prompt"],
            params["length_scale"],
            params["noise_w_scale"],
            params["noise_scale"],
        )

        if pcm is None:
            consecutive_failures += 1
            log.info("Generation failed (%s), retrying (%d/%d)",
                     category, consecutive_failures, MAX_CONSECUTIVE_FAILURES)
            continue

        scores = evaluate_clip(pcm, category=category)
        if not scores["pass"]:
            consecutive_failures += 1
            log.info(
                "Rejected (%s): %s (dur=%.2fs rms=%.0f clip=%.4f sil=%.3f)",
                category, params["prompt"], scores["duration"], scores["rms"],
                scores["clipping_ratio"], scores["silence_ratio"],
            )
            continue

        consecutive_failures = 0
        filename = _next_filename(params["prompt"], existing_filenames)
        save_clip_to(pcm, filename, clip_dir)
        existing_filenames.add(filename)

        entry = {
            "filename": filename,
            "created_at": time.time(),
            "params": params,
            "scores": scores,
        }
        meta.append(entry)
        log.info(
            "Saved (%s): %s (dur=%.2fs rms=%.0f)",
            category, filename, scores["duration"], scores["rms"],
        )

    # Rotate if pre-existing clips pushed over cap
    meta = _rotate_pool(meta, clip_dir, size_cap)
    _save_meta(meta, meta_path)

    print(f"Clip factory: {category} pool at {len(meta)} clips (cap {size_cap})", flush=True)


# ---------------------------------------------------------------------------
# Public pool management functions
# ---------------------------------------------------------------------------

def top_up_pool() -> None:
    """Generate nonverbal clips until the pool reaches MIN_POOL_SIZE."""
    _top_up(CLIP_DIR, POOL_META, MIN_POOL_SIZE, POOL_SIZE_CAP,
            random_synthesis_params, "nonverbal")


def top_up_ack_pool() -> None:
    """Generate acknowledgment clips until the pool reaches ACK_MIN_POOL_SIZE."""
    _top_up(ACK_CLIP_DIR, ACK_POOL_META, ACK_MIN_POOL_SIZE, ACK_POOL_SIZE_CAP,
            random_ack_params, "acknowledgment")


# ---------------------------------------------------------------------------
# Daemon mode
# ---------------------------------------------------------------------------

def daemon_mode(check_interval: int = 300) -> None:
    """Run pool top-ups periodically."""
    log.info("Daemon started (interval=%ds)", check_interval)
    while True:
        try:
            top_up_pool()
            top_up_ack_pool()
        except Exception as e:
            log.error("Error during top-up: %s", e)
        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Non-verbal filler clip factory")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode (periodic top-up)")
    parser.add_argument("--interval", type=int, default=300, help="Daemon check interval in seconds (default: 300)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.daemon:
        try:
            daemon_mode(check_interval=args.interval)
        except KeyboardInterrupt:
            log.info("Daemon interrupted")
    else:
        top_up_pool()
        top_up_ack_pool()


if __name__ == "__main__":
    main()
