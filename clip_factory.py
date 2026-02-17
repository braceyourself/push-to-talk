#!/usr/bin/env python3
"""
Background clip factory daemon for non-verbal filler generation.

Generates, evaluates, and manages a rotating pool of non-verbal audio clips
(hums, breaths) via Piper TTS. Can run as a one-shot pool top-up or as a
background daemon that periodically ensures the pool stays healthy.

Usage:
    python clip_factory.py              # One-shot: top up pool and exit
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

# Non-verbal prompts that Piper can synthesize
PROMPTS = ["Hmm", "Mmm", "Mhm", "Hm", "Mmhmm", "Hmmm", "Ahh", "Uhh"]

# Maximum consecutive generation failures before giving up
MAX_CONSECUTIVE_FAILURES = 20

log = logging.getLogger("clip_factory")


# ---------------------------------------------------------------------------
# Synthesis parameter generation
# ---------------------------------------------------------------------------

def random_synthesis_params() -> dict:
    """Return randomized Piper TTS parameters for clip diversity."""
    return {
        "prompt": random.choice(PROMPTS),
        "length_scale": round(random.uniform(0.7, 1.8), 2),
        "noise_w_scale": round(random.uniform(0.3, 1.5), 2),
        "noise_scale": round(random.uniform(0.4, 1.0), 2),
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

def evaluate_clip(pcm_data: bytes) -> dict:
    """Evaluate audio quality of raw PCM data. Returns scores dict with pass/fail."""
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
    prefix = prompt.lower()
    n = 1
    while True:
        name = f"{prefix}_{n:03d}.wav"
        if name not in existing_filenames:
            return name
        n += 1


def save_clip(pcm_data: bytes, filename: str) -> Path:
    """Write raw PCM data to a WAV file. Returns the full path."""
    filepath = CLIP_DIR / filename
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return filepath


# ---------------------------------------------------------------------------
# Pool metadata persistence
# ---------------------------------------------------------------------------

def load_pool_meta() -> list[dict]:
    """Read pool.json. Returns empty list if file missing or corrupt."""
    if not POOL_META.exists():
        return []
    try:
        return json.loads(POOL_META.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to read pool.json: %s", e)
        return []


def save_pool_meta(meta: list[dict]) -> None:
    """Write pool metadata to pool.json."""
    POOL_META.parent.mkdir(parents=True, exist_ok=True)
    POOL_META.write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Pool rotation
# ---------------------------------------------------------------------------

def rotate_pool(meta: list[dict]) -> list[dict]:
    """Remove oldest clips when pool exceeds POOL_SIZE_CAP."""
    if len(meta) <= POOL_SIZE_CAP:
        return meta

    # Sort by creation time ascending (oldest first)
    meta.sort(key=lambda c: c.get("created_at", 0))

    while len(meta) > POOL_SIZE_CAP:
        oldest = meta.pop(0)
        clip_path = CLIP_DIR / oldest["filename"]
        if clip_path.exists():
            clip_path.unlink()
            log.info("Rotated out: %s", oldest["filename"])

    return meta


# ---------------------------------------------------------------------------
# Main pool management
# ---------------------------------------------------------------------------

def top_up_pool() -> None:
    """Generate clips until the pool reaches MIN_POOL_SIZE."""
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_pool_meta()

    # Reconcile: remove metadata entries for clips that no longer exist on disk
    existing_wavs = {p.name for p in CLIP_DIR.glob("*.wav")}
    before = len(meta)
    meta = [m for m in meta if m["filename"] in existing_wavs]
    if len(meta) < before:
        log.info("Pruned %d orphaned metadata entries", before - len(meta))

    existing_filenames = {m["filename"] for m in meta}
    consecutive_failures = 0

    while len(meta) < MIN_POOL_SIZE:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.error(
                "Too many consecutive failures (%d). Pool at %d clips.",
                consecutive_failures, len(meta),
            )
            break

        params = random_synthesis_params()
        pcm = generate_clip(
            params["prompt"],
            params["length_scale"],
            params["noise_w_scale"],
            params["noise_scale"],
        )

        if pcm is None:
            consecutive_failures += 1
            log.info("Generation failed, retrying (%d/%d)", consecutive_failures, MAX_CONSECUTIVE_FAILURES)
            continue

        scores = evaluate_clip(pcm)
        if not scores["pass"]:
            consecutive_failures += 1
            log.info(
                "Rejected: %s (dur=%.2fs rms=%.0f clip=%.4f sil=%.3f)",
                params["prompt"], scores["duration"], scores["rms"],
                scores["clipping_ratio"], scores["silence_ratio"],
            )
            continue

        consecutive_failures = 0
        filename = _next_filename(params["prompt"], existing_filenames)
        save_clip(pcm, filename)
        existing_filenames.add(filename)

        entry = {
            "filename": filename,
            "created_at": time.time(),
            "params": params,
            "scores": scores,
        }
        meta.append(entry)
        log.info(
            "Saved: %s (dur=%.2fs rms=%.0f)",
            filename, scores["duration"], scores["rms"],
        )

    # Rotate if pre-existing clips pushed over cap
    meta = rotate_pool(meta)
    save_pool_meta(meta)

    print(f"Clip factory: pool at {len(meta)} clips (cap {POOL_SIZE_CAP})", flush=True)


# ---------------------------------------------------------------------------
# Daemon mode
# ---------------------------------------------------------------------------

def daemon_mode(check_interval: int = 300) -> None:
    """Run top_up_pool() periodically."""
    log.info("Daemon started (interval=%ds)", check_interval)
    while True:
        try:
            top_up_pool()
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


if __name__ == "__main__":
    main()
