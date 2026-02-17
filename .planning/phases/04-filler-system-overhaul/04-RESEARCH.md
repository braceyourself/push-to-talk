# Phase 4: Filler System Overhaul - Research

**Researched:** 2026-02-17
**Domain:** Audio clip management, local TTS synthesis, audio quality evaluation
**Confidence:** HIGH

## Summary

The filler system overhaul replaces the current two-tier filler approach (Ollama smart filler + canned clips fallback) with a single-tier system: non-verbal audio clips only. The current codebase has `_filler_manager()` which first tries generating contextual text via Ollama (localhost:11434), then falls back to pre-recorded WAV clips in `audio/fillers/`. The overhaul removes the Ollama path entirely, replaces verbal clips ("Got it", "Right", "On it") with non-verbal sounds ("Hmm", "Mmm", "Mhm"), and adds a background clip factory subprocess that generates new clips via Piper TTS, evaluates their quality, and maintains a capped rotating pool.

Piper TTS is already installed and working at `~/.local/share/push-to-talk/venv/bin/piper` with the `en_US-lessac-medium` voice model (22050Hz output). Testing confirms Piper can synthesize non-verbal vocalizations: "Hmm" (0.60s), "Mmm" (0.85s), "Mhm" (0.96s), "Hm" (0.79s), "Mmhmm" (0.94s), "Ahh" (0.36s), "Uhh" (0.49s). Duration and character can be varied using `--length-scale` (0.7-2.0) and `--noise-w-scale` (prosody variation) parameters.

For naturalness evaluation, the practical approach is a lightweight heuristic checker using numpy (already installed) rather than a deep learning model. NISQA-TTS via torchmetrics requires librosa and produces MOS scores, but for sub-second non-verbal clips the simpler checks (duration range, RMS energy, silence ratio, clipping) are sufficient and avoid adding dependencies.

**Primary recommendation:** Remove `_generate_smart_filler()` and the Ollama dependency. Replace the existing verbal clip categories with non-verbal clips. Add a `clip_factory.py` daemon (following the `learner.py` subprocess pattern) that runs periodically to generate, evaluate, and rotate clips in the pool.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Piper TTS | 1.0.0 (installed) | Generate non-verbal audio clips | Already integrated for filler TTS and main TTS fallback. Produces 22050Hz PCM. |
| Python stdlib `wave` | 3.12 | Read/write WAV files | Zero dependency. Already used throughout codebase for clip I/O. |
| Python stdlib `struct` | 3.12 | PCM byte manipulation | Already used in `_resample_22050_to_24000()`. |
| numpy | 2.3.5 (installed) | Audio analysis (RMS, clipping, silence detection) | Already installed for Whisper. Sufficient for basic audio quality checks. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python stdlib `subprocess` | 3.12 | Run Piper CLI from clip factory | Clip factory is a separate process, uses blocking subprocess calls (not asyncio). |
| Python stdlib `json` | 3.12 | Clip metadata persistence | Track clip generation params, quality scores, creation date. |
| Python stdlib `random` | 3.12 | Parameter variation for diverse clips | Vary noise_w, length_scale, noise_scale for each generation. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numpy RMS/clipping checks | NISQA-TTS via torchmetrics | NISQA gives MOS naturalness score but requires librosa dependency, and may not work well on sub-second non-verbal clips. Too heavy for this use case. |
| numpy RMS/clipping checks | librosa | More features (spectral analysis, ZCR) but adds a dependency. numpy is sufficient for duration, RMS, silence, clipping checks. |
| Piper TTS | Bark / Coqui TTS | Bark has more expressive non-verbal capabilities but is much larger (~5GB model), slower, and not installed. Piper is already working and produces acceptable non-verbal sounds. |
| Separate subprocess | asyncio task in live_session | Clip factory should not run in the hot pipeline event loop. It is slow (Piper synthesis + evaluation) and runs infrequently. Subprocess model matches learner.py pattern. |

**Installation:**
```bash
# No new dependencies required. All tools already installed.
```

## Architecture Patterns

### Recommended Project Structure

```
push-to-talk/
├── live_session.py        # Modified: remove smart filler, simplify _filler_manager
├── clip_factory.py        # NEW: background daemon for clip generation/rotation
├── generate_fillers.py    # REMOVE or repurpose (currently uses OpenAI TTS)
├── audio/
│   └── fillers/
│       ├── nonverbal/     # NEW category replacing acknowledge/thinking/tool_use
│       │   ├── hmm_001.wav
│       │   ├── mmm_002.wav
│       │   └── ...
│       └── pool.json      # NEW: metadata tracking clip quality, age, params
```

### Pattern 1: Clip Factory as Background Daemon (like learner.py)

**What:** A standalone Python script spawned as a subprocess by the live session (or run independently via cron/systemd timer). It manages the clip pool lifecycle: generate -> evaluate -> add/reject -> rotate old clips out.

**When to use:** Run once at session start to top up the pool, then optionally on a timer during long sessions.

**Example:**
```python
#!/usr/bin/env python3
"""Background clip factory daemon for non-verbal filler generation."""

import subprocess
import wave
import json
import random
import time
import numpy as np
from pathlib import Path

PIPER_CMD = str(Path.home() / ".local/share/push-to-talk/venv/bin/piper")
PIPER_MODEL = str(Path.home() / ".local/share/push-to-talk/piper-voices/en_US-lessac-medium.onnx")
CLIP_DIR = Path(__file__).parent / "audio" / "fillers" / "nonverbal"
POOL_META = Path(__file__).parent / "audio" / "fillers" / "pool.json"

POOL_SIZE_CAP = 20  # Maximum clips in the pool
MIN_POOL_SIZE = 10  # Generate until we reach this
SAMPLE_RATE = 22050

# Non-verbal prompts and their parameter ranges
PROMPTS = [
    "Hmm", "Mmm", "Mhm", "Hm", "Mmhmm", "Hmmm",
    "Ahh", "Uhh",
]

def generate_clip(prompt: str, length_scale: float, noise_w: float) -> bytes | None:
    """Generate a single clip via Piper, return raw PCM bytes."""
    try:
        result = subprocess.run(
            [PIPER_CMD, '--model', PIPER_MODEL, '--output-raw',
             '--length-scale', str(length_scale),
             '--noise-w-scale', str(noise_w)],
            input=prompt.encode(),
            capture_output=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None

def evaluate_clip(pcm_data: bytes) -> dict:
    """Evaluate audio quality. Returns scores dict."""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
    duration = len(samples) / SAMPLE_RATE
    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))
    clipping = np.sum(np.abs(samples) >= 32000) / len(samples)
    silence = np.sum(np.abs(samples) < 500) / len(samples)

    return {
        "duration": duration,
        "rms": float(rms),
        "peak": int(peak),
        "clipping_ratio": float(clipping),
        "silence_ratio": float(silence),
        "pass": (
            0.2 < duration < 2.0
            and rms > 300
            and clipping < 0.01
            and silence < 0.7
        ),
    }
```

### Pattern 2: Simplified Filler Manager (no Ollama)

**What:** Replace the current two-stage `_filler_manager` with a single-stage clip-only approach. Remove `_generate_smart_filler()`, `_tts_to_pcm()` for fillers, and the `_spoken_filler` dedup logic.

**When to use:** Every time a filler is needed (user finishes speaking, LLM is processing).

**Example:**
```python
async def _filler_manager(self, user_text: str, cancel_event: asyncio.Event):
    """Play a non-verbal filler clip while waiting for LLM response."""
    # Stage 1: Wait 300ms gate — skip filler if LLM responds fast
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=0.3)
        return
    except asyncio.TimeoutError:
        pass

    if cancel_event.is_set():
        return

    # Play a non-verbal clip
    clip = self._pick_filler("nonverbal")
    if clip:
        await self._play_filler_audio(clip, cancel_event)

    # Stage 2: If still waiting after 4s, play another clip
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=4.0)
        return
    except asyncio.TimeoutError:
        pass

    if not cancel_event.is_set():
        clip = self._pick_filler("nonverbal")
        if clip:
            await self._play_filler_audio(clip, cancel_event)
```

### Pattern 3: Pool Rotation with Metadata

**What:** Track clip metadata (creation time, quality scores, play count) in a JSON sidecar file. When the pool exceeds the cap, remove the oldest or lowest-quality clips.

**When to use:** After successful clip generation and evaluation.

**Example:**
```python
def rotate_pool(pool_meta_path: Path, clip_dir: Path, cap: int):
    """Remove oldest clips when pool exceeds cap."""
    meta = json.loads(pool_meta_path.read_text()) if pool_meta_path.exists() else []

    # Sort by creation time, keep newest
    meta.sort(key=lambda c: c.get("created_at", 0))

    while len(meta) > cap:
        oldest = meta.pop(0)
        clip_path = clip_dir / oldest["filename"]
        if clip_path.exists():
            clip_path.unlink()

    pool_meta_path.write_text(json.dumps(meta, indent=2))
```

### Anti-Patterns to Avoid

- **Running Piper synthesis in the asyncio event loop:** Piper can take 100-500ms per clip. Never block the pipeline. The clip factory runs as a separate process.
- **Using NISQA/deep learning for sub-second clip evaluation:** Overkill. Simple RMS/duration/clipping checks are sufficient and have zero extra dependencies.
- **Keeping verbal fillers alongside non-verbal:** The whole point is removing verbal fillers that conflict with LLM responses. Do not mix categories.
- **Generating clips on-demand during live session:** Too slow and unpredictable. Pre-generate a pool and pick from it at runtime.
- **Storing clip quality in the WAV filename:** Use a JSON metadata file. Filenames should be simple identifiers.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCM resampling 22050->24000 | Custom resampler | Existing `_resample_22050_to_24000()` | Already works, tested in production. Linear interpolation is sufficient for non-verbal clips. |
| WAV file I/O | Custom binary parser | Python stdlib `wave` module | Standard, already used everywhere in the codebase. |
| Random clip selection without repeats | Custom shuffle logic | Existing `_pick_filler()` with last-index tracking | Already handles no-repeat-adjacent correctly. |
| Subprocess spawning pattern | Custom process manager | Follow `learner.py` pattern (subprocess.Popen with start_new_session=True) | Proven pattern in this codebase. Clean shutdown, no zombie processes. |

**Key insight:** The codebase already has all the building blocks: Piper TTS integration, WAV I/O, PCM resampling, subprocess daemon pattern (learner.py), filler clip loading and playback. This phase is primarily about removing code (Ollama smart filler) and reorganizing existing patterns into a cleaner architecture.

## Common Pitfalls

### Pitfall 1: Piper Produces Silence for Pure Punctuation

**What goes wrong:** Passing "..." or other non-word inputs to Piper produces 0 bytes of output.
**Why it happens:** Piper's eSpeak-NG phonemizer maps text to phonemes. Pure punctuation has no phonemes.
**How to avoid:** Only use word-like prompts: "Hmm", "Mmm", "Mhm", "Ahh", "Uhh", etc. Never pass punctuation-only strings.
**Warning signs:** 0-byte Piper output, empty WAV files in pool.

### Pitfall 2: Clip Duration Variance from Piper Parameters

**What goes wrong:** Varying `length_scale` too aggressively produces clips that are either too short (<0.2s, inaudible) or too long (>2s, blocks the pipeline).
**Why it happens:** `length_scale` is a direct multiplier on phoneme duration. Extreme values compound with already-short prompts.
**How to avoid:** Constrain `length_scale` to 0.7-1.8 range. Evaluate duration post-generation and reject outliers.
**Warning signs:** Clips that feel rushed or unnaturally drawn out.

### Pitfall 3: Filler Dedup Logic Must Be Updated

**What goes wrong:** The current `_spoken_filler` dedup compares filler text against the first LLM sentence. With non-verbal clips, there is no text to compare.
**Why it happens:** The dedup was designed for smart fillers that said words like "Got it" which Claude might also say.
**How to avoid:** Remove the `_spoken_filler` dedup entirely. Non-verbal clips (hums, breaths) cannot duplicate LLM text output.
**Warning signs:** Dead code left checking `_spoken_filler` for non-verbal clips that have no text.

### Pitfall 4: Category Structure Changes

**What goes wrong:** The current code loads clips from `audio/fillers/{acknowledge,thinking,tool_use}/`. Changing to a single `nonverbal/` category without updating all references breaks loading.
**Why it happens:** Category names are hardcoded in `_load_filler_clips()` and `_classify_filler_category()`.
**How to avoid:** Either keep backward-compatible categories (rename contents but keep structure) OR update all references simultaneously. The simpler approach: use a single `nonverbal` category since all clips are now the same type.
**Warning signs:** "No filler clips loaded" at startup, `_filler_clips` dict is empty.

### Pitfall 5: Clip Factory Running During Audio Playback

**What goes wrong:** If the clip factory runs Piper TTS while the live session is playing audio, both compete for audio resources.
**Why it happens:** Piper with `--output-raw` writes to stdout, not speakers. But if someone accidentally uses `--output-file` with playback, or the system audio setup has conflicts.
**How to avoid:** Clip factory uses `--output-raw` (stdout pipe) and writes to WAV files manually. Never plays audio directly. Factory should run at session start or between sessions, not during active playback.
**Warning signs:** Audio glitches during clip generation.

### Pitfall 6: aiohttp Dependency Left Behind

**What goes wrong:** After removing the Ollama smart filler, the `import aiohttp` inside `_generate_smart_filler()` is no longer needed. But if other code depends on aiohttp it should stay.
**Why it happens:** The aiohttp import is inside the function body, so it only fails when called. But the dependency in the environment may or may not be needed elsewhere.
**How to avoid:** Check if aiohttp is used anywhere else in the codebase before removing. (Currently: only in `_generate_smart_filler()`.)
**Warning signs:** ImportError at runtime if removed from environment but used elsewhere.

## Code Examples

Verified patterns from the existing codebase:

### Piper TTS CLI invocation (existing, from live_session.py)
```python
# Source: live_session.py line 611-627
async def _tts_to_pcm(self, text: str) -> bytes | None:
    """Convert text to PCM audio via Piper. Returns resampled 24kHz bytes."""
    process = await asyncio.create_subprocess_exec(
        PIPER_CMD, '--model', PIPER_MODEL, '--output-raw',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )
    stdout, _ = await asyncio.wait_for(
        process.communicate(input=text.encode()),
        timeout=3.0
    )
    if stdout:
        return self._resample_22050_to_24000(stdout)
```

### Learner daemon spawn pattern (existing, from live_session.py)
```python
# Source: live_session.py line 222-236
def _spawn_learner(self):
    """Spawn the background learner daemon."""
    learner_script = Path(__file__).parent / "learner.py"
    cmd = [sys.executable, str(learner_script), str(self._session_log_path)]
    self._learner_process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
```

### Clip loading pattern (existing, from live_session.py)
```python
# Source: live_session.py line 476-509
def _load_filler_clips(self):
    """Load pre-generated filler WAV files as raw PCM bytes."""
    filler_dir = Path(__file__).parent / "audio" / "fillers"
    for category in ("acknowledge", "thinking", "tool_use"):
        cat_dir = filler_dir / category
        clips = []
        for wav_path in sorted(cat_dir.glob("*.wav")):
            with wave.open(str(wav_path), 'rb') as wf:
                pcm = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
            if rate != SAMPLE_RATE:
                pcm = self._resample_22050_to_24000(pcm)
            clips.append(pcm)
        if clips:
            self._filler_clips[category] = clips
```

### numpy-based audio quality evaluation (new)
```python
# For clip_factory.py — evaluate generated clips before adding to pool
import numpy as np

def evaluate_clip(pcm_data: bytes, sample_rate: int = 22050) -> dict:
    """Basic quality checks for a generated non-verbal clip."""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
    n = len(samples)
    duration = n / sample_rate

    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))
    clipping_ratio = np.sum(np.abs(samples) >= 32000) / n
    silence_ratio = np.sum(np.abs(samples) < 500) / n

    passes = (
        0.2 < duration < 2.0        # Not too short or long
        and rms > 300                 # Audible signal present
        and clipping_ratio < 0.01     # No clipping
        and silence_ratio < 0.7       # Not mostly silence
    )

    return {
        "duration": round(duration, 3),
        "rms": round(float(rms), 1),
        "peak": int(peak),
        "clipping_ratio": round(float(clipping_ratio), 4),
        "silence_ratio": round(float(silence_ratio), 3),
        "pass": passes,
    }
```

### Piper parameter variation for diverse clips (new)
```python
# Tested parameter ranges for non-verbal synthesis
import random

def random_synthesis_params() -> dict:
    """Generate random Piper TTS parameters for clip diversity."""
    return {
        "prompt": random.choice([
            "Hmm", "Mmm", "Mhm", "Hm", "Mmhmm", "Hmmm",
            "Ahh", "Uhh",
        ]),
        "length_scale": round(random.uniform(0.7, 1.8), 2),
        "noise_w_scale": round(random.uniform(0.3, 1.5), 2),
        "noise_scale": round(random.uniform(0.4, 1.0), 2),
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ollama smart filler (contextual LLM text) | Remove entirely | This phase | Eliminates conflict where filler says "Got it" and Claude also says "Got it" |
| OpenAI TTS for initial clip generation | Piper TTS for all clip generation | Already partial (Piper used for smart filler TTS) | Zero cloud dependency for fillers |
| Verbal filler clips ("Right", "Okay", "Let me check") | Non-verbal clips ("Hmm", "Mmm", "Mhm") | This phase | No semantic content to conflict with LLM response |
| Static clip pool (generate_fillers.py one-time) | Dynamic clip factory with rotation | This phase | Fresh, varied clips; avoids listener habituation |

**Deprecated/outdated:**
- `generate_fillers.py`: Uses OpenAI TTS API, generates verbal clips. Will be replaced by clip_factory.py.
- `_generate_smart_filler()`: Ollama-based contextual filler. Being removed entirely.
- `_spoken_filler` / filler text dedup: Only relevant for verbal fillers. Remove with smart filler.
- `FILLER_TOOL_KEYWORDS` / `FILLER_THINKING_KEYWORDS`: Category classification for verbal fillers. Not needed with single non-verbal category.
- `_classify_filler_category()`: Same reason -- no categories to classify into.

## Open Questions

Things that couldn't be fully resolved:

1. **Single category vs. multiple non-verbal categories?**
   - What we know: Current code has 3 categories (acknowledge, thinking, tool_use). Non-verbal clips are all similar in nature (hums, breaths).
   - What's unclear: Whether having varied subcategories (short hums for acknowledge, longer hums for thinking) adds value or just complexity.
   - Recommendation: Start with a single `nonverbal` category. If user feedback indicates a need for variation by context, add subcategories later. YAGNI.

2. **When should the clip factory run?**
   - What we know: It should not run during active playback. The learner daemon runs for the full session.
   - What's unclear: Whether it should run once at startup (top up pool), continuously in the background, or only between sessions.
   - Recommendation: Run at session start to ensure MIN_POOL_SIZE clips exist. Optionally run between sessions via a flag or manual invocation. Do not run continuously during a session -- the pool is small enough that startup generation is sufficient.

3. **Should old verbal clip files be deleted or kept?**
   - What we know: Current clips are in `audio/fillers/{acknowledge,thinking,tool_use}/`. They are committed to git (via `.gitignore` exception).
   - What's unclear: Whether to delete them immediately or keep them as a fallback.
   - Recommendation: Delete the old verbal clip directories after confirming the new non-verbal pool works. Clean break.

4. **Pool metadata persistence format**
   - What we know: Need to track clip filename, creation date, quality scores, generation params.
   - What's unclear: Whether a simple JSON file or something more structured is needed.
   - Recommendation: Single `pool.json` file in the clip directory. The pool is small (<50 entries) so JSON read/write is fine.

5. **Personality prompt update**
   - What we know: `personality/context.md` line 37 says: "Brief acknowledgments like 'Got it' or 'Sure thing' are spoken automatically while you process." This is no longer true with non-verbal fillers.
   - What's unclear: What the replacement text should be.
   - Recommendation: Update to say something like: "Non-verbal acknowledgments (hums, breaths) are played automatically while you process. Never start your response with a greeting, acknowledgment, or filler phrase."

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `live_session.py` (1801 lines) -- full filler system, pipeline architecture, Piper integration
- Codebase analysis: `generate_fillers.py` -- current OpenAI TTS-based clip generation
- Codebase analysis: `learner.py` -- subprocess daemon pattern to replicate
- Codebase analysis: `pipeline_frames.py` -- FrameType.FILLER frame type
- Codebase analysis: `indicator.py` -- Settings UI for fillers toggle
- Codebase analysis: `push-to-talk.py` -- Config handling, `live_fillers` setting
- Direct testing: Piper TTS CLI with non-verbal prompts and parameter variations -- verified output durations and quality

### Secondary (MEDIUM confidence)
- [Piper TTS GitHub](https://github.com/rhasspy/piper) -- noise_scale, noise_w_scale, length_scale parameter documentation
- [TorchMetrics NISQA docs](https://lightning.ai/docs/torchmetrics/stable/audio/non_intrusive_speech_quality_assessment.html) -- NISQA API for potential future use
- [Piper DeepWiki](https://deepwiki.com/rhasspy/piper/2-core-tts-engine) -- VITS model architecture, synthesis parameters

### Tertiary (LOW confidence)
- Web search: "piper TTS generate non-verbal sounds" -- confirmed Piper does not natively support non-verbal sounds, but word-like prompts work
- Web search: "python audio quality assessment" -- identified NISQA, VERSA, AQP toolkits (decided against for this use case)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools already installed and tested in production
- Architecture: HIGH - Following existing codebase patterns (learner.py subprocess daemon, clip loading)
- Pitfalls: HIGH - All identified from direct codebase analysis and hands-on Piper testing

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (stable -- no fast-moving dependencies)
