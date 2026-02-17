# Phase 5: Barge-in - Research

**Researched:** 2026-02-17
**Domain:** Voice activity detection during AI playback, audio pipeline interruption, conversation context management
**Confidence:** HIGH

## Summary

Barge-in enables the user to interrupt the AI mid-speech by speaking. This requires three coordinated changes to the existing pipeline: (1) keep the microphone live during playback and run VAD on incoming audio to detect user speech, (2) cancel/fade the AI's playback when speech is detected, and (3) annotate the conversation context so the AI knows it was interrupted.

The existing codebase already has the scaffolding: a `_vad_monitor_stage()` that currently does nothing (because the mic is muted via `pactl` during playback), a `request_interrupt()` / `_check_interrupt()` mechanism that increments `generation_id` to discard stale frames, and a `_playback_stage()` that writes chunks sequentially. The core work is: (a) replacing mic muting with an STT gate during playback, (b) wiring VAD into the audio capture stream during playback, (c) implementing fade-out + filler clip on interrupt, and (d) sending an interruption annotation to the CLI before the user's next message.

**Primary recommendation:** Use the `silero-vad` pip package (v6.2.0, already installed) with its `OnnxWrapper` model at 16kHz. Resample the 24kHz mic input to 16kHz for VAD. Accumulate consecutive high-probability frames to reach the ~0.5s sustained speech threshold before triggering. Apply a linear fade to PCM chunks in the playback stage. Send the interrupted text annotation as a user message to the CLI.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `silero-vad` | 6.2.0 | Voice activity detection during playback | Industry-standard lightweight VAD, ONNX-based, <1ms per chunk, already installed |
| `onnxruntime` | 1.24.1 | ONNX inference backend for Silero VAD | Installed as dependency of silero-vad, CPU-only, fast |
| `numpy` | 1.26.4 | PCM audio manipulation (fade, resampling, VAD prep) | Already used throughout codebase for audio processing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `PyAudio` | 0.2.14 | Audio playback (existing) | Already the playback backend in `_playback_stage()` |
| `torch` | 2.10.0 | Tensor operations for VAD model input | Already installed; silero-vad can use torch tensors as input |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Silero VAD ONNX | Silero VAD JIT (PyTorch) | Both work; ONNX is faster (<1ms vs ~few ms), already loads via `load_silero_vad(onnx=True)` |
| Raw `OnnxWrapper` calls | `VADIterator` high-level API | VADIterator manages speech start/end detection internally with `min_silence_duration_ms`; raw calls give more control for our custom ~0.5s sustained speech logic. **Use raw model calls** since we need custom accumulation logic |
| Manual ONNX model file + onnxruntime | `silero-vad` pip package | Package bundles model + utilities, handles loading. Much simpler than downloading ONNX files manually. **Use the pip package** |

**Installation:**
```bash
pip install silero-vad  # Already installed: v6.2.0
# Brings in: onnxruntime, torchaudio (dependencies)
```

**Note:** The existing code in `_load_vad_model()` uses manual ONNX file loading from `models/silero_vad.onnx` which doesn't exist. Replace with `silero_vad.load_silero_vad(onnx=True)` which bundles the model.

## Architecture Patterns

### Current Pipeline (Before Barge-in)

```
Audio Capture (pw-record)
    ↓ audio_in_q [AUDIO_RAW frames]
STT Stage (Whisper)         ← GATES on: playing_audio, muted
    ↓ stt_out_q [TRANSCRIPT frames]
LLM Stage (Claude CLI)
    ↓ llm_out_q [TEXT_DELTA frames]
TTS Stage (Piper)
    ↓ audio_out_q [TTS_AUDIO frames]
Playback Stage (PyAudio)    ← Sets playing_audio=True, calls _mute_mic()

[VAD Monitor Stage]         ← Currently a no-op (mic is muted)
[Interrupt Loop]            ← Polls _interrupt_requested every 50ms
```

**Problem:** During playback, `_mute_mic()` physically mutes the mic via `pactl`, so the audio capture stage sends silence to `audio_in_q`, and the STT stage ignores frames when `playing_audio` is True. VAD cannot hear anything.

### Target Pipeline (With Barge-in)

```
Audio Capture (pw-record)   ← Mic stays LIVE during playback (no pactl mute)
    ↓ audio_in_q [AUDIO_RAW frames]
    ├─→ STT Stage (Whisper) ← GATES on: stt_gated flag (replaces playing_audio check)
    │       ↓ stt_out_q
    └─→ VAD Monitor          ← Reads from audio_in_q DURING playback
            ↓ triggers barge-in → sets _barge_in_triggered
                → fade out playback
                → play trailing filler
                → ungate STT
                → set cooldown timer

LLM Stage (Claude CLI)      ← Receives "[interrupted after: ...]" annotation
TTS Stage (Piper)            ← Stale frames discarded by generation_id
Playback Stage (PyAudio)     ← Applies fade when barge-in triggers, no mic mute
```

### Pattern 1: STT Gating Instead of Mic Mute

**What:** Replace physical mic muting (`pactl set-source-mute`) with a software gate flag (`self._stt_gated`) that the STT stage checks before processing audio. The mic stays physically live so VAD can hear speech during playback.

**When to use:** Always during playback when barge-in is enabled.

**Key changes:**
- Remove `_mute_mic()` / `_unmute_mic()` calls during playback (keep them for session start/stop cleanup)
- Add `self._stt_gated = False` flag
- STT stage checks `self._stt_gated` instead of `self.playing_audio`
- When playback starts: set `_stt_gated = True` (not `_mute_mic()`)
- When playback ends or barge-in triggers: set `_stt_gated = False`

**Critical detail:** Audio capture still sends frames to `audio_in_q` during playback. Both STT and VAD consume from this queue. Since `asyncio.Queue` is single-consumer, either: (a) have both read from the same queue with a tee/broadcast pattern, or (b) have VAD monitor read directly from the capture stage via a separate mechanism.

**Recommended approach:** Modify `_audio_capture_stage()` to also push frames to a `_vad_in_q` when `playing_audio` is True. Or simpler: have the VAD monitor run VAD inline on each chunk as it passes through `audio_in_q` in the STT stage (the STT stage already reads every chunk, it just discards them when gated).

**Simplest approach:** Run VAD in the STT stage itself. When `_stt_gated` is True, the STT stage still reads frames but instead of buffering for transcription, it runs VAD on each chunk. If VAD detects sustained speech, it triggers barge-in. This avoids queue duplication entirely.

### Pattern 2: Sustained Speech Detection (~0.5s Threshold)

**What:** Accumulate consecutive VAD-positive frames until enough time has passed to confirm intentional speech (not a cough or noise burst).

**Configuration:**
- Silero VAD processes 512 samples at 16kHz = 32ms per chunk
- 0.5s ÷ 0.032s = ~16 consecutive positive chunks needed
- Use VAD threshold of 0.5 (Silero default, well-tested)
- Reset counter on any chunk below threshold

**Implementation:**
```python
# In STT stage, when _stt_gated:
speech_prob = self._run_vad(chunk_resampled_16k)
if speech_prob > VAD_THRESHOLD:
    self._barge_speech_frames += 1
    if self._barge_speech_frames >= BARGE_IN_FRAMES_REQUIRED:  # ~16
        self._trigger_barge_in()
else:
    self._barge_speech_frames = 0
```

### Pattern 3: Playback Fade-Out (100-200ms)

**What:** When barge-in triggers, apply a linear volume ramp from 1.0 to 0.0 over 100-200ms on the currently-playing audio chunks, then stop.

**Implementation:**
```python
# In _playback_stage, when barge-in is signaled:
fade_samples = int(SAMPLE_RATE * 0.15)  # 150ms = 3600 samples at 24kHz
# Apply linear fade to current + next few chunks
fade_envelope = np.linspace(1.0, 0.0, fade_samples)
samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
samples[:len(fade_envelope)] *= fade_envelope[:len(samples)]
faded = samples.astype(np.int16).tobytes()
```

**Key detail:** The fade happens in the playback stage because that's where we have the PyAudio stream. Signal barge-in via an asyncio.Event or flag that the playback stage checks each iteration.

### Pattern 4: Trailing Filler Clip on Interrupt

**What:** After the fade, play a short non-verbal filler clip (breath, "mm") to simulate a natural interruption. This reuses the existing `_pick_filler("nonverbal")` and `_play_filler_audio()` infrastructure.

**When to use:** Immediately after the fade-out completes, before the user's speech is processed.

**Implementation:** After fade, pick a filler clip and push its frames to the playback queue with the NEW generation_id (so it doesn't get discarded). The filler clip should be short and the `cancel_event` should be set when the user's STT result arrives.

**Alternative:** Skip the filler entirely if the fade is clean enough. The filler adds naturalness but also adds complexity. Claude's discretion per CONTEXT.md.

### Pattern 5: Interrupted Text Annotation

**What:** After barge-in, send the AI context about what happened. The Claude CLI maintains conversation history internally via stream-json, so we inject this as a user message.

**Format:**
```
[System: You were interrupted. The user heard up to: "...last spoken sentence...".
The rest of your response ("...unspoken text...") was not spoken aloud.
The user is about to speak — listen to what they say.]
```

**How to track spoken vs unspoken text:** The LLM stage already accumulates `full_response` text. The playback stage can track how many bytes were played (`_bytes_played`). Since we know the TTS rate and text-to-audio mapping is roughly proportional, we can estimate which sentence was last spoken. More precisely: track which `TEXT_DELTA` frames were fully played vs which were in the queue when interrupted.

**Simpler approach:** Track the last sentence that was sent to TTS and confirmed played (i.e., the sentence whose TTS_AUDIO frames have all been written to PyAudio). Everything after that is "interrupted."

### Pattern 6: Cooldown Period (1-2s)

**What:** After a barge-in triggers, ignore further barge-in attempts for 1-2 seconds to prevent rapid-fire interruption cascading.

**Implementation:** Set `self._barge_in_cooldown_until = time.time() + 1.5` when barge-in fires. Check this timestamp before allowing another trigger.

### Anti-Patterns to Avoid

- **Muting the mic during playback:** This is the current approach and prevents barge-in entirely. Must be replaced with STT gating.
- **Hard-cutting audio (no fade):** Sounds jarring and unnatural. Always fade.
- **Running VAD on 24kHz audio:** Silero VAD only supports 8kHz and 16kHz. Must resample before inference.
- **Using VADIterator for barge-in detection:** VADIterator's built-in speech start/end detection has its own timing parameters that don't align with our ~0.5s sustained speech requirement. Use raw model probability calls with custom accumulation.
- **Cancelling the LLM/TTS pipeline on barge-in:** Per CONTEXT.md decision, cancel playback only. Let the LLM finish generating text (just don't speak it). The text goes into `full_response` for context annotation.
- **Separate VAD queue/thread:** Overcomplicates. Run VAD inline in the STT stage where audio frames already flow.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Voice activity detection | Custom energy/RMS threshold | `silero-vad` v6.2.0 (ONNX) | Neural network vastly outperforms energy thresholds for real speech vs noise; handles diverse audio conditions |
| 24kHz→16kHz resampling for VAD | Custom interpolation | numpy decimation (take every 1.5th sample) | Existing pattern in `_run_vad()` works fine for VAD purposes (doesn't need high-quality resampling) |
| PCM volume fade | Custom byte manipulation | numpy: `np.frombuffer` → multiply by envelope → `tobytes()` | Correct int16 overflow handling, vectorized, fast |
| Conversation context injection | Custom protocol handling | `_send_to_cli()` with annotation text | Claude CLI stream-json protocol accepts user messages that become part of conversation history |

**Key insight:** The existing codebase already has 80% of the infrastructure. The VAD model loads, the interrupt mechanism works, the generation_id system discards stale frames, the filler system plays clips. The main new work is wiring these together and replacing mic muting with STT gating.

## Common Pitfalls

### Pitfall 1: Headphone Bleed Triggering VAD

**What goes wrong:** Even with headphones, some audio leaks from earpieces, and the mic picks it up. The VAD detects the AI's own speech as "user speech" and triggers false barge-ins.

**Why it happens:** Cheap headphones have poor isolation. Open-back headphones leak significantly. Microphone gain may be high.

**How to avoid:** The 0.5s sustained speech threshold is the primary defense — brief bleed won't sustain for 0.5s at high VAD confidence. Additionally, during playback we know exactly what audio is being played, so a simple energy comparison could detect if the mic input correlates with the playback output (echo-like). However, per CONTEXT.md, headphones are assumed and echo cancellation is deferred. The 0.5s threshold should suffice.

**Warning signs:** Barge-in triggering repeatedly without the user speaking. Test with headphones at various volume levels.

### Pitfall 2: VAD State Corruption Across Segments

**What goes wrong:** Silero VAD is stateful — the h and c tensors accumulate context across chunks. If not reset between conversation turns, stale state from previous audio affects current detection accuracy.

**Why it happens:** The model maintains RNN hidden state that carries forward. After barge-in or turn transitions, this state reflects the previous audio context, not the current one.

**How to avoid:** Call `model.reset_states()` at: (a) start of each playback period (when STT becomes gated), (b) after barge-in triggers and is handled, (c) on any state transition where audio context changes meaningfully.

**Warning signs:** VAD giving inconsistent results, detecting speech in silence after a loud previous segment.

### Pitfall 3: Queue Starvation / Blocking During Barge-in

**What goes wrong:** When barge-in triggers, multiple things happen simultaneously: playback fade, queue draining, filler playback, STT ungating. If these aren't coordinated, the pipeline can deadlock or drop frames.

**Why it happens:** asyncio queues with `maxsize` can block on `put()` if the consumer is paused. During barge-in, the playback stage is fading/stopping, but TTS may still be producing frames. If `audio_out_q` (maxsize=200) fills up, TTS blocks, which blocks LLM processing.

**How to avoid:** Increment `generation_id` immediately when barge-in triggers. All pipeline stages check generation_id and discard stale frames. The existing pattern in `_check_interrupt()` already does this. Drain queues after incrementing generation_id (existing code does this too).

**Warning signs:** Pipeline hangs after barge-in, audio stops but session doesn't recover.

### Pitfall 4: STT Picking Up Residual Audio After Barge-in

**What goes wrong:** After barge-in ungates STT, the audio buffer may contain fragments of the AI's faded speech mixed with the user's voice. Whisper transcribes this mixed audio, producing garbage.

**Why it happens:** The audio capture stage was recording continuously. The buffer accumulated during playback contains both the user's speech (which triggered barge-in) and leaked/faded AI audio.

**How to avoid:** Clear the STT audio buffer when barge-in triggers. Start fresh from the point of detection. The 0.5s of speech that triggered barge-in has already passed, but we can either: (a) discard it and let the user continue speaking, or (b) keep a rolling buffer of the last ~0.5s for immediate transcription.

**Recommended:** Clear the STT buffer on barge-in. The user will continue speaking after the interruption, and the silence detection will capture their complete thought.

### Pitfall 5: Barge-in During Tool Use

**What goes wrong:** The AI is using tools (status: "tool_use"), and audio from a filler clip is playing. User speaks, triggering barge-in. But there's no active LLM response to interrupt — the tool is still running.

**Why it happens:** Filler clips play during tool use. VAD doesn't distinguish between TTS response playback and filler playback.

**How to avoid:** Only enable barge-in detection when `self.playing_audio` is True AND the playback is TTS_AUDIO (not FILLER). Or more conservatively: disable barge-in during tool_use status entirely — the user can still use the keyboard interrupt key.

**Recommended:** Disable barge-in VAD during filler-only playback. Only activate when TTS_AUDIO frames are being played.

### Pitfall 6: Race Between Fade Completion and User Speech Processing

**What goes wrong:** The fade takes 150ms. During that 150ms, the user is speaking. If STT is ungated immediately on barge-in trigger, it starts buffering the user's speech while the fade is still audible. The user hears both their voice and the fading AI simultaneously, which is natural (like interrupting a person), but if STT processes too early, it might get confused.

**Why it happens:** Barge-in trigger → immediate STT ungate, but fade hasn't completed.

**How to avoid:** This is actually fine for headphone mode — the user's mic won't pick up the speaker output significantly. STT should start immediately per CONTEXT.md ("Immediately start listening after barge-in triggers"). The fade is cosmetic for the user's ears, not for the mic.

## Code Examples

### Example 1: Loading Silero VAD via pip package (replaces manual ONNX)

```python
# Source: verified locally via pip install silero-vad 6.2.0
from silero_vad import load_silero_vad

def _load_vad_model(self):
    """Load Silero VAD model via pip package."""
    try:
        self._vad_model = load_silero_vad(onnx=True)
        self._barge_speech_frames = 0
        self._barge_in_cooldown_until = 0
        print("Live session: Silero VAD loaded (ONNX)", flush=True)
    except Exception as e:
        print(f"Live session: Failed to load VAD: {e}", flush=True)
        self.barge_in_enabled = False
```

### Example 2: Running VAD on resampled audio chunk

```python
# Source: verified locally, Silero VAD requires 16kHz input, 512 samples per chunk
import torch
import numpy as np

def _run_vad_chunk(self, audio_bytes_24k: bytes) -> float:
    """Run VAD on a 24kHz PCM chunk, returns speech probability."""
    if not self._vad_model:
        return 0.0

    # Convert 24kHz PCM to float32 samples
    samples_24k = np.frombuffer(audio_bytes_24k, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample 24kHz → 16kHz (take every 1.5th sample)
    indices = np.arange(0, len(samples_24k), 1.5).astype(int)
    indices = indices[indices < len(samples_24k)]
    samples_16k = samples_24k[indices]

    # VAD expects exactly 512 samples at 16kHz
    if len(samples_16k) < 512:
        samples_16k = np.pad(samples_16k, (0, 512 - len(samples_16k)))

    # Process in 512-sample windows
    max_prob = 0.0
    for i in range(0, len(samples_16k) - 511, 512):
        chunk = torch.from_numpy(samples_16k[i:i+512])
        prob = self._vad_model(chunk, 16000).item()
        max_prob = max(max_prob, prob)

    return max_prob
```

### Example 3: Linear PCM fade-out

```python
# Source: standard numpy audio processing pattern
import numpy as np

def _apply_fade_out(self, pcm_data: bytes, fade_ms: int = 150) -> bytes:
    """Apply linear fade-out to 16-bit PCM audio."""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    fade_samples = min(int(SAMPLE_RATE * fade_ms / 1000), len(samples))

    if fade_samples > 0:
        envelope = np.linspace(1.0, 0.0, fade_samples)
        samples[:fade_samples] *= envelope

    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
```

### Example 4: Sending interruption annotation to Claude CLI

```python
# Source: existing _send_to_cli() pattern in live_session.py
async def _send_interrupt_annotation(self, spoken_text: str, unspoken_text: str):
    """Notify Claude CLI that the AI was interrupted."""
    annotation = (
        f"[The user interrupted you. They heard up to: \"{spoken_text[-200:]}\". "
        f"Your remaining response was: \"{unspoken_text[:300]}\". "
        f"It was not spoken. Adjust based on what the user says next.]"
    )
    await self._send_to_cli(annotation)
    # Don't read response — the next user message will trigger the actual response
```

### Example 5: STT stage with VAD integration (conceptual)

```python
# In _stt_stage(), replace the playing_audio/muted gate:
if self._stt_gated:
    # Don't buffer for transcription, but DO run VAD
    if self.barge_in_enabled and self.playing_audio:
        prob = self._run_vad_chunk(frame.data)
        if prob > VAD_THRESHOLD:
            self._barge_speech_frames += 1
            if (self._barge_speech_frames >= BARGE_IN_FRAMES_REQUIRED
                    and time.time() > self._barge_in_cooldown_until):
                self._trigger_barge_in()
        else:
            self._barge_speech_frames = 0
    # Clear buffer regardless
    audio_buffer.clear()
    silence_start = None
    has_speech = False
    speech_chunk_count = 0
    continue
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual ONNX file + onnxruntime | `silero-vad` pip package v6.2 bundles model | Nov 2025 (v6.2) | No need to manage model files separately |
| WebRTC VAD | Silero VAD (neural network) | 2021-2022 | Much higher accuracy, same speed (<1ms) |
| Mic mute during playback | STT gating (mic stays live) | This phase | Enables VAD-based barge-in |
| Keyboard-only interrupt | VAD-based voice interrupt + keyboard fallback | This phase | More natural conversation flow |

**Deprecated/outdated:**
- The existing `_load_vad_model()` manually loads ONNX from `models/silero_vad.onnx` which doesn't exist. Replace with `load_silero_vad(onnx=True)` from the pip package.
- The existing `_run_vad()` method manually manages h/c/sr tensors. The pip package's `OnnxWrapper` handles state internally.
- The existing `_vad_monitor_stage()` is a no-op with a TODO comment. Replace with integrated VAD in the STT stage.

## Open Questions

1. **VAD and audio_in_q consumption**
   - What we know: `audio_in_q` is an asyncio.Queue consumed by `_stt_stage()`. Only one consumer reads from a queue.
   - What's unclear: Should VAD run inside the STT stage (simplest, no queue duplication) or as a separate consumer with a broadcast pattern?
   - Recommendation: Run VAD inside the STT stage. When gated, the STT stage already reads and discards frames — just add VAD processing before discarding. This avoids queue fan-out complexity.

2. **Tracking which text was spoken vs interrupted**
   - What we know: `full_response` accumulates all LLM text. `_bytes_played` tracks playback bytes. The sentence buffer sends complete sentences to TTS.
   - What's unclear: Exact mapping between played bytes and source text (TTS is not constant-rate).
   - Recommendation: Track at the sentence level. Maintain a list of sentences sent to TTS and a counter of sentences whose TTS audio has fully played. On interrupt, sentences up to the counter are "spoken", the rest are "interrupted". This is imprecise but sufficient — the annotation doesn't need word-level accuracy.

3. **Filler clip selection for interruption context**
   - What we know: CONTEXT.md says "contextually appropriate filler clip during fade-out" and this is marked as Claude's discretion.
   - What's unclear: What makes a filler "contextually appropriate" for an interruption moment vs a thinking moment.
   - Recommendation: Use the same nonverbal clip pool. Short clips (< 0.5s) are better for interruption moments. Could tag clips by duration and prefer shorter ones during barge-in, but this is refinement — any nonverbal clip is fine for v1.

4. **Overlay visual pulse**
   - What we know: CONTEXT.md specifies "brief visual pulse/flash on the overlay when barge-in activates."
   - What's unclear: How to implement a temporary visual change in the GTK overlay from the asyncio pipeline thread.
   - Recommendation: Use the existing `_set_status()` callback which calls `GLib.idle_add()` for thread-safety. Flash a temporary status (e.g., "interrupted" with a distinct color) for ~300ms, then transition to "listening". Or simpler: just transition directly to "listening" — the quick status change in the history panel provides the visual record.

## Sources

### Primary (HIGH confidence)
- `silero-vad` pip package v6.2.0 — verified locally: `pip install silero-vad`, model loads via `load_silero_vad(onnx=True)`, raw inference returns probabilities, OnnxWrapper manages state
- Existing codebase (`live_session.py`) — verified: `_vad_monitor_stage()`, `_run_vad()`, `_check_interrupt()`, `request_interrupt()`, generation_id system, `_mute_mic()` / `_unmute_mic()`, `_playback_stage()`, `_stt_stage()`
- Pipecat SileroVADAnalyzer docs — verified VAD params: threshold=0.5-0.7, start_secs=0.2, stop_secs=0.2, 512 samples at 16kHz, 256 at 8kHz

### Secondary (MEDIUM confidence)
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad) — chunk sizes, sample rates, streaming usage patterns
- [Silero VAD Wiki: Examples and Dependencies](https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies) — VADIterator API, reset_states(), dependency requirements
- [Optimizing Voice Agent Barge-In Detection 2025](https://sparkco.ai/blog/optimizing-voice-agent-barge-in-detection-for-2025) — industry patterns: sub-100ms response, 10-20ms frame processing, echo cancellation approaches
- [Pipecat framework](https://github.com/pipecat-ai/pipecat) — reference architecture for barge-in in voice agents

### Tertiary (LOW confidence)
- [Building Performant Voice AI Agents](https://medium.com/@danielostapenko/building-performant-voice-ai-agents-ce0810eb2cf8) — general patterns for voice agent interruption handling

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — silero-vad pip package verified locally, all dependencies confirmed installed, API tested
- Architecture: HIGH — all patterns verified against existing codebase, no external dependencies needed, builds on existing infrastructure (generation_id, interrupt mechanism, filler system)
- Pitfalls: MEDIUM — headphone bleed and VAD state corruption are known issues in voice agent literature; specific thresholds may need tuning in practice
- Code examples: HIGH — all examples verified locally or derived from existing working code patterns

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (30 days — silero-vad is stable, codebase architecture is stable)
