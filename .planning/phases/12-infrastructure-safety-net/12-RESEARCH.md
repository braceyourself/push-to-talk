# Phase 12: Infrastructure + Safety Net - Research

**Researched:** 2026-02-21
**Domain:** Continuous audio capture, STT pipeline, echo cancellation, transcript buffer, VRAM management, hallucination filtering
**Confidence:** HIGH (stack and architecture verified against official docs, codebase, and hardware)

## Summary

Phase 12 builds the always-on listening foundation: continuous audio capture with Silero VAD gating, faster-whisper distil-large-v3 for streaming STT, PipeWire WebRTC echo cancellation, a bounded transcript ring buffer, VRAM monitoring for concurrent Whisper + Ollama, and hallucination filtering tuned for ambient audio. The phase produces no autonomous AI behavior -- it outputs a clean transcript stream that Phase 13's decision engine will consume.

The system already has all the building blocks: Silero VAD v5 loaded via ONNX for barge-in detection (live_session.py line 1306), faster-whisper with int8_float16 quantization and multi-layer hallucination filtering (line 1448), pasimple for PulseAudio/PipeWire recording (line 2046), and a daemon-thread capture pattern with async queue bridging (line 2044). The transformation is: remove the PTT gating, run capture and STT continuously, route audio through PipeWire's echo-cancelled virtual source, upgrade the Whisper model from large-v3 to distil-large-v3, expand the hallucination filter for ambient noise conditions, and add a bounded transcript buffer as the output.

**Primary recommendation:** Validate VRAM budget empirically first (go/no-go gate), then configure PipeWire AEC (30-min spike), then build the continuous STT and transcript buffer as independently testable components before pipeline integration.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faster-whisper | 1.2.1 (installed) | Continuous STT via CTranslate2 | Already in use; 4x faster than openai-whisper, int8 quantization, built-in VAD filter |
| Silero VAD v5 | ONNX (installed) | Voice activity detection for capture gating | Already loaded in codebase; <1ms per chunk on CPU; v5 fixed near-silence false detection |
| pasimple | 0.0.3 (installed) | PulseAudio/PipeWire audio recording | Already in use; supports device_name param for targeting AEC virtual source |
| libpipewire-module-echo-cancel | PipeWire 1.0.5 (installed) | WebRTC acoustic echo cancellation | Ships with PipeWire; libspa-aec-webrtc.so confirmed present on this machine |
| nvidia-ml-py | latest | VRAM monitoring via NVML | Official NVIDIA Python bindings; thin wrapper, no transitive deps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| collections.deque | stdlib | Ring buffer for transcript segments | Bounded maxlen, O(1) append/pop, thread-safe in CPython |
| onnxruntime | (installed) | Silero VAD inference | Already used for barge-in VAD; CPU-only, no GPU VRAM impact |
| numpy | (installed) | Audio sample conversion | Already used throughout codebase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| faster-whisper distil-large-v3 | whisper_streaming (ufal) | whisper_streaming adds LocalAgreement-n policy for streaming; adds complexity, but the existing chunked approach is simpler and proven in this codebase |
| PipeWire AEC | Software echo fingerprinting | PipeWire AEC operates at the audio driver level (zero latency, transparent to app); software fingerprinting is a fallback if AEC doesn't work well enough |
| collections.deque | Custom ring buffer class | deque with maxlen is sufficient; custom class only needed if time-based eviction (rather than count-based) is required |
| nvidia-ml-py | subprocess nvidia-smi | nvidia-ml-py is ~100x faster than parsing nvidia-smi output; direct NVML access |

**Installation:**
```bash
# In the service venv
pip install nvidia-ml-py

# PipeWire AEC config (no package install needed -- module already present)
mkdir -p ~/.config/pipewire/pipewire.conf.d/
# Create config file (see Code Examples section)
systemctl --user restart pipewire pipewire-pulse
```

## Architecture Patterns

### Recommended Component Structure
```
live_session.py (existing)
    |-- continuous_stt.py (new)     # VAD-gated Whisper loop, produces TranscriptSegments
    |-- transcript_buffer.py (new)  # Bounded ring buffer with time-based eviction
    |-- vram_monitor.py (new)       # NVML-based GPU memory watchdog
    |-- (echo cancellation)         # PipeWire config, not Python code
```

### Pattern 1: VAD-Gated Continuous Capture
**What:** Silero VAD gates which audio chunks are sent to Whisper. Audio capture runs at all times; VAD runs on every chunk; only speech segments accumulate into a buffer that gets transcribed. This resolves the CONTEXT.md conflict: capture is truly continuous (CSTR-01), but Whisper only processes speech segments (not silence/noise).
**When to use:** Always -- this is the core pipeline pattern.
**How it works:**
1. PulseAudio/PipeWire capture thread reads 4096-byte chunks at 24kHz 16-bit mono (~85ms each)
2. Each chunk is downsampled to 16kHz and run through Silero VAD (<1ms, CPU)
3. If VAD probability > threshold (0.5): accumulate chunk into speech buffer, reset silence counter
4. If VAD probability <= threshold: increment silence counter
5. When silence exceeds duration threshold (0.8s) AND speech buffer has content: send buffer to Whisper
6. Safety cap: force transcription after MAX_BUFFER_SECONDS (10s) regardless of silence detection
7. Whisper runs in thread executor (blocking call, GPU), returns transcript
8. Transcript goes through hallucination filter, then into TranscriptBuffer

**Key insight:** The existing `_stt_stage` (line 2178) already implements this exact pattern with energy-based silence detection. The change is: replace energy thresholds with Silero VAD probabilities (already loaded), remove the PTT gating (`_stt_flush_event`), and keep the capture running through AI playback (using AEC instead of `_stt_gated`).

### Pattern 2: Echo-Cancelled Audio Source
**What:** PipeWire's echo-cancel module creates a virtual audio source that subtracts speaker output from microphone input using WebRTC AEC.
**When to use:** Always when the system has both mic and speaker active.
**How it works:**
1. A PipeWire config file loads `libpipewire-module-echo-cancel` with `monitor.mode = true`
2. This creates a virtual source named "Echo Cancellation Source"
3. The module reads the default speaker output (monitor ports) as the reference signal
4. It subtracts the reference from the microphone input using the WebRTC AEC algorithm
5. Applications record from the virtual source instead of the raw mic
6. In Python: `pasimple.PaSimple(..., device_name="Echo Cancellation Source")`

**The `monitor.mode = true` is critical:** Without it, the module creates a virtual sink that applications must route audio through. With it, the module monitors the default output directly -- no application routing changes needed except the capture device name.

### Pattern 3: Transcript Ring Buffer with Time-Based Eviction
**What:** A bounded buffer holding recent transcript segments with both count-based and time-based limits.
**When to use:** As the output of the STT pipeline and input to Phase 13's decision engine.
**How it works:**
1. `collections.deque(maxlen=N)` provides count-based bounding (e.g., 200 segments max)
2. Each segment is a dataclass: `TranscriptSegment(text, timestamp, confidence, source)`
3. On each append, also evict segments older than the time window (5 minutes)
4. `get_context()` method returns the buffer contents formatted for LLM consumption
5. `get_since(timestamp)` returns segments after a given time for incremental reads
6. Thread-safe: deque.append() and len() are atomic in CPython

### Pattern 4: Proactive VRAM Management
**What:** Monitor GPU VRAM usage and take preemptive action before OOM.
**When to use:** Continuously while Whisper and Ollama are loaded.
**How it works:**
1. NVML initialization at startup, periodic polling (every 30s)
2. Three thresholds: WARNING (75% = 6144MB), CRITICAL (87.5% = 7168MB), EMERGENCY (95% = 7782MB)
3. WARNING: log message, reduce Ollama num_ctx to 1024
4. CRITICAL: unload Ollama model (keep Whisper -- transcription has priority per CONTEXT.md decision)
5. EMERGENCY: fall back to CPU-only Whisper (distil-large-v3 or small model)
6. Recovery: when VRAM drops below WARNING, reload Ollama model

### Anti-Patterns to Avoid
- **Transcribing every chunk individually:** Each Whisper inference has fixed overhead. Accumulate speech segments and transcribe in batches (0.5-10 seconds of audio). The existing pattern does this correctly.
- **Running VAD on GPU:** Silero VAD ONNX runs on CPU in <1ms. Moving it to GPU wastes VRAM and adds no benefit. Current code correctly uses `CPUExecutionProvider`.
- **Unbounded audio buffer accumulation:** The existing MAX_BUFFER_SECONDS safety cap (10s) is essential. Without it, a noisy environment where VAD never detects silence would grow the buffer indefinitely.
- **Blocking the event loop with Whisper:** Whisper inference takes 0.5-3s. Must run in thread executor (already done at line 1428). Never await directly in the async loop.
- **Reloading the Whisper model per transcription:** The model must stay loaded in GPU memory between calls. Current code does this correctly (lazy init at line 1453, stays loaded).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Echo cancellation | Software audio fingerprinting / correlation | PipeWire `libpipewire-module-echo-cancel` with WebRTC AEC | Operates at audio driver level; handles latency compensation, adaptive filtering, and speaker/mic coupling automatically; battle-tested in video conferencing |
| Voice activity detection | Energy threshold + zero-crossing rate | Silero VAD v5 (already loaded) | Handles variable noise floors, non-stationary noise, and gradual volume changes that energy thresholds miss; <1ms on CPU |
| GPU VRAM monitoring | Parsing `nvidia-smi` output | `nvidia-ml-py` (NVML Python bindings) | Direct C library access via ctypes; ~100x faster than subprocess; provides memory, utilization, temperature in a single call |
| Ring buffer | Custom circular array with index tracking | `collections.deque(maxlen=N)` | Thread-safe append/pop, automatic eviction at maxlen, zero-copy iteration, stdlib |
| Whisper hallucination filtering | Custom NLP-based classifier | Multi-layer filter: VAD pre-gate + no_speech_prob + avg_logprob + compression_ratio + phrase blocklist | Existing pattern (line 1474-1489) is proven; extend rather than replace |
| Audio capture reconnection | Manual socket/file descriptor management | pasimple context manager + reconnect loop | Existing pattern (line 2034-2067) handles PulseAudio errors with automatic retry |

**Key insight:** The existing codebase already has proven patterns for all of these except echo cancellation and VRAM monitoring. The phase is mostly about reconfiguring and extending existing components, not building from scratch.

## Common Pitfalls

### Pitfall 1: PipeWire AEC Device Not Found by pasimple
**What goes wrong:** After configuring echo cancel, `pasimple.PaSimple(device_name="Echo Cancellation Source")` fails because the device name doesn't match.
**Why it happens:** PipeWire exposes the echo-cancelled source with a node name from the config, but pasimple uses PulseAudio device names. The PulseAudio name may differ from the PipeWire node name. Monitor mode vs non-monitor mode also changes what devices are created.
**How to avoid:** After configuring AEC, run `pactl list short sources` and find the exact source name containing "echo" or "Echo". Use that exact string as the `device_name`. Alternatively, set the PipeWire source node.name to something identifiable and use `pactl` to find the corresponding PulseAudio name. Fall back to default source if AEC source not found (graceful degradation).
**Warning signs:** "Connection refused" or "No such device" errors from pasimple at startup. The system captures raw mic audio without echo cancellation.

### Pitfall 2: Whisper Hallucination Explosion on Ambient Audio
**What goes wrong:** With continuous capture, Whisper processes keyboard typing, HVAC hum, and room tone -- generating "thank you", "thanks for watching", "so", "the end" etc. at rates up to 40% of segments.
**Why it happens:** Whisper was trained on video transcriptions. When given non-speech audio, it generates the most common phrases from its training data. The existing 18-phrase blocklist was tuned for PTT (clean speech). Research shows the top 10 hallucinated phrases account for over 50% of all hallucinations (arXiv 2501.11378).
**How to avoid:** Multi-layer defense: (1) Silero VAD pre-gate prevents non-speech from reaching Whisper at all, (2) faster-whisper's built-in `vad_filter=True` provides a second VAD pass, (3) `no_speech_prob >= 0.6` threshold (already at line 1475), (4) `avg_logprob < -1.0` (already at line 1481), (5) `compression_ratio > 2.4` (already at line 1487), (6) expanded phrase blocklist (~30+ phrases), (7) single-word reject for segments with only one word and no_speech_prob > 0.3. Use `condition_on_previous_text=False` (already set at line 1463) to prevent hallucination loops.
**Warning signs:** Transcript log filling with repeated phrases. High hallucination rate in the stats display.

### Pitfall 3: VRAM OOM During Concurrent Whisper + Ollama Inference
**What goes wrong:** Both Whisper and Ollama run GPU inference at the same time, combined VRAM exceeds 8GB, CUDA OOM kills one or both processes.
**Why it happens:** Whisper distil-large-v3 int8_float16 uses ~1481MB static + ~300MB during inference. Ollama Llama 3.2 3B Q4_K_M uses ~2000-2500MB + KV cache growth (~110MB per 1000 tokens). Combined: ~4000-4500MB baseline, spikes to ~5500-6000MB during concurrent inference. Should fit 8GB, but CTranslate2 and Ollama may each allocate CUDA memory pools with headroom.
**How to avoid:** VRAM validation spike MUST be the first task. Load both models simultaneously, run Whisper transcription while Ollama generates response, monitor with `nvidia-smi dmon -s u`. If peak exceeds 7GB: (a) reduce Ollama num_ctx to 1024, (b) if still too high, use Whisper small model for continuous STT (fallback chain per CONTEXT.md). Proactive VRAM monitoring (Pattern 4) detects pressure before OOM.
**Warning signs:** CUDA memory allocation errors. Ollama or Whisper silently failing. System slowdown from GPU memory swapping.

### Pitfall 4: faster-whisper Memory Leak in Long Sessions
**What goes wrong:** RSS memory grows over hours of continuous transcription, eventually causing system instability.
**Why it happens:** CTranslate2 may not fully release CUDA memory between transcriptions. Community reports (faster-whisper issue #249, #390, #992) document gradual growth, though the #249 root cause was a cpu_threads misconfiguration. GPU memory (~312MB) may remain allocated after model deletion, only released when the process exits.
**How to avoid:** Monitor RSS every 5 minutes. If growth exceeds 20% of baseline after 2+ hours, perform a model reload: delete the WhisperModel object, call `gc.collect()` and `torch.cuda.empty_cache()` (if torch is available), then recreate the model. Schedule a periodic model reload every 2 hours as a preventive measure. Log reload events for post-mortem analysis.
**Warning signs:** Gradual RSS growth in system stats. Increasing Whisper inference latency.

### Pitfall 5: Echo Cancellation Fails Silently
**What goes wrong:** PipeWire AEC module loads but doesn't effectively cancel echo. AI hears its own TTS and generates transcripts of it.
**Why it happens:** AEC effectiveness depends on the acoustic coupling between speaker and microphone, the quality of the reference signal alignment, and the hardware characteristics. The module may be loaded but not properly correlated if latency is too high or the reference signal path is wrong. `monitor.mode = true` captures from the default output, but if the application plays audio through a different sink, the reference signal misses it.
**How to avoid:** Test with a known audio clip: play it through speakers, record from the AEC source, verify the recording is significantly attenuated. Add a software-level fallback: when the system detects that a transcript closely matches recently spoken TTS text (cosine similarity or substring match against a spoken-sentences ring buffer), tag it as `source=ai` and filter it. This is already partially implemented via `_spoken_sentences` in the codebase.
**Warning signs:** Transcript log showing text that matches what the AI just said. Feedback loop where AI responds to its own previous response.

### Pitfall 6: Silero VAD Threshold Too Aggressive or Too Lenient
**What goes wrong:** Too high (>0.7): short utterances ("hi", "hey") get missed -- user speaks but nothing is transcribed. Too low (<0.3): keyboard typing and ambient noise trigger Whisper transcription constantly, wasting GPU and producing hallucinations.
**Why it happens:** VAD threshold interacts with the room's noise floor, microphone sensitivity, and speaker distance. A value tuned for one environment may fail in another.
**How to avoid:** Start with 0.5 (current barge-in threshold, line 2274). Add auto-calibration: measure ambient VAD scores during a 5-second startup window, set threshold at ambient_max + 0.15. Make threshold configurable in config.json. Log VAD trigger rates for tuning. Per CLAUDE.md: "Never gate safety caps on has_speech" -- the MAX_BUFFER_SECONDS cap must fire unconditionally.
**Warning signs:** User reports "it doesn't hear me" (threshold too high) or transcript log full of noise (threshold too low).

## Code Examples

### PipeWire Echo Cancellation Config
```bash
# ~/.config/pipewire/pipewire.conf.d/echo-cancel.conf
# Source: https://docs.pipewire.org/page_module_echo_cancel.html
# Source: https://wiki.archlinux.org/title/PipeWire/Examples

context.modules = [
    {   name = libpipewire-module-echo-cancel
        args = {
            # Use monitor.mode to capture from default output (no sink routing needed)
            monitor.mode = true

            # WebRTC AEC is the default and only supported algorithm
            # library.name = aec/libspa-aec-webrtc

            capture.props = {
                node.name = "Echo Cancellation Capture"
                # Optionally target a specific mic (comment out for default)
                # node.target = "alsa_input.usb-Creative_Technology_Ltd_Sound_Blaster_GC7_F3AD470FB0F33647-03.analog-stereo"
                node.passive = true
            }
            source.props = {
                node.name   = "Echo Cancellation Source"
                node.description = "Echo-Cancelled Microphone"
            }
            playback.props = {
                node.name = "Echo Cancellation Playback"
                node.passive = true
            }
        }
    }
]
```

After creating this file:
```bash
systemctl --user restart pipewire pipewire-pulse

# Verify the source was created:
pactl list short sources | grep -i echo
# Expected output similar to:
# 12345   Echo_Cancellation_Source   PipeWire   s16le 1ch 48000Hz   IDLE

# Find the exact PulseAudio device name:
ECHO_SOURCE=$(pactl list short sources | grep -i "echo.*source" | awk '{print $2}')
echo "Use device_name='$ECHO_SOURCE' in pasimple"
```

### Capture with Echo-Cancelled Source
```python
# Source: existing pattern from live_session.py line 2046, adapted for AEC
import pasimple

# Try echo-cancelled source first, fall back to default
AEC_DEVICE = None  # Will be auto-detected or configured
try:
    # Test if AEC source exists
    test_pa = pasimple.PaSimple(
        pasimple.PA_STREAM_RECORD,
        pasimple.PA_SAMPLE_S16LE,
        1, 24000,
        app_name='push-to-talk',
        device_name="Echo Cancellation Source"
    )
    test_pa.read(4096)  # Quick test read
    del test_pa
    AEC_DEVICE = "Echo Cancellation Source"
    print("Audio: Using echo-cancelled source")
except Exception:
    AEC_DEVICE = None  # Fall back to default source
    print("Audio: AEC source not found, using default mic (echo cancellation disabled)")

# In the record thread:
with pasimple.PaSimple(
    pasimple.PA_STREAM_RECORD,
    pasimple.PA_SAMPLE_S16LE,
    CHANNELS, SAMPLE_RATE,
    app_name='push-to-talk',
    device_name=AEC_DEVICE,  # None = default source
) as pa:
    while not stop_event.is_set():
        data = pa.read(CHUNK_SIZE)
        # ... enqueue to audio_in_q
```

### VRAM Monitoring with NVML
```python
# Source: nvidia-ml-py PyPI / NVML API docs
import pynvml

class VRAMMonitor:
    """Monitor GPU VRAM usage and trigger actions at thresholds."""

    # Thresholds for RTX 3070 (8192 MB)
    WARNING_MB = 6144      # 75% -- reduce Ollama context
    CRITICAL_MB = 7168     # 87.5% -- unload Ollama
    EMERGENCY_MB = 7782    # 95% -- fall back to CPU Whisper

    def __init__(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.total_mb = info.total // (1024 * 1024)

    def get_usage_mb(self) -> int:
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used // (1024 * 1024)

    def get_free_mb(self) -> int:
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.free // (1024 * 1024)

    def check(self) -> str:
        """Returns 'ok', 'warning', 'critical', or 'emergency'."""
        used = self.get_usage_mb()
        if used >= self.EMERGENCY_MB:
            return 'emergency'
        elif used >= self.CRITICAL_MB:
            return 'critical'
        elif used >= self.WARNING_MB:
            return 'warning'
        return 'ok'

    def shutdown(self):
        pynvml.nvmlShutdown()
```

### Transcript Ring Buffer
```python
# Source: Python stdlib collections.deque + dataclass pattern
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class TranscriptSegment:
    text: str
    timestamp: float  # time.time()
    confidence: float = 1.0  # from Whisper logprob
    source: str = "user"  # "user", "ai", "other", "filtered"
    no_speech_prob: float = 0.0

class TranscriptBuffer:
    """Bounded ring buffer for transcript segments with time-based eviction."""

    def __init__(self, max_segments: int = 200, max_age_seconds: float = 300.0):
        self._buffer = deque(maxlen=max_segments)
        self._max_age = max_age_seconds
        self._lock = Lock()  # For time-based eviction (deque ops are atomic but iteration isn't)

    def append(self, segment: TranscriptSegment):
        with self._lock:
            self._evict_old()
            self._buffer.append(segment)

    def get_context(self, max_tokens: int = 2048) -> str:
        """Return buffer contents formatted for LLM consumption."""
        with self._lock:
            self._evict_old()
            lines = []
            char_count = 0
            for seg in reversed(self._buffer):
                line = f"[{seg.source}] {seg.text}"
                if char_count + len(line) > max_tokens * 4:  # ~4 chars per token
                    break
                lines.append(line)
                char_count += len(line)
            return "\n".join(reversed(lines))

    def get_since(self, timestamp: float) -> list[TranscriptSegment]:
        with self._lock:
            return [s for s in self._buffer if s.timestamp > timestamp]

    def _evict_old(self):
        cutoff = time.time() - self._max_age
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def __len__(self):
        return len(self._buffer)
```

### Expanded Hallucination Filter
```python
# Source: arXiv 2501.11378 (Whisper hallucination research) + existing filter at line 2193
# Top hallucinated phrases from research: "thank you" (24.76%), "thanks for watching" (10.32%),
# "so" (3.80%), "thank you for watching" (2.58%), "the" (2.50%)

HALLUCINATION_PHRASES = {
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
}

def is_hallucination(text: str, no_speech_prob: float = 0.0) -> bool:
    """Multi-layer hallucination check for continuous transcription."""
    cleaned = text.lower().strip().rstrip('.!?,')

    # Layer 1: exact phrase match
    if cleaned in HALLUCINATION_PHRASES:
        return True

    # Layer 2: single word with elevated no_speech_prob
    words = cleaned.split()
    if len(words) == 1 and no_speech_prob > 0.3:
        return True

    # Layer 3: very short text (1-2 chars) -- likely noise
    if len(cleaned) <= 2:
        return True

    # Layer 4: repetitive content (e.g., "thank you thank you thank you")
    if len(words) >= 4:
        unique_words = set(words)
        if len(unique_words) <= 2:
            return True

    return False
```

### VAD-Gated Continuous STT (Skeleton)
```python
# Source: existing _stt_stage pattern (line 2178) + Silero VAD (line 1306)
# This shows the core loop structure -- not a complete implementation

async def continuous_stt_loop(self):
    """Continuously capture audio, gate with VAD, transcribe speech segments."""
    audio_buffer = bytearray()
    silence_chunks = 0
    SILENCE_CHUNKS_THRESHOLD = 10  # ~850ms at 85ms/chunk
    MAX_BUFFER_CHUNKS = 118  # ~10 seconds
    MIN_BUFFER_CHUNKS = 6   # ~510ms

    speech_detected = False
    chunks_in_buffer = 0

    while self.running:
        try:
            frame = self._audio_in_q.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.02)
            continue

        if frame.type != FrameType.AUDIO_RAW:
            continue

        # Run VAD on every chunk (CPU, <1ms)
        vad_prob = self._run_vad(frame.data)

        if vad_prob > self._vad_threshold:
            # Speech detected
            audio_buffer.extend(frame.data)
            chunks_in_buffer += 1
            silence_chunks = 0
            speech_detected = True
        else:
            # Silence
            if speech_detected:
                # Still accumulating post-speech silence
                audio_buffer.extend(frame.data)
                chunks_in_buffer += 1
                silence_chunks += 1

                if silence_chunks >= SILENCE_CHUNKS_THRESHOLD:
                    # End of utterance -- transcribe
                    if chunks_in_buffer >= MIN_BUFFER_CHUNKS:
                        pcm_data = bytes(audio_buffer)
                        transcript = await asyncio.get_event_loop().run_in_executor(
                            None, self._whisper_transcribe, pcm_data
                        )
                        if transcript and not is_hallucination(transcript):
                            segment = TranscriptSegment(
                                text=transcript,
                                timestamp=time.time(),
                                source="user"
                            )
                            self._transcript_buffer.append(segment)

                    audio_buffer.clear()
                    chunks_in_buffer = 0
                    silence_chunks = 0
                    speech_detected = False

        # Safety cap: force transcription after max buffer
        if chunks_in_buffer >= MAX_BUFFER_CHUNKS:
            pcm_data = bytes(audio_buffer)
            transcript = await asyncio.get_event_loop().run_in_executor(
                None, self._whisper_transcribe, pcm_data
            )
            if transcript and not is_hallucination(transcript):
                segment = TranscriptSegment(
                    text=transcript,
                    timestamp=time.time(),
                    source="user"
                )
                self._transcript_buffer.append(segment)

            audio_buffer.clear()
            chunks_in_buffer = 0
            silence_chunks = 0
            speech_detected = False
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Whisper large-v3 for STT | distil-large-v3 for continuous STT | 2023 (distil-whisper release) | 51% smaller (756M vs 1550M params), 6.3x faster, within 1% WER; ~1481MB vs ~2953MB VRAM with int8 |
| Energy threshold silence detection | Silero VAD speech detection | Silero v5, 2025 | Handles variable noise floors; v5 fixed near-silence false positives that plagued earlier versions |
| PulseAudio module-echo-cancel | PipeWire libpipewire-module-echo-cancel | PipeWire 0.3+ | Ships as default audio stack on modern Ubuntu; WebRTC AEC is built-in; monitor.mode simplifies config |
| nvidia-smi subprocess parsing | nvidia-ml-py NVML bindings | Long established | Direct C library access, ~100x faster, structured data instead of text parsing |
| Manual hallucination blacklist | Research-backed phrase list + multi-layer filter | arXiv 2501.11378 (Jan 2025) | Top 10 phrases account for 50%+ of hallucinations; "Bag of Hallucinations" approach provides systematic coverage |

**Deprecated/outdated:**
- `pynvml` package (gpuopenanalytics): deprecated in favor of `nvidia-ml-py` (official NVIDIA package)
- Silero VAD v4 and earlier: v5 fixed significant near-silence false detection issues; faster-whisper PR #884 updated to v5
- `condition_on_previous_text=True`: Confirmed to cause hallucination loops in continuous transcription; always set to False

## VRAM Budget Analysis

**This is the go/no-go gate for Phase 12.** Must be validated empirically before building anything else.

### Theoretical Budget (RTX 3070, 8192 MB)

| Component | Compute Type | Static VRAM | Peak VRAM | Source |
|-----------|-------------|-------------|-----------|--------|
| Whisper distil-large-v3 | int8_float16 | ~1481 MB | ~1800 MB | faster-whisper benchmark #1030 |
| Ollama Llama 3.2 3B | Q4_K_M | ~2000 MB | ~2500 MB | Ollama VRAM guide + KV cache |
| CUDA runtime overhead | -- | ~300 MB | ~500 MB | Typical allocation |
| **Total** | | **~3781 MB** | **~4800 MB** | |
| **Headroom** | | **~4411 MB** | **~3392 MB** | |

This looks comfortable on paper (~58% utilization at peak), but CUDA memory pool fragmentation, CTranslate2 workspace allocations, and Ollama's memory management strategy can increase actual usage. The worst case is both models running inference simultaneously during a burst.

### Fallback Chain (per CONTEXT.md: Whisper has priority)

| VRAM State | Action | VRAM Freed |
|------------|--------|------------|
| < 6144 MB (75%) | Normal operation | -- |
| 6144-7168 MB | Reduce Ollama num_ctx to 1024 | ~200 MB |
| 7168-7782 MB | Unload Ollama model entirely | ~2000-2500 MB |
| > 7782 MB (95%) | Switch Whisper to CPU mode | All GPU VRAM |

### Validation Procedure (30-minute spike)
```bash
# 1. Load Whisper distil-large-v3
python3 -c "
from faster_whisper import WhisperModel
m = WhisperModel('distil-large-v3', device='cuda', compute_type='int8_float16')
print('Whisper loaded')
input('Press Enter to continue...')
"

# 2. In another terminal, load Ollama
ollama run llama3.2:3b "Hello" --verbose

# 3. Monitor VRAM during concurrent use
nvidia-smi dmon -s u -d 2  # VRAM usage every 2 seconds

# 4. Stress test: run continuous Whisper transcription while Ollama generates
# Use the actual audio capture + transcription code, not just model loading
```

## Clarification: VAD + Continuous Capture Interaction

The CONTEXT.md says "Whisper runs continuously (no VAD gating)" but CSTR-01 says "gated by Silero VAD to only process speech segments." These are compatible:

- **Audio capture** runs continuously -- the microphone is always recording, no PTT button
- **Silero VAD** runs on every audio chunk -- determines if each chunk contains speech
- **Whisper STT** only processes accumulated speech segments -- VAD gates what reaches Whisper

The "no VAD gating" in CONTEXT.md means "no requirement for the user to press a button (PTT) to start capture." The capture is always on. VAD gates what gets transcribed, not what gets captured. This is the correct architecture: running Whisper on every 85ms chunk of ambient audio would waste GPU, produce hallucinations, and add no value.

## Open Questions

1. **Exact PulseAudio device name for AEC source**
   - What we know: PipeWire creates a node named "Echo Cancellation Source" per the config. PulseAudio sees PipeWire nodes.
   - What's unclear: The exact string pasimple needs. It may be the node name directly, or a PulseAudio-formatted name like `Echo_Cancellation_Source` (spaces replaced with underscores).
   - Recommendation: 30-minute spike on this machine. Create the config, restart PipeWire, run `pactl list short sources`, test with pasimple. This is a low-risk investigation with a deterministic answer.

2. **distil-large-v3 Hallucination Rate vs large-v3**
   - What we know: HuggingFace docs say distil-large-v3 has "improved robustness to hallucinations" due to WER-filtered training data. The arXiv hallucination research only tested large-v3 (40.3% rate on non-speech).
   - What's unclear: The actual hallucination rate of distil-large-v3 on ambient audio. It may be significantly lower than large-v3's 40.3% but we have no direct measurement.
   - Recommendation: Accept this uncertainty. The multi-layer filter (VAD pre-gate + Whisper metrics + phrase blocklist) handles both models. Monitor hallucination rate in production and tune thresholds as needed.

3. **CTranslate2 CUDA Memory Pool Behavior**
   - What we know: faster-whisper (CTranslate2 backend) may pre-allocate a CUDA memory pool larger than the model's actual weights. Some users report ~312MB of unreleased GPU memory after model deletion (issue #992).
   - What's unclear: Whether this pool is bounded and whether it interferes with Ollama's own CUDA allocations.
   - Recommendation: The VRAM validation spike will reveal actual behavior. If memory pools conflict, the process isolation approach (running Whisper in a subprocess) is a known workaround from issue #660.

4. **PipeWire AEC Effectiveness with USB Sound Card**
   - What we know: This machine uses a Creative Sound Blaster GC7 (USB). PipeWire AEC works with any PipeWire source/sink. The GC7 has its own DAC/ADC which means the speaker-to-mic acoustic path is short (same device).
   - What's unclear: How well WebRTC AEC handles the specific acoustic coupling of this hardware. USB audio devices may have higher latency than onboard audio, affecting AEC convergence.
   - Recommendation: Test during the 30-minute spike. Play a known audio clip through speakers, record from AEC source, compare. If AEC doesn't cancel well, the software fallback (transcript fingerprinting against spoken sentences) is the backup.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `live_session.py` (3000+ lines) -- direct inspection of audio capture (line 2020-2067), STT pipeline (line 2178-2380), VAD (line 1306-1391), Whisper transcription (line 1448-1506), hallucination filter (line 2193-2207)
- [PipeWire Echo Cancel Module docs](https://docs.pipewire.org/page_module_echo_cancel.html) -- official configuration reference
- [PipeWire/Examples ArchWiki](https://wiki.archlinux.org/title/PipeWire/Examples) -- practical echo cancel config with monitor.mode
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper) -- API, benchmarks, model support
- [faster-whisper benchmark issue #1030](https://github.com/SYSTRAN/faster-whisper/issues/1030) -- VRAM measurements: distil-large-v3 int8 = 1481MB, large-v3 int8 = 2953MB
- [pasimple GitHub](https://github.com/henrikschnor/pasimple) -- PaSimple API, device_name parameter, PipeWire compatibility
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad) -- v6.2, <1ms/chunk, ONNX 4-5x faster than PyTorch
- nvidia-smi on this machine -- RTX 3070, 8192 MB, currently 1155 MB used
- PipeWire system inspection -- v1.0.5, libspa-aec-webrtc.so confirmed at /usr/lib/x86_64-linux-gnu/spa-0.2/aec/
- pactl on this machine -- Sound Blaster GC7 source/sink confirmed

### Secondary (MEDIUM confidence)
- [distil-whisper/distil-large-v3 HuggingFace](https://huggingface.co/distil-whisper/distil-large-v3) -- 756M params, 6.3x faster, within 1% WER, reduced hallucinations
- [arXiv 2501.11378: Whisper ASR Hallucinations](https://arxiv.org/html/2501.11378v1) -- 40.3% hallucination rate on non-speech, top phrases documented, Bag of Hallucinations approach
- [arXiv 2505.12969: Calm-Whisper](https://arxiv.org/html/2505.12969v1) -- 80% hallucination reduction via targeted fine-tuning of 3 decoder heads
- [whisper_streaming (ufal)](https://github.com/ufal/whisper_streaming) -- LocalAgreement-n streaming policy, 3.3s latency
- [faster-whisper memory issues #249, #390, #992](https://github.com/guillaumekln/faster-whisper/issues/249) -- CPU memory from thread misconfiguration; GPU 312MB residual
- [Ollama VRAM requirements guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) -- 3B model ~2-3GB VRAM with Q4_K_M
- [nvidia-ml-py PyPI](https://pypi.org/project/nvidia-ml-py/) -- official NVIDIA NVML Python bindings

### Tertiary (LOW confidence)
- Combined VRAM usage under concurrent inference load -- theoretical estimate only; empirical validation required
- PipeWire AEC effectiveness with USB Sound Blaster GC7 -- standard WebRTC AEC, but hardware-specific coupling is untested
- distil-large-v3 hallucination rate on ambient audio -- claimed "improved" but no direct measurement exists

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified installed or available; versions confirmed; API checked against official docs
- Architecture: HIGH -- patterns derived from existing working codebase with exact line references; PipeWire module confirmed installed
- VRAM budget: MEDIUM -- theoretical numbers from published benchmarks; actual concurrent usage on this hardware unknown (go/no-go gate)
- Pitfalls: HIGH -- multi-source corroboration (academic research, GitHub issues, codebase inspection, community reports)

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable domain; faster-whisper and PipeWire are mature)
