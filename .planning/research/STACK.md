# Stack Research: Deepgram Streaming STT + Local Decision Model

**Domain:** Always-on voice assistant with cloud STT and local decision model
**Researched:** 2026-02-22
**Confidence:** HIGH (Deepgram SDK, pricing), MEDIUM (VRAM budgets, model quality)
**Supersedes:** Previous STACK.md (2026-02-21) which assumed local Whisper STT

## Executive Summary

The architectural pivot from local Whisper to Deepgram streaming STT requires one new dependency (`deepgram-sdk` 5.3.2) and changes the VRAM landscape significantly. Whisper distil-large-v3 consumed ~1.5-2GB GPU VRAM; removing it frees that entirely for a larger/better local decision model. The recommended architecture is: Silero VAD (local, CPU) gates audio to Deepgram Nova-3 (cloud, ~$0.46/hr of actual speech), which returns word-level streaming results with ~150ms latency. The freed VRAM opens the door for Llama 3.1 8B (Q4_K_M, ~5GB) instead of the originally planned Llama 3.2 3B (~2.5GB), giving substantially better reasoning and classification quality.

Deepgram bills per second of audio sent, not per WebSocket connection time. Sending KeepAlive messages during silence costs nothing. Combined with local Silero VAD gating (only stream speech segments), the cost for typical desktop use (~15-30 min of actual speech per 8hr workday) works out to $0.12-0.23/day.

## Stack Changes from Previous STACK.md

| Component | Previous Plan (2026-02-21) | New Plan (2026-02-22) | Rationale |
|-----------|---------------------------|----------------------|-----------|
| STT | faster-whisper (local, GPU) | Deepgram Nova-3 (cloud, streaming) | 150ms streaming latency vs 850ms silence gap + 500ms-2s batch inference |
| STT VRAM | ~1.5-2GB (distil-large-v3 int8_float16) | 0 MB (cloud) | Frees GPU for larger decision model |
| Decision model | Llama 3.2 3B (Q4, ~2.5GB VRAM) | Llama 3.1 8B (Q4_K_M, ~5GB VRAM) | Freed VRAM allows 2.5x larger model with better reasoning |
| New dependency | `ollama` 0.6.1 | `ollama` 0.6.1 + `deepgram-sdk` 5.3.2 | One additional library |
| faster-whisper | Kept (continuous use) | Kept but idle (fallback only) | Offline/degradation fallback when Deepgram unavailable |

## Recommended Stack Additions

### 1. Deepgram Python SDK

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `deepgram-sdk` | 5.3.2 (stable) | Streaming STT via WebSocket | Official SDK. Handles WebSocket lifecycle, KeepAlive, reconnection. Supports threaded and async patterns. Already in requirements.txt as `deepgram-sdk>=3.0` (needs version pin update). |

**Confidence: HIGH** -- Verified on [PyPI](https://pypi.org/project/deepgram-sdk/) (5.3.2, released 2026-01-29). v6.0.0-rc.2 exists (2026-02-18) but is pre-release with breaking changes; avoid until stable.

**Key API surface for this project:**

```python
from deepgram import DeepgramClient, LiveOptions
from deepgram.core.events import EventType

# Connection setup
deepgram = DeepgramClient()  # reads DEEPGRAM_API_KEY from env

options = LiveOptions(
    model="nova-3",
    language="en",
    encoding="linear16",
    sample_rate=24000,       # match existing capture rate
    channels=1,
    # Streaming control
    interim_results=True,     # get partial results as speech happens
    utterance_end_ms="1000",  # signal after 1s of silence
    endpointing=300,          # finalize after 300ms silence (good for conversation)
    vad_events=True,          # get speech start/end events
    smart_format=True,        # auto-format numbers, dates
    punctuate=True,           # add punctuation
)

# Threaded (recommended by Deepgram for real-time audio)
with deepgram.listen.v1.connect(options) as connection:
    connection.on(EventType.OPEN, on_open)
    connection.on(EventType.MESSAGE, on_message)    # transcript results
    connection.on(EventType.CLOSE, on_close)
    connection.on(EventType.ERROR, on_error)

    # Send audio data
    connection.send(audio_bytes)

    # KeepAlive during silence (every 3-5s)
    connection.keep_alive()

    # Graceful shutdown
    connection.finish()
```

**Message types received:**
- `ListenV1Results` -- Transcript with `is_final`, `speech_final`, `channel.alternatives[0].transcript`
- `ListenV1Metadata` -- Session info (request_id, model, duration)
- `ListenV1UtteranceEnd` -- Silence detected (after `utterance_end_ms`)
- `ListenV1SpeechStarted` -- VAD speech start event

**Critical behavior for decision engine integration:**
1. `interim_results=True`: Returns partial transcripts every ~1s as user speaks (for real-time display)
2. `is_final=True`: A segment is finalized (Deepgram won't revise it)
3. `speech_final=True`: An endpoint was detected (natural pause in speech)
4. Buffer pattern: Accumulate all `is_final=True` transcripts. When `speech_final=True` arrives, the buffer contains the complete utterance. Feed this to the decision model.
5. `UtteranceEnd`: After `utterance_end_ms` of silence, signals user stopped talking entirely

**KeepAlive mechanism:**
- Send `{"type": "KeepAlive"}` as a text WebSocket frame every 3-5 seconds during silence
- If no audio or KeepAlive sent within 10 seconds, connection closes with NET-0001 error
- SDK's `connection.keep_alive()` method handles the message format
- KeepAlive messages incur no billing cost

**Connection limits:**
- 60-minute maximum per WebSocket connection
- Must create a new connection after 60 minutes (implement reconnection logic)
- No concurrent connection limit documented for pay-as-you-go

**Dependencies:** `deepgram-sdk` 5.3.2 requires Python >=3.8,<4.0. Dependencies are httpx (already present), websockets (already present), and aiohttp. The `websockets` package is already in requirements.txt for the OpenAI Realtime API integration.

### 2. Deepgram Nova-3 (Cloud Service)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Deepgram Nova-3 | Latest (API-side) | Streaming speech-to-text | ~150ms first-word latency. WER ~6.84% streaming (best-in-class). Native endpointing + utterance detection. Word-level timestamps. Handles accents and background noise well. |

**Confidence: HIGH** -- [Official pricing](https://deepgram.com/pricing), [benchmarks](https://deepgram.com/learn/speech-to-text-benchmarks), [API docs](https://developers.deepgram.com/reference/speech-to-text/listen-streaming) all verified.

**Pricing analysis for this project:**

| Metric | Value | Source |
|--------|-------|--------|
| Pay-as-you-go rate | $0.0077/min ($0.46/hr) | [Deepgram pricing page](https://deepgram.com/pricing) |
| Billing granularity | Per-second (no rounding) | [Deepgram pricing FAQ](https://deepgram.com/learn/speech-to-text-api-pricing-breakdown-2025) |
| WebSocket idle cost | $0 (only audio sent is billed) | [Deepgram discussion #1423](https://github.com/orgs/deepgram/discussions/1423) |
| KeepAlive cost | $0 | Same source |
| Free credit | $200 (no expiry, no CC required) | Pricing page |
| Free credit duration | ~433 hours of streaming audio | Calculated: $200 / $0.46/hr |

**Estimated daily cost (desktop use with local VAD gating):**

| Usage Pattern | Speech Minutes/Day | Daily Cost |
|---------------|-------------------|------------|
| Light (casual chat) | 15 min | $0.12 |
| Moderate (regular conversation) | 30 min | $0.23 |
| Heavy (podcast-like all day) | 120 min | $0.92 |
| Raw always-on (no VAD gate) | 480 min (8hr) | $3.70 |

The VAD gate is critical: without it, 8 hours of always-on streaming costs $3.70/day. With Silero VAD gating (only send speech segments), typical cost is $0.12-0.23/day. The $200 free credit covers ~870-1667 days of light use.

### 3. Upgraded Decision Model: Llama 3.1 8B

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Llama 3.1 8B Instruct | `llama3.1:8b` via Ollama | Transcript monitoring and response decisions | Freed VRAM from removing Whisper allows upgrading from 3B to 8B. Significantly better reasoning, instruction following, and classification accuracy. Still fits in the VRAM budget with Q4_K_M quantization (~5GB). |

**Confidence: MEDIUM** -- VRAM figure of ~5GB for Q4_K_M is from community benchmarks and Ollama documentation. Real-world usage may vary with KV cache growth. Needs empirical validation.

**Why upgrade from Llama 3.2 3B to 3.1 8B:**

| Benchmark | Llama 3.2 3B | Llama 3.1 8B | Improvement |
|-----------|-------------|-------------|-------------|
| MMLU | ~60% | ~68% | +8 points |
| GPQA Diamond | Lower | Higher | Significant |
| HumanEval | Lower | Higher | Significant |
| Instruction following | Good | Better | Matters for structured JSON decisions |
| Reasoning depth | Adequate for simple classification | Strong enough for nuanced context analysis | Critical for "should I respond?" decisions |

The decision model's job is nuanced: given a rolling transcript window, it must distinguish "user talking to AI" from "user talking to someone else" from "TV dialogue" from "user thinking aloud." The 8B model's stronger reasoning makes these judgments more reliable.

**VRAM requirement (Q4_K_M):**

| Component | Estimated VRAM |
|-----------|---------------|
| Model weights | ~4.5 GB |
| KV cache (short context, ~2K tokens) | ~0.3 GB |
| CUDA overhead | ~0.2 GB |
| **Total** | **~5.0 GB** |

**Risk: Context window management.** The KV cache grows linearly with context length. At 32K tokens, the KV cache alone needs ~4.5GB, which would exceed the 8GB budget. Must cap context via Ollama's `num_ctx` option (recommend 4096 tokens max, adding ~0.6GB KV cache).

**Fallback chain:**
1. Primary: Llama 3.1 8B (Q4_K_M, ~5GB VRAM)
2. Fallback A: Llama 3.2 3B (Q4_K_M, ~2.5GB VRAM) -- if 8B causes VRAM pressure
3. Fallback B: Heuristic classifier only (0 VRAM) -- if Ollama is down

**Installation:**
```bash
# Pull the upgraded model (4.7GB download, one-time)
ollama pull llama3.1:8b

# Keep the 3B as fallback
# ollama pull llama3.2:3b  (already installed from previous plan)
```

**Ollama configuration adjustment:**
```bash
# Previous plan reserved 4GB for Whisper. Now Whisper is off GPU.
# Set OLLAMA_GPU_OVERHEAD to ~1.5GB (for CUDA runtime + desktop + Silero VAD)
sudo systemctl edit ollama.service
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=-1"
# Environment="OLLAMA_GPU_OVERHEAD=1500000000"
# Environment="OLLAMA_NUM_PARALLEL=1"
```

### 4. Ollama Python Client (Unchanged)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `ollama` | 0.6.1 | Python client for Ollama API | Same as previous STACK.md. AsyncClient for decision model queries. Zero new transitive deps. |

**Confidence: HIGH** -- No change from previous research. See previous STACK.md for full API reference.

### 5. Existing Stack Retained

| Component | Status | Notes |
|-----------|--------|-------|
| `faster-whisper` | Kept, idle by default | Fallback STT when Deepgram unavailable (offline mode). Model not loaded unless needed -- 0 VRAM when idle. |
| Silero VAD (ONNX) | Repurposed | Previously: barge-in detection during TTS. New role: cost gate for Deepgram (only stream speech segments). Still ~50MB, CPU-only. |
| `pasimple` | Unchanged | Audio capture source. Same 24kHz 16-bit mono. Now feeds Deepgram WebSocket instead of Whisper. |
| `pyaudio` | Unchanged | Audio playback. |
| `model2vec` | Unchanged | Semantic classification for quick responses. |
| `pynput` | Modified role | Keyboard listener. PTT key becomes optional "force respond" key. AI hotkey still used for mode switching. |
| `onnxruntime` | Unchanged | Silero VAD inference. CPU-only provider. |
| `pynvml` | Updated thresholds | VRAMMonitor thresholds change: no Whisper means more headroom, but 8B model uses more than 3B. |

## VRAM Budget Analysis (New Architecture)

**Hardware:** NVIDIA RTX 3070, 8192 MiB total

### Previous Architecture (Whisper + Llama 3.2 3B)

| Model | Precision | VRAM | Notes |
|-------|-----------|------|-------|
| Whisper distil-large-v3 | int8_float16 | ~1.5-2.0 GB | CTranslate2 |
| Llama 3.2 3B | Q4_K_M | ~2.0-2.5 GB | Via Ollama |
| Silero VAD | ONNX CPU | ~50 MB | Negligible |
| CUDA runtime + desktop | -- | ~1.5 GB | Baseline |
| **Total** | | **~5.0-6.0 GB** | **2-3 GB headroom** |

### New Architecture (Deepgram + Llama 3.1 8B)

| Model | Precision | VRAM | Notes |
|-------|-----------|------|-------|
| Whisper | -- | 0 MB | Off GPU (cloud STT) |
| Llama 3.1 8B | Q4_K_M | ~5.0 GB | Via Ollama, num_ctx capped at 4096 |
| Silero VAD | ONNX CPU | ~50 MB | Negligible |
| CUDA runtime + desktop | -- | ~1.5 GB | Baseline |
| **Total** | | **~6.5 GB** | **~1.7 GB headroom** |

### Alternative: Conservative VRAM Budget (Llama 3.2 3B)

| Model | Precision | VRAM | Notes |
|-------|-----------|------|-------|
| Whisper | -- | 0 MB | Off GPU |
| Llama 3.2 3B | Q4_K_M | ~2.5 GB | Smaller but less capable |
| Silero VAD | ONNX CPU | ~50 MB | Negligible |
| CUDA runtime + desktop | -- | ~1.5 GB | Baseline |
| **Total** | | **~4.0 GB** | **~4.2 GB headroom** |

**Recommendation:** Start with Llama 3.1 8B. If VRAM pressure causes issues (monitor with VRAMMonitor), fall back to 3.2 3B. The 8B model's better reasoning quality is worth the tighter VRAM budget for a desktop-only single-user application where no other GPU workloads compete.

**Updated VRAMMonitor thresholds for new architecture:**

```python
# New thresholds (Llama 3.1 8B + no Whisper)
WARNING_MB = 7000      # 85% -- getting tight
CRITICAL_MB = 7500     # 91.5% -- consider downgrading model
EMERGENCY_MB = 7900    # 96.5% -- force model unload
```

## Alternatives Considered

### For Streaming STT

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Deepgram Nova-3 | AssemblyAI Universal-Streaming | Similar latency (~300ms) and streaming support. $0.15/hr for connection time (more expensive than Deepgram's ~$0.46/hr for audio). Deepgram's per-second-of-audio billing is better for VAD-gated use where the WebSocket is open but mostly idle. |
| Deepgram Nova-3 | Groq Whisper | Batch only, not streaming. ~24s for 30min file (fast for batch), but adds chunking latency for real-time use. No native WebSocket streaming. Would require building chunked processing pipeline (500ms-2s latency per chunk). |
| Deepgram Nova-3 | Google Cloud Speech-to-Text | More expensive ($0.024/min streaming for enhanced model). Requires GCP setup. Deepgram has simpler API and lower latency for conversational use. |
| Deepgram Nova-3 | Local Whisper (faster-whisper) | Previous approach. Batch-only means 850ms silence gap + 500ms-2s inference = 1.5-3s total latency. Too slow for natural conversational awareness. Keep as offline fallback only. |

**Confidence: MEDIUM** -- AssemblyAI and Groq pricing may have changed. Cross-referenced multiple sources but some are from 2025.

### For Decision Model

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Llama 3.1 8B (Q4) | Llama 3.2 3B (Q4) | Still the fallback option. 3B is adequate but 8B gives meaningfully better reasoning for the nuanced "should I respond?" classification. With Whisper off GPU, we have the VRAM for 8B. |
| Llama 3.1 8B (Q4) | Qwen 2.5 7B (Q4) | Similar VRAM (~5GB). Qwen 2.5 has strong benchmarks but Llama 3.1 has wider Ollama community support and more tested instruction-following behavior. Could be swapped if Llama 3.1 8B disappoints in practice. |
| Llama 3.1 8B (Q4) | Qwen2.5-Omni-3B (audio-native) | Theoretically interesting: audio-native model could skip STT entirely and classify directly from audio. But: (a) not well-supported in Ollama for audio input, (b) 3B is too small for reliable classification even with audio understanding, (c) adds massive complexity (custom inference pipeline), (d) would still need text transcripts for the response backends. Not worth the risk for uncertain gains. |
| Llama 3.1 8B (Q4) | Qwen3-Omni-30B-A3B (MoE) | Audio-native successor with 3B active parameters (30B total). Impressive benchmarks but: (a) requires vLLM for efficient inference (Ollama MoE support is limited), (b) total model size ~30GB on disk, (c) VRAM unknown for A3B active params with quantization, (d) no established Ollama integration for audio modality. Future option when ecosystem matures. |
| Llama 3.1 8B (Q4) | Fine-tuned small classifier | A fine-tuned 1B or 3B model specifically for "should I respond?" classification would be faster and more accurate for this specific task. But: requires training data collection (thousands of labeled examples), training infrastructure, and ongoing maintenance. The general-purpose 8B model with good prompting is the right starting point. If classification accuracy is a bottleneck later, fine-tuning a smaller model is a v3.0 optimization. |
| Llama 3.1 8B (Q4) | Phi-4 Mini (3.8B) | Slightly larger than 3B but not the reasoning leap of 8B. Not worth the middle ground. Either use 3B for minimum VRAM or 8B for maximum quality. |

### For Audio-Native Models (Skip STT Entirely?)

Evaluated and rejected for v2.0:

**Qwen2.5-Omni-3B:**
- Can do ASR (speech recognition) natively, performs well on Librispeech benchmarks
- Available via Ollama as `qwen2.5:3b-omni`
- Problem: Ollama's multimodal support for audio input is immature. The model is designed for transformers/vLLM inference pipelines, not Ollama's text-centric API.
- Problem: Even if it worked, 3B is too small for reliable "should I respond?" decisions on ambient conversation context.
- Problem: You still need text transcripts for the response backends (Claude CLI, personality context), so you can't skip STT entirely.
- Verdict: Interesting technology but wrong tool for this job in 2026.

**Qwen3-Omni-30B-A3B:**
- Next-gen audio-native model with MoE architecture (30B total, 3B active)
- Open-source SOTA on 32 of 36 audio benchmarks
- Problem: Requires vLLM-Omni for efficient inference (Ollama MoE support is poor)
- Problem: Model weight files are ~30GB on disk
- Problem: VRAM requirement for quantized A3B inference is undocumented
- Verdict: Most promising future option but ecosystem is too immature for production use today.

## What NOT to Add

| Avoid | Why | Impact if Added |
|-------|-----|-----------------|
| `deepgram-sdk` v6 (pre-release) | v6.0.0-rc.2 has breaking API changes from v5. Not stable yet. Would require migration when stable release lands. | API instability, undocumented behavior |
| Qwen2.5-Omni via Ollama | Audio modality support in Ollama is immature. Would require custom inference code. Still need text transcripts anyway. | Complex custom pipeline, uncertain reliability |
| openWakeWord | Same rationale as previous STACK.md: STT transcript already provides name detection. Now even less needed since Deepgram returns results in ~150ms (faster than Whisper's 500ms+). | Unnecessary complexity |
| Custom WebSocket client | Deepgram SDK handles WebSocket lifecycle, KeepAlive, reconnection. Rolling your own is worse in every dimension. | Maintenance burden, miss edge cases |
| AssemblyAI SDK | Evaluated and rejected. Connection-time billing is worse for always-on-but-mostly-silent use pattern. | Higher cost |
| Redis / message queue | Same as previous STACK.md: asyncio.Queue is sufficient. | Unnecessary infrastructure |

## Integration Points with Existing Stack

### 1. ContinuousSTT Replacement

**Current:** `continuous_stt.py` -- VAD-gated Whisper batch transcription loop.
**New:** Replace Whisper transcription with Deepgram streaming. Keep the VAD gating but repurpose it:

```python
# Current flow (continuous_stt.py):
#   pasimple capture -> VAD check -> accumulate buffer -> Whisper transcribe -> TranscriptBuffer

# New flow:
#   pasimple capture -> VAD check -> if speech: send to Deepgram WebSocket
#                                    if silence: send KeepAlive
#   Deepgram events -> on_message -> TranscriptBuffer

# Key architectural change:
# - No more batch accumulation and silence-gap-triggered transcription
# - Deepgram handles endpointing and utterance detection server-side
# - VAD is now a COST GATE, not a transcription trigger
```

### 2. TranscriptBuffer (Unchanged)

`transcript_buffer.py` -- Ring buffer accepting `TranscriptSegment` objects. No changes needed. Deepgram results are converted to `TranscriptSegment` with `source="user"` and fed to the same buffer.

### 3. Audio Capture (Minor Change)

`pasimple` capture thread is unchanged. The audio bytes go to Deepgram WebSocket instead of a local Whisper model. Same 24kHz 16-bit mono format. Deepgram supports `linear16` encoding natively with configurable `sample_rate`.

### 4. VRAMMonitor (Threshold Update)

Thresholds need adjustment for the new VRAM profile (no Whisper, larger Ollama model). The emergency fallback changes from "switch Whisper to CPU" to "downgrade Ollama model from 8B to 3B."

### 5. EventBus (New Event Types)

```python
# New events for Deepgram integration
DEEPGRAM_CONNECTED = "deepgram_connected"
DEEPGRAM_DISCONNECTED = "deepgram_disconnected"
DEEPGRAM_INTERIM = "deepgram_interim"       # Partial transcript (for display)
DEEPGRAM_FINAL = "deepgram_final"           # Final transcript segment
DEEPGRAM_UTTERANCE_END = "deepgram_utterance_end"  # Silence detected
DEEPGRAM_ERROR = "deepgram_error"
```

### 6. API Key Management

```python
# Deepgram API key follows same pattern as OpenAI:
# 1. Environment variable: DEEPGRAM_API_KEY
# 2. Config file: ~/.config/deepgram/api_key
# 3. DeepgramClient() reads DEEPGRAM_API_KEY from env automatically

# In config.json, add:
# "deepgram_api_key_source": "env"  (or "file" with path)
```

### 7. Graceful Degradation Chain (Updated)

```
Normal operation:
  Audio -> Silero VAD -> Deepgram WebSocket -> TranscriptBuffer -> Decision Model (Ollama 8B)

Deepgram down (network issue):
  Audio -> Silero VAD -> faster-whisper (load on demand) -> TranscriptBuffer -> Decision Model

Ollama down:
  Audio -> Silero VAD -> Deepgram -> TranscriptBuffer -> Heuristic classifier + Claude CLI

Both down (offline + GPU issue):
  Audio -> Silero VAD -> faster-whisper (CPU) -> TranscriptBuffer -> Heuristic classifier only
```

## Performance Budget (Updated)

| Operation | Target | Estimated | Notes |
|-----------|--------|-----------|-------|
| Deepgram first-word latency | <300ms | ~150ms | [Deepgram benchmarks](https://deepgram.com/learn/speech-to-text-benchmarks). Network dependent. |
| Deepgram interim results | ~1s intervals | ~1s | Partial transcripts during speech |
| Deepgram final result after speech | <500ms | ~200-300ms | After endpointing triggers |
| VAD gating decision | <5ms | <1ms | Silero ONNX on CPU |
| Decision model (Ollama 8B, non-streaming) | <1s | ~300-500ms | Llama 3.1 8B with `num_predict: 100`, structured JSON. Slower than 3B (~200ms) but still fast. |
| Decision model (Ollama 8B, streaming) | <500ms TTFT | ~300ms | First token for quick responses. |
| Name detection in transcript | <1ms | <0.1ms | Simple string `in` check on Deepgram results |
| End-to-end: user speaks -> AI starts responding | <2s | ~1-1.5s | VAD (<1ms) + Deepgram stream (~150ms) + endpointing (300ms) + decision model (~400ms) + TTS start (~200ms) |

**Latency comparison with previous architecture:**

| Stage | Previous (Whisper) | New (Deepgram) | Improvement |
|-------|-------------------|----------------|-------------|
| Silence detection | 850ms (local) | 300ms (Deepgram endpointing) | -550ms |
| Transcription | 500ms-2s (batch) | ~150ms (streaming) | -350ms to -1.85s |
| Total STT latency | 1.35-2.85s | ~450ms | 67-84% faster |
| Decision model | ~200ms (3B) | ~400ms (8B) | +200ms (tradeoff for quality) |
| Net end-to-end improvement | | | ~0.5-2.25s faster |

## Installation

```bash
# 1. Upgrade deepgram-sdk (already in requirements.txt, needs version pin)
source ~/.local/share/push-to-talk/venv/bin/activate
pip install deepgram-sdk==5.3.2

# 2. Pull the upgraded decision model (4.7GB download, one-time)
ollama pull llama3.1:8b

# 3. Set Deepgram API key
# Option A: Environment variable (recommended)
# Add to ~/.config/systemd/user/push-to-talk.service [Service] section:
# Environment="DEEPGRAM_API_KEY=your-key-here"
# Or source from a file in the service ExecStart script

# Option B: Config file
# mkdir -p ~/.config/deepgram
# echo "your-key-here" > ~/.config/deepgram/api_key

# 4. Update Ollama GPU overhead reservation
sudo systemctl edit ollama.service
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=-1"
# Environment="OLLAMA_GPU_OVERHEAD=1500000000"
# Environment="OLLAMA_NUM_PARALLEL=1"
sudo systemctl daemon-reload && sudo systemctl restart ollama

# 5. Verify Deepgram
python3 -c "
from deepgram import DeepgramClient
client = DeepgramClient()
print('Deepgram client initialized successfully')
print(f'SDK version: {client.__class__.__module__}')
"

# 6. Verify Ollama with 8B model
python3 -c "
import asyncio
from ollama import AsyncClient
async def test():
    r = await AsyncClient().chat(
        model='llama3.1:8b',
        messages=[{'role':'user','content':'Respond with only: OK'}],
        options={'num_predict': 10}
    )
    print(f'8B model response: {r.message.content}')
asyncio.run(test())
"
```

Update `requirements.txt`:
```
# Streaming STT (Deepgram)
deepgram-sdk>=5.3,<6.0

# Local LLM monitoring (Ollama client)
ollama>=0.6
```

Note: Pin `deepgram-sdk` below 6.0 to avoid the breaking v6 API changes until v6 goes stable.

## Version Compatibility

| Package | Version | Compatible With | Notes |
|---------|---------|-----------------|-------|
| `deepgram-sdk` | 5.3.2 | Python >=3.8,<4.0 | Stable release. Avoid v6 pre-release. |
| `ollama` | 0.6.1 | Python 3.8+, httpx >=0.27, pydantic >=2.9 | No change. |
| `httpx` | 0.28.1 (installed) | deepgram-sdk 5.3.2, ollama 0.6.1 | Shared dependency, already present. |
| `websockets` | installed | deepgram-sdk 5.3.2 | Already present for OpenAI Realtime. |
| `faster-whisper` | 1.2.1 (installed) | Kept as fallback | Not loaded by default. 0 VRAM when idle. |
| Ollama server | latest | RTX 3070 (CUDA), Llama 3.1 8B | Primary decision model. |
| Deepgram Nova-3 | API-managed | Streaming STT | No local version management needed. |

## Data Footprint

| Component | Size | Notes |
|-----------|------|-------|
| `deepgram-sdk` package | ~2 MB | Pure Python, thin HTTP/WebSocket wrapper |
| Llama 3.1 8B model | 4.7 GB on disk | In `~/.ollama/models/`. Replaces or supplements 3B (2.0GB). |
| Llama 3.2 3B model | 2.0 GB on disk | Keep as fallback |
| Whisper distil-large-v3 | ~1.5 GB on disk | Keep for offline fallback. Not loaded by default. |
| `ollama` Python package | ~100 KB | No change |
| **Total new disk usage** | ~4.8 GB | Deepgram SDK + 8B model download |

## Deepgram API Configuration Quick Reference

For roadmap authors -- key configuration choices and their tradeoffs:

**Recommended settings for always-on voice assistant:**

```python
LiveOptions(
    model="nova-3",           # Best accuracy + speed
    language="en",            # Single language, monolingual pricing ($0.0077/min)
    encoding="linear16",      # Raw PCM, no encoding overhead
    sample_rate=24000,        # Match existing capture rate
    channels=1,               # Mono mic input
    interim_results=True,     # Real-time display + early decision model input
    utterance_end_ms="1000",  # 1s silence = user done talking
    endpointing=300,          # 300ms pause = natural breath, finalize segment
    vad_events=True,          # Know when speech starts/stops
    smart_format=True,        # Format numbers, dates, currencies
    punctuate=True,           # Add punctuation
)
```

**Settings to avoid:**
- `endpointing=10` (default): Too aggressive, fragments natural speech
- `utterance_end_ms` < 1000: Interim results are ~1s intervals, so <1000ms adds no benefit
- `diarize=True`: Adds latency, increases cost. Speaker detection is better done locally.
- `multichannel=True`: Only one mic, one speaker stream

## Sources

- [Deepgram Python SDK GitHub](https://github.com/deepgram/deepgram-python-sdk) -- v5.3.2 API, v6.0.0-rc.2 pre-release status (HIGH confidence)
- [Deepgram PyPI](https://pypi.org/project/deepgram-sdk/) -- v5.3.2 stable, released 2026-01-29 (HIGH confidence)
- [Deepgram Pricing](https://deepgram.com/pricing) -- $0.0077/min streaming, $200 free credit (HIGH confidence)
- [Deepgram Streaming API Reference](https://developers.deepgram.com/reference/speech-to-text/listen-streaming) -- Parameters, message types, events (HIGH confidence)
- [Deepgram Endpointing + Interim Results Guide](https://developers.deepgram.com/docs/understand-endpointing-interim-results) -- Configuration tradeoffs (HIGH confidence)
- [Deepgram KeepAlive](https://developers.deepgram.com/docs/audio-keep-alive) -- 10s timeout, 3-5s send interval, text frame format (HIGH confidence)
- [Deepgram Utterance End](https://developers.deepgram.com/docs/utterance-end) -- utterance_end_ms behavior (HIGH confidence)
- [Deepgram VAD Cost Discussion](https://github.com/orgs/deepgram/discussions/1216) -- Local VAD before Deepgram, billing per audio sent (HIGH confidence)
- [Deepgram Billing Discussion](https://github.com/orgs/deepgram/discussions/1423) -- WebSocket idle = $0, KeepAlive = $0 (HIGH confidence)
- [Deepgram Nova-3 Benchmarks](https://deepgram.com/learn/speech-to-text-benchmarks) -- WER ~6.84% streaming (HIGH confidence)
- [Llama 3.1 8B vs 3.2 3B Comparison](https://medium.com/@marketing_novita.ai/llama-3-1-8b-vs-llama-3-2-3b-balancing-power-and-mobile-efficiency-eb4c3856c4af) -- Benchmark comparison (MEDIUM confidence, community source)
- [Ollama VRAM Requirements Guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) -- 8B Q4 ~5-6GB, 3B Q4 ~2-3GB (MEDIUM confidence, community source)
- [Ollama Performance on 8GB GPUs](https://aimuse.blog/article/2025/06/08/ollama-performance-tuning-on-8gb-gpus-a-practical-case-study-with-qwen3-models) -- KV cache growth, 7.6GB cliff on RTX 3070 (MEDIUM confidence, community source)
- [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni) -- Audio-native model evaluation (HIGH confidence for capabilities, LOW for Ollama integration)
- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni) -- MoE audio model, requires vLLM (HIGH confidence)
- [AssemblyAI Pricing](https://www.assemblyai.com/pricing) -- $0.15/hr connection time (MEDIUM confidence, may have changed)
- [Groq vs Deepgram Comparison](https://deepgram.com/learn/whisper-vs-deepgram) -- Batch vs streaming tradeoffs (HIGH confidence)
- Existing codebase: `continuous_stt.py`, `vram_monitor.py`, `requirements.txt`, `input_classifier.py` -- Direct code inspection (HIGH confidence)

---
*Stack research for: Deepgram Streaming STT + Local Decision Model*
*Researched: 2026-02-22*
*Supersedes: v2.0 Always-On Observer STACK.md (2026-02-21)*
