# Stack Research: v2.0 Always-On Observer

**Domain:** Always-on voice assistant with local LLM monitoring layer
**Researched:** 2026-02-21
**Confidence:** HIGH (core stack), MEDIUM (VRAM budgeting)

## Executive Summary

The v2.0 always-on observer requires three stack additions: the **Ollama Python client** (`ollama` 0.6.1) for local LLM integration, **Ollama server** with Llama 3.2 3B for the monitoring layer, and architectural changes to the existing audio/STT pipeline for continuous operation. No other new dependencies are needed. The `ollama` library's only dependencies are `httpx` (0.28.1, already installed) and `pydantic` (2.12.5, already installed), so it adds zero transitive dependencies. Name-based interruption ("hey Russel") should use the existing Whisper STT transcript stream -- not a separate wake word library -- because the always-on STT already produces continuous transcripts that can be string-matched. The critical constraint is GPU VRAM: the RTX 3070 has 8GB, Whisper large-v3 with int8 uses ~3-4GB, and Llama 3.2 3B Q4 uses ~2-3GB, leaving them both fitting but tight. A fallback strategy (Whisper on CPU or downgrade to distil-large-v3) must be planned.

## Recommended Stack Additions

### 1. Ollama Server (System-Level)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Ollama | latest (install via curl script) | Local LLM inference server | Manages model loading, GPU allocation, and HTTP API. Runs as a systemd service. Handles model lifecycle (keep_alive, loading, unloading) without application code. Already decided in PROJECT.md -- Ollama + Llama 3.2 3B is the monitoring layer. |
| Llama 3.2 3B | `llama3.2:3b` (2.0GB download) | Transcript monitoring and response decisions | 128K context window. Fits in ~2-3GB VRAM with Q4_K_M quantization. Outperforms Gemma 2 2.6B on instruction following and summarization. Fast enough for monitoring (~200ms for short classification responses with `num_predict` limit). |

**Confidence: HIGH** -- Ollama's install is a single curl command. Llama 3.2 3B is a first-party Meta model with 57.5M+ downloads on Ollama's registry. Verified model page at [ollama.com/library/llama3.2](https://ollama.com/library/llama3.2).

**Installation:**
```bash
# Install Ollama server
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (2.0GB download, one-time)
ollama pull llama3.2:3b

# Configure for always-on (model stays loaded in GPU)
# Add to /etc/systemd/system/ollama.service [Service] section:
# Environment="OLLAMA_KEEP_ALIVE=-1"
# Environment="OLLAMA_GPU_OVERHEAD=500000000"  # Reserve 500MB for other CUDA apps
```

### 2. Ollama Python Client

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `ollama` | 0.6.1 | Python client for Ollama API | Official library. AsyncClient integrates directly with existing asyncio event loop. Streaming support via `async for`. Structured JSON output via `format` parameter with Pydantic schemas. Dependencies: httpx >=0.27 (have 0.28.1) + pydantic >=2.9 (have 2.12.5) = zero new transitive deps. |

**Confidence: HIGH** -- Verified on [PyPI](https://pypi.org/project/ollama/) (0.6.1, released 2025-11-13). Dependencies confirmed present in venv. AsyncClient API verified via [GitHub README](https://github.com/ollama/ollama-python).

**Integration pattern:**
```python
from ollama import AsyncClient

# Non-streaming for fast classification decisions
async def should_respond(transcript_window: str) -> dict:
    response = await AsyncClient().chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'system', 'content': MONITOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': transcript_window}
        ],
        format={  # Structured output via JSON schema
            'type': 'object',
            'properties': {
                'should_respond': {'type': 'boolean'},
                'reason': {'type': 'string'},
                'complexity': {'type': 'string', 'enum': ['quick', 'deep']},
                'response_text': {'type': 'string'}
            },
            'required': ['should_respond', 'reason', 'complexity']
        },
        options={
            'temperature': 0,       # Deterministic for classification
            'num_predict': 100,     # Cap output tokens for speed
        },
        keep_alive=-1,  # Keep model loaded
    )
    return json.loads(response.message.content)

# Streaming for Ollama-generated responses (quick mode)
async def generate_quick_response(prompt: str):
    async for part in await AsyncClient().chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        options={'num_predict': 200},
    ):
        yield part['message']['content']
```

**Key API details verified:**
- `AsyncClient()` defaults to `http://localhost:11434`
- `format` parameter accepts JSON schema dict or `"json"` string for unstructured JSON
- `options` dict supports: `temperature`, `num_predict`, `top_p`, `top_k`, `seed`, `num_ctx`, `stop`
- `keep_alive` accepts: seconds (int), duration string ("5m", "24h"), or -1 (forever)
- Error handling: catch `ollama.ResponseError`, check `.status_code` (404 = model not found)
- Streaming returns `AsyncIterator[ChatResponse]` objects with `.message.content`

### 3. Existing Stack Modifications (No New Dependencies)

These changes use existing dependencies but are architecturally significant:

| Component | Current | v2.0 Change | Why |
|-----------|---------|-------------|-----|
| Audio capture (`pasimple`) | PTT-gated: only captures when key held | Always-on: captures continuously, independent of LLM state | Decoupled input stream. Same `pasimple.PaSimple` API, remove PTT gate condition. |
| Whisper STT (`faster-whisper`) | Triggered by silence detection after PTT | Continuous: rolling buffer, transcribe on silence boundaries | Same model, same API. Change is in the buffering/trigger logic, not the library. |
| Silero VAD (`onnxruntime`) | Only runs during AI playback for barge-in | Runs continuously alongside STT for voice activity | Already loaded. Extend usage from barge-in-only to continuous VAD signal. |
| Generation ID system | Increment on interrupt/new turn | Extend to track monitoring vs response generation | Existing mechanism, new semantics. |
| StreamComposer | Manages TTS audio queue with barge-in | Add Ollama-generated TTS segments alongside Claude-generated | Same API, new content source. |

**Confidence: HIGH** -- All modifications use existing, tested code and libraries.

## VRAM Budget Analysis

**Hardware:** NVIDIA RTX 3070, 8192 MiB total, ~4259 MiB free (current baseline with desktop overhead)

| Model | Precision | Estimated VRAM | Notes |
|-------|-----------|----------------|-------|
| Whisper large-v3 | int8_float16 | ~3.0-3.5 GB | Current production model. CTranslate2 int8 quantization. Based on benchmarks showing large-v2 int8 at 3.1GB. |
| Llama 3.2 3B | Q4_K_M (Ollama default) | ~2.0-2.5 GB | 2.0GB model file + KV cache + CUDA overhead (~0.5GB). |
| Silero VAD | ONNX | ~50 MB | Negligible. |
| **Total** | | **~5.0-6.0 GB** | Fits in 8GB with 2-3GB headroom for CUDA runtime + desktop. |

**Risk: MEDIUM** -- The 8GB budget is workable but not generous. If both models are loaded simultaneously with large KV caches, VRAM pressure could cause Ollama to partially offload to CPU (slower inference) or CTranslate2 to OOM. Mitigation strategies:

1. **Primary plan:** Both models on GPU. Ollama's `OLLAMA_GPU_OVERHEAD` env var reserves VRAM for other CUDA applications (set to ~4GB to account for Whisper).
2. **Fallback A:** Switch Whisper to `distil-large-v3` -- 51% smaller (756M vs 1550M params), 6.3x faster, within 1% WER accuracy. Reduces Whisper VRAM to ~1.5-2GB.
3. **Fallback B:** Switch Whisper to CPU with `int8` compute type. Slower (~2-3x real-time instead of ~10x) but frees all GPU for Ollama.
4. **Fallback C:** Use Whisper "small" model (244M params, ~1GB VRAM with int8). PROJECT.md mentions "small" but codebase uses "large-v3". The small model is sufficient for always-on monitoring where accuracy is less critical than PTT mode.

**Confidence: MEDIUM** -- VRAM numbers are estimated from multiple sources. Exact concurrent usage needs empirical testing on the actual hardware.

## Alternatives Considered

### For Local LLM

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Ollama + Llama 3.2 3B | llama.cpp directly | Ollama wraps llama.cpp with model management, HTTP API, GPU scheduling, and keep_alive. Raw llama.cpp requires manual model loading, no HTTP server, and custom Python bindings. Ollama is the right abstraction for an application that needs an always-available LLM endpoint. |
| Ollama + Llama 3.2 3B | Haiku (Anthropic API) | PROJECT.md explicitly chose Ollama: "Free, local, ~200ms, fits local-first philosophy. Haiku comparable but costs money and needs network." |
| Ollama + Llama 3.2 3B | Ollama + Phi-4 Mini | Phi-4 Mini (3.8B) is slightly larger and would use more VRAM. Llama 3.2 3B is proven for instruction following and fits the VRAM budget better. Could be a future upgrade if VRAM allows. |
| Ollama + Llama 3.2 3B | Ollama + Qwen 3 0.6B | Too small for reliable monitoring decisions. The 3B parameter size is the sweet spot for understanding conversation context and making respond/don't-respond judgments. |
| Ollama + Llama 3.2 3B | vLLM | Production inference server, overkill for single-user desktop app. Heavier resource usage, more complex setup. Ollama is designed for exactly this use case. |

### For Ollama Python Integration

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `ollama` official library (0.6.1) | Raw `httpx` calls to Ollama REST API | The official library adds Pydantic response types, error handling, and AsyncClient with streaming iterators. Since both its dependencies (httpx, pydantic) are already installed, there is no cost to using it. Raw httpx would mean reimplementing response parsing, streaming iteration, and error handling. |
| `ollama` official library (0.6.1) | LangChain `ChatOllama` | Massive dependency tree (langchain-core, langchain-community). Wrong abstraction -- this is a direct LLM call, not a chain/agent. |
| `ollama` official library (0.6.1) | OpenAI-compatible API (`openai` library) | Ollama exposes OpenAI-compatible endpoints, but the native Ollama library has better feature coverage (structured output via `format`, `keep_alive`, model management). The openai library is already installed but is used for TTS, not LLM. |

### For Name-Based Interruption

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Whisper transcript string matching | openWakeWord (dedicated wake word library) | PROJECT.md explicitly states "Wake word detection hardware -- using software-based name recognition instead." The always-on Whisper STT already produces continuous transcripts. Checking `if "hey russel" in transcript.lower()` is trivial, requires zero new dependencies, and leverages the existing STT pipeline. openWakeWord would require a separate 16kHz audio stream, a separate ONNX model (~200KB but still another model to load), and training a custom "hey Russel" model. The STT approach is simpler, more maintainable, and "free" since we're already transcribing everything. |
| Whisper transcript string matching | Picovoice Porcupine | Commercial license required for custom wake words. Cloud dependency for training. Contradicts local-first philosophy. |
| Whisper transcript string matching | Simple energy-based keyword spotting | Not robust enough. "Hey Russel" needs speech recognition, not just audio energy. |

**Note on openWakeWord for future consideration:** If Whisper-based name detection proves too slow (Whisper transcription takes 200-500ms vs openWakeWord's real-time ~80ms latency), openWakeWord could be added later as a fast pre-filter. It runs on CPU with negligible resources. But start with the simpler approach first.

### For Continuous Audio Capture

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `pasimple` (existing, PulseAudio Simple API) | `sounddevice` (PortAudio wrapper) | pasimple is already integrated and working. It provides simple blocking reads in a daemon thread, which is the correct pattern for continuous capture. sounddevice adds PortAudio dependency and callback-based API that doesn't match the existing queue-based architecture. |
| `pasimple` (existing) | `pipewire_python` | Direct PipeWire bindings. The system runs PipeWire with PulseAudio compatibility layer, so pasimple works through PipeWire already. No benefit to switching to native PipeWire API for simple mic capture. |
| `pasimple` (existing) | PyAudio (already installed) | PyAudio is used for playback, not capture. pasimple is used for capture. This separation works well -- no reason to change it. |

## What NOT to Add

| Avoid | Why | Impact if Added |
|-------|-----|-----------------|
| openWakeWord | Whisper transcript already provides name detection. Adding a separate audio processing pipeline for wake word detection adds complexity without clear benefit when STT is always running. | +onnxruntime model, separate audio stream, training pipeline |
| LangChain / LlamaIndex | Direct Ollama API calls are sufficient. The monitoring layer is a simple prompt-in/decision-out loop, not a multi-step chain or RAG pipeline. | Massive dependency tree, wrong abstraction, slower |
| A second Whisper model | One Whisper instance handles all STT. Don't load a separate small model for "fast pre-screening" alongside the large model -- that doubles VRAM. | +1-3GB VRAM, doubled audio processing |
| Redis / message queue | asyncio.Queue is sufficient for inter-stage communication within a single process. The monitoring layer and response backends are coroutines in the same event loop. | External service dependency, operational complexity |
| WebSocket server | Ollama uses HTTP (REST), not WebSockets. The existing EventBus JSONL handles inter-process communication. No need for a WebSocket layer. | Unnecessary complexity |
| pydantic (for structured output schemas) | Already installed (2.12.5). Mentioned here because it might seem like a new dependency -- it is not. | N/A (already present) |

## Installation

```bash
# 1. Install Ollama server (system-level, one-time)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the monitoring model (2.0GB download, one-time)
ollama pull llama3.2:3b

# 3. Configure Ollama for always-on with GPU overhead reservation
sudo systemctl edit ollama.service
# Add:
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=-1"
# Environment="OLLAMA_GPU_OVERHEAD=4000000000"

sudo systemctl daemon-reload && sudo systemctl restart ollama

# 4. Install Python client in PTT venv
source ~/.local/share/push-to-talk/venv/bin/activate
pip install ollama==0.6.1

# 5. Verify
python3 -c "
import asyncio
from ollama import AsyncClient
async def test():
    r = await AsyncClient().chat(model='llama3.2:3b', messages=[{'role':'user','content':'Say hi'}])
    print(r.message.content)
asyncio.run(test())
"
```

Add to `requirements.txt`:
```
# Local LLM monitoring (Ollama client)
ollama>=0.6
```

## Integration Points with Existing Stack

### 1. Audio Capture Stage (live_session.py `_audio_capture_stage`)

**Current:** Audio frames flow to `_audio_in_q` only when PTT key is held (or in live mode with idle timeout).

**v2.0:** Audio frames flow continuously. The capture loop removes all PTT/mute gating. Frames are always pushed to the queue. A separate "monitoring STT" consumer processes them independently from the "response LLM" stage.

```python
# Current pattern (preserve the pasimple threading):
def record_thread():
    with pasimple.PaSimple(pasimple.PA_STREAM_RECORD, ...) as pa:
        while not stop_event.is_set():
            data = pa.read(CHUNK_SIZE)
            loop.call_soon_threadsafe(_enqueue_audio, PipelineFrame(...))

# v2.0 change: remove mute/gate checks from the consumer side
# Audio capture itself is already always-on in the daemon thread
```

### 2. STT Stage (live_session.py `_stt_stage`)

**Current:** Accumulates audio, detects silence, transcribes, emits TRANSCRIPT frames to `_stt_out_q`. STT is gated during AI playback.

**v2.0:** STT runs continuously. Transcripts feed TWO consumers:
1. **Monitoring queue** (`_monitor_q`): Every transcript goes here for the Ollama observer.
2. **Response queue** (`_stt_out_q`): Only when the observer decides to respond, the relevant transcript is forwarded.

```python
# New: dual-output from STT stage
if transcript and not _is_hallucination(transcript):
    # Always feed the monitor
    await self._monitor_q.put(transcript)

    # Check for name-based interrupt
    if self._check_name_interrupt(transcript):
        await self._trigger_barge_in()
```

### 3. New: Monitor Stage (new coroutine)

A new pipeline stage that consumes transcripts from `_monitor_q` and calls Ollama:

```python
async def _monitor_stage(self):
    """Consume continuous transcripts, decide when AI should respond."""
    transcript_window = []  # Rolling window of recent transcripts

    while self.running:
        transcript = await self._monitor_q.get()
        transcript_window.append(transcript)

        # Trim window to last N seconds / entries
        transcript_window = transcript_window[-10:]

        # Ask Ollama: should we respond?
        decision = await self._ollama_should_respond(
            '\n'.join(transcript_window)
        )

        if decision['should_respond']:
            if decision['complexity'] == 'quick':
                # Ollama generates the response directly
                await self._quick_respond_ollama(decision)
            else:
                # Forward to Claude CLI for deep response
                await self._stt_out_q.put(PipelineFrame(
                    type=FrameType.TRANSCRIPT,
                    generation_id=self.generation_id,
                    data='\n'.join(transcript_window)
                ))
```

### 4. Response Backend Selection (new logic)

**Current:** All responses go through Claude CLI.

**v2.0:** Configurable backend based on conditions:

```python
async def _select_backend(self, complexity: str) -> str:
    """Choose response backend based on conditions."""
    if complexity == 'quick':
        return 'ollama'  # Fast local response

    # Check network availability for Claude
    if not await self._check_network():
        return 'ollama'  # Fallback when offline

    return 'claude'  # Deep response via CLI
```

### 5. Name-Based Interruption (extend existing barge-in)

**Current:** VAD detects sustained speech during playback, triggers barge-in.

**v2.0:** Additionally check transcript content for "hey russel":

```python
def _check_name_interrupt(self, transcript: str) -> bool:
    """Check if user said the AI's name to interrupt."""
    lower = transcript.lower().strip()
    triggers = ['hey russel', 'hey russell', 'russel', 'russell']
    return any(trigger in lower for trigger in triggers)
```

### 6. Pipeline Frames (pipeline_frames.py)

New frame types for the monitoring layer:

```python
class FrameType(Enum):
    # ... existing types ...
    MONITOR_DECISION = auto()  # Ollama decided to respond
    OLLAMA_RESPONSE = auto()   # Response text from Ollama backend
```

### 7. EventBus (event_bus.py)

New event types:

```python
class EventType(str, Enum):
    # ... existing types ...
    MONITOR_TRANSCRIPT = "monitor_transcript"   # Transcript sent to monitor
    MONITOR_DECISION = "monitor_decision"       # Ollama's respond/ignore decision
    BACKEND_SELECTED = "backend_selected"       # claude vs ollama for response
    OLLAMA_RESPONSE = "ollama_response"         # Response from Ollama backend
```

## Performance Budget

| Operation | Target | Estimated | Notes |
|-----------|--------|-----------|-------|
| Ollama monitoring call (non-streaming) | <500ms | ~200ms | Llama 3.2 3B with `num_predict: 100`, `temperature: 0`, structured JSON output. Model kept loaded via `keep_alive: -1`. |
| Ollama quick response (streaming) | <1s TTFT | ~200ms | First token in ~200ms, full response streams over 1-2s. |
| Whisper continuous transcription | <500ms per segment | ~200-400ms | Same as current. Runs in executor thread. 10x real-time factor on GPU. |
| Name detection in transcript | <1ms | <0.1ms | Simple string `in` check. |
| Backend selection logic | <5ms | <1ms | Simple condition checks. |
| End-to-end: user speaks to AI starts responding | <2s | ~1-1.5s | silence detection (0.8s) + STT (0.3s) + monitoring (0.2s) + backend selection (<0.01s) + TTS start (0.2s) |

## Version Compatibility

| Package | Version | Compatible With | Notes |
|---------|---------|-----------------|-------|
| `ollama` | 0.6.1 | Python 3.8+, httpx >=0.27, pydantic >=2.9 | All requirements met in current venv. |
| `httpx` | 0.28.1 (installed) | ollama 0.6.1 | Already present, no upgrade needed. |
| `pydantic` | 2.12.5 (installed) | ollama 0.6.1 | Already present, no upgrade needed. |
| Ollama server | latest | RTX 3070 (CUDA), Llama 3.2 3B | NVIDIA GPU support is Ollama's primary platform. |
| `faster-whisper` | 1.2.1 (installed) | CTranslate2 4.7.1, CUDA 12 | No change needed. |
| `pasimple` | installed | PipeWire compat layer | No change needed. |

## Ollama API Quick Reference

For roadmap authors -- key Ollama API details for phase planning:

**Structured output (critical for monitoring):**
```python
# Pass JSON schema to format parameter
response = await AsyncClient().chat(
    model='llama3.2:3b',
    messages=[...],
    format={'type': 'object', 'properties': {...}, 'required': [...]},
    options={'temperature': 0}  # Deterministic for classification
)
result = json.loads(response.message.content)  # Guaranteed valid JSON
```

**Streaming (for quick responses):**
```python
async for part in await AsyncClient().chat(
    model='llama3.2:3b', messages=[...], stream=True
):
    text_chunk = part['message']['content']
```

**Error handling:**
```python
import ollama
try:
    response = await AsyncClient().chat(...)
except ollama.ResponseError as e:
    if e.status_code == 404:
        # Model not loaded -- pull it
        await AsyncClient().pull('llama3.2:3b')
```

**Model management:**
```python
# List loaded models
models = await AsyncClient().list()

# Check if model is loaded
ps = await AsyncClient().ps()  # Running models with VRAM usage
```

## Data Footprint

| Component | Size | Notes |
|-----------|------|-------|
| Ollama server | ~500MB installed | System-level install |
| Llama 3.2 3B model | 2.0GB on disk | Stored in `~/.ollama/models/` |
| `ollama` Python package | ~100KB | Thin wrapper around httpx |
| **Total new disk usage** | ~2.5GB | Dominated by the LLM model file |

## Sources

- [Ollama Python library GitHub](https://github.com/ollama/ollama-python) -- AsyncClient API, streaming, structured output (HIGH confidence)
- [Ollama PyPI](https://pypi.org/project/ollama/) -- v0.6.1, dependencies: httpx + pydantic (HIGH confidence)
- [Ollama /api/chat docs](https://docs.ollama.com/api/chat) -- format, options, keep_alive, streaming parameters (HIGH confidence)
- [Ollama structured outputs docs](https://docs.ollama.com/capabilities/structured-outputs) -- JSON schema via format parameter (HIGH confidence)
- [Ollama FAQ](https://docs.ollama.com/faq) -- keep_alive, OLLAMA_MAX_LOADED_MODELS, GPU memory management (HIGH confidence)
- [Llama 3.2 model page](https://ollama.com/library/llama3.2) -- 3B/1B sizes, 2.0GB/1.3GB, 128K context (HIGH confidence)
- [Ollama VRAM guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) -- 3B model ~2-3GB VRAM with Q4 (MEDIUM confidence, community source)
- [Whisper memory requirements](https://github.com/openai/whisper/discussions/5) -- Model size vs VRAM table (HIGH confidence, official repo)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper) -- int8 quantization, distil-large-v3 support (HIGH confidence)
- [openWakeWord GitHub](https://github.com/dscripka/openWakeWord) -- Evaluated and rejected for initial implementation (HIGH confidence)
- [distil-whisper HuggingFace](https://huggingface.co/distil-whisper/distil-large-v3) -- 6.3x faster, 51% smaller, within 1% WER (HIGH confidence)
- Existing codebase: `live_session.py`, `pipeline_frames.py`, `event_bus.py`, `stream_composer.py`, `requirements.txt` -- Direct code inspection (HIGH confidence)
- System inspection: `nvidia-smi`, `pip show`, venv package versions -- Direct verification (HIGH confidence)

---
*Stack research for: v2.0 Always-On Observer*
*Researched: 2026-02-21*
