# Phase 6: Polish & Verification - Research

**Researched:** 2026-02-18
**Domain:** STT filtering, tool-use speech flow, overlay states, pre-tool acknowledgment clips
**Confidence:** HIGH (codebase-driven, verified against official docs)

## Summary

This phase verifies and tunes existing pre-work features (STT filtering, tool-use speech suppression, overlay states) and adds new capabilities (pre-tool acknowledgment clips, dynamic tool-use overlay, STT false trigger rejection indicator). The research examines four domains: (1) Whisper `no_speech_prob` threshold tuning and complementary filtering strategies, (2) Claude CLI stream-json tool_use event structure for extracting tool intent summaries, (3) pre-recorded acknowledgment clip generation and playback architecture, and (4) overlay status communication for richer tool-use state.

Most requirements are already coded (STT-01, FLOW-01, FLOW-02, OVL-01, OVL-02 marked "committed"). The work centers on tuning thresholds, adding acknowledgment clips, enriching the overlay's tool_use state with AI-summarized intent, and adding a visual rejection indicator.

**Primary recommendation:** Structure into 3 plans: (1) STT tuning + rejection indicator, (2) acknowledgment clip factory + pre-tool playback + long-chain fillers, (3) dynamic tool-use overlay + history enrichment + end-to-end verification.

## Standard Stack

No new libraries needed. This phase works entirely within the existing stack.

### Core (Already Installed)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| whisper | "small" model | Local STT with `no_speech_prob` filtering | In use |
| Silero VAD | ONNX model | Voice activity detection for barge-in | In use |
| Piper TTS | en_US-lessac-medium | TTS + clip generation | In use |
| PyAudio | - | Audio playback | In use |
| GTK3 (Cairo) | - | Overlay widget rendering | In use |
| Claude CLI | stream-json | LLM processing | In use |

### No New Dependencies
This phase is entirely about tuning existing code and adding features within the current architecture. No pip installs needed.

## Architecture Patterns

### Current Status Communication Pipeline
```
LiveSession._set_status(status_string)
  -> on_status callback
    -> set_status() in push-to-talk.py
      -> STATUS_FILE.write_text(status)
        -> indicator.py polls STATUS_FILE every 100ms via GLib.timeout_add
          -> LiveOverlayWidget.update_status(status)
```

The status is a single string written to a file. For richer tool-use state (AI-summarized intent), this mechanism needs extension.

### Pattern 1: Extending Status with Metadata via JSON File
**What:** Write a companion JSON file alongside the status file to carry tool intent data.
**When to use:** When the overlay needs more than just a status string.
**Example:**
```python
# In live_session.py, when tool_use detected:
def _set_status(self, status, metadata=None):
    self.on_status(status)
    if metadata:
        meta_path = Path(__file__).parent / "status_meta"
        meta_path.write_text(json.dumps(metadata))

# Usage:
self._set_status("tool_use", {"intent": "Checking the pipeline code"})
```

```python
# In indicator.py LiveOverlayWidget.update_status:
def update_status(self, status, metadata=None):
    if status == 'tool_use' and metadata and 'intent' in metadata:
        self.tool_intent = metadata['intent']
    # ... existing logic ...
```

### Pattern 2: Extracting Tool Intent from stream-json Events
**What:** The Claude CLI `content_block_start` event for tool_use contains the tool name in `content_block.name`. Combined with accumulated `input_json_delta` partials, we can infer intent.
**When to use:** Every time a tool_use content block starts.

The stream-json event structure for tool_use (verified from official Anthropic docs):
```json
{
  "type": "content_block_start",
  "index": 1,
  "content_block": {
    "type": "tool_use",
    "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
    "name": "get_weather",
    "input": {}
  }
}
```

**Key insight:** The `name` field is available immediately at `content_block_start`. The `input` arrives incrementally via `input_json_delta` events. For intent summarization, we only need the tool name -- we can map MCP tool names to human-readable descriptions.

```python
# Tool name -> human-readable intent mapping
TOOL_INTENT_MAP = {
    "spawn_task": "Starting a task",
    "list_tasks": "Checking tasks",
    "get_task_status": "Checking task progress",
    "get_task_result": "Getting task results",
    "cancel_task": "Cancelling a task",
    "send_notification": "Sending a notification",
}

# In _read_cli_response, at content_block_start for tool_use:
tool_name = content_block.get("name", "")
intent = TOOL_INTENT_MAP.get(tool_name, f"Using {tool_name}")
self._set_status("tool_use", {"intent": intent})
```

### Pattern 3: Acknowledgment Clip Pool (Separate from Nonverbal Fillers)
**What:** Pre-generated WAV clips of short verbal acknowledgments ("let me check", "one sec") stored in a separate directory from nonverbal fillers.
**When to use:** Before tool calls that take noticeable time (>300ms gate).

```
audio/fillers/
  nonverbal/          # Existing: hums, breaths (for general waiting)
  acknowledgment/     # NEW: verbal pre-tool phrases
```

The clip factory already has the infrastructure for generation and quality evaluation. A new `acknowledgment` category reuses the same factory pattern with different prompts.

### Pattern 4: STT Rejection Visual Indicator
**What:** Brief visual feedback when audio is rejected by no_speech_prob filter.
**When to use:** When STT rejects non-speech audio, flash the overlay briefly.

```python
# In _whisper_transcribe, after rejection:
self._set_status("stt_rejected")
# Overlay shows brief dim/flash, then returns to "listening"
```

### Anti-Patterns to Avoid
- **Over-filtering STT:** Setting `no_speech_prob` threshold too low (e.g., 0.3) will reject legitimate quiet speech. The user explicitly wants balanced sensitivity.
- **Generating acknowledgment clips live:** TTS takes 100-300ms per clip. Pre-generate and load at startup for instant playback.
- **Using tool input JSON for intent:** Parsing partial JSON from `input_json_delta` is complex and unnecessary when the tool `name` alone provides sufficient intent.
- **Replacing the status file with a socket:** The polling mechanism is simple and works. Adding a socket for overlay communication adds complexity for minimal benefit.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio quality evaluation | Custom signal analysis | Existing `evaluate_clip()` in clip_factory.py | Already tuned with proper thresholds (RMS, clipping, silence) |
| Clip generation | New TTS system | Existing Piper TTS + clip_factory.py pattern | Same voice model, proven generation pipeline |
| Non-speech detection | Custom VAD/energy analysis | Whisper's `no_speech_prob` + existing RMS/hallucination filters | Already in place, just needs threshold tuning |
| Tool intent summarization | LLM-based summarization | Static tool name -> description mapping | MCP tools are known at compile time, no AI needed |
| Inter-process status | WebSocket/pipe | Existing file-based polling (STATUS_FILE) | Works reliably at 100ms poll interval, no new IPC complexity |

## Common Pitfalls

### Pitfall 1: Whisper no_speech_prob Unreliability for Non-Speech Sounds
**What goes wrong:** `no_speech_prob` is not a reliable VAD. Throat clearing and coughs are NOT silence -- they have acoustic energy that Whisper interprets as having speech content. The `no_speech_prob` for a cough can be LOW (0.2-0.4), meaning the cough passes the filter.
**Why it happens:** Whisper's `no_speech_prob` is trained to detect silence, not to distinguish speech from non-speech sounds. Coughs and throat clears have spectral characteristics that overlap with speech.
**How to avoid:** Use a multi-layer filtering approach:
1. `no_speech_prob` threshold (currently 0.6, tune to 0.5-0.7 range)
2. `logprob_threshold` (add as supplementary filter -- currently not used)
3. Existing hallucination phrase list (already catches "hmm", "uh", etc.)
4. Existing RMS energy gate (`SPEECH_ENERGY_MIN = 25`, `SPEECH_CHUNKS_MIN = 5`)
5. Consider adding `compression_ratio_threshold` check for repetitive hallucinations
**Warning signs:** Cough transcripts appearing as "Thank you" or single nonsense words.
**Confidence:** HIGH -- verified against Whisper source code and community findings.

### Pitfall 2: Acknowledgment Clips Playing for Fast Tool Calls
**What goes wrong:** If the tool call completes in under 200ms, the acknowledgment clip starts playing but the response is already being spoken, causing audio overlap.
**Why it happens:** No gate/delay before playing acknowledgment.
**How to avoid:** Reuse the existing filler gate pattern: wait 300ms (configurable) before playing. If the tool call completes within that window, skip the acknowledgment entirely. This is exactly how `_filler_manager` works today.
**Warning signs:** "One sec" playing simultaneously with the AI's actual response.

### Pitfall 3: Status File Race Condition for Tool Intent
**What goes wrong:** Writing JSON metadata to a companion file creates a TOCTOU race: overlay reads status "tool_use" but the metadata file hasn't been written yet, or reads stale metadata from a previous tool call.
**Why it happens:** Two separate file writes are not atomic.
**How to avoid:** Write the status and metadata as a single JSON payload to the status file. The overlay parser checks if the content starts with `{` (JSON) vs plain text (simple status).

```python
# In set_status():
if metadata:
    STATUS_FILE.write_text(json.dumps({"status": status, **metadata}))
else:
    STATUS_FILE.write_text(status)

# In overlay check_status():
content = STATUS_FILE.read_text().strip()
if content.startswith('{'):
    data = json.loads(content)
    status = data['status']
    metadata = data
else:
    status = content
    metadata = None
```

### Pitfall 4: Overlay History Getting Too Long with Tool Intents
**What goes wrong:** If every tool call adds a rich history entry like "Checking pipeline code", the history panel fills up with tool entries, pushing out useful state transitions.
**Why it happens:** Multiple tool calls in a chain each add an entry.
**How to avoid:** Cap history at 10 visible entries (already HISTORY_MAX_VISIBLE = 10, storage capped at 50). For consecutive tool_use entries, update the latest entry rather than appending a new one -- show the progression as a single evolving entry.

### Pitfall 5: Whisper Hallucinations on Post-Barge-in Audio
**What goes wrong:** After barge-in, the shortened silence threshold (0.4s vs 0.8s) makes Whisper more likely to trigger on residual audio artifacts from the interrupted playback, producing hallucinated text.
**Why it happens:** The mic is live during playback (for VAD). After barge-in, there may be echo/reverb from the speaker that gets picked up.
**How to avoid:** The existing `has_speech` flag and `SPEECH_CHUNKS_MIN = 5` requirement already provide protection. Verify during testing that post-barge-in transcriptions are clean. If not, consider a brief (100ms) post-barge-in dead zone.

## Code Examples

### Example 1: Enhanced Whisper Transcription with Multi-Layer Filtering
```python
# Source: Current _whisper_transcribe method + research findings
def _whisper_transcribe(self, pcm_data: bytes) -> str | None:
    # ... existing temp file creation ...

    result = self.whisper_model.transcribe(
        tmp_path, language="en",
        condition_on_previous_text=False,
    )

    segments = result.get("segments", [])
    if segments:
        kept = []
        for s in segments:
            no_speech = s.get("no_speech_prob", 0)
            avg_logprob = s.get("avg_logprob", 0)
            compression = s.get("compression_ratio", 0)
            text = s.get("text", "").strip()

            # Layer 1: no_speech_prob filter (existing, tunable)
            if no_speech >= 0.6:  # Tune this value
                print(f"STT: Rejected (no_speech={no_speech:.2f}): \"{text[:40]}\"", flush=True)
                continue

            # Layer 2: logprob confidence filter (NEW)
            if avg_logprob < -1.0:  # Low confidence
                print(f"STT: Rejected (logprob={avg_logprob:.2f}): \"{text[:40]}\"", flush=True)
                continue

            # Layer 3: compression ratio filter (NEW, catches repetitive hallucinations)
            if compression > 2.4:
                print(f"STT: Rejected (compression={compression:.2f}): \"{text[:40]}\"", flush=True)
                continue

            kept.append(text)

        text = "".join(kept).strip()
        if not kept and segments:
            # Signal rejection to overlay
            self._set_status("stt_rejected")
    else:
        text = result.get("text", "").strip()

    os.unlink(tmp_path)
    return text if text else None
```

### Example 2: Tool Intent Extraction from stream-json
```python
# Source: Anthropic streaming docs + current _read_cli_response
# In _read_cli_response, within the content_block_start handler:

if inner_type == "content_block_start":
    content_block = event.get("content_block", {})
    if content_block.get("type") == "tool_use":
        tool_name = content_block.get("name", "unknown")
        intent = TOOL_INTENT_MAP.get(tool_name, f"Using {tool_name}")
        self._set_status("tool_use", {"intent": intent})
        # ... existing tool_use handling ...
```

### Example 3: Acknowledgment Clip Factory Extension
```python
# New prompts for acknowledgment clips (extend clip_factory.py)
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

# Store in audio/fillers/acknowledgment/
ACK_CLIP_DIR = Path(__file__).parent / "audio" / "fillers" / "acknowledgment"
```

### Example 4: Status File with Embedded Metadata
```python
# In push-to-talk.py set_status():
def set_status(status, metadata=None):
    try:
        if metadata:
            import json
            STATUS_FILE.write_text(json.dumps({"status": status, **metadata}))
        else:
            STATUS_FILE.write_text(status)
    except:
        pass

# In indicator.py check_status() — LiveOverlayWidget context:
def check_status(self):
    try:
        if STATUS_FILE.exists():
            content = STATUS_FILE.read_text().strip()
            if content.startswith('{'):
                import json
                data = json.loads(content)
                new_status = data.get('status', 'idle')
                metadata = data
            else:
                new_status = content
                metadata = None
            if new_status != self.status or metadata:
                self.update_status(new_status, metadata)
    except:
        pass
```

### Example 5: Overlay Rejection Flash
```python
# In LiveOverlayWidget — handle brief stt_rejected flash:
def update_status(self, status, metadata=None):
    if status == 'stt_rejected':
        # Brief visual flash — dim the dot, restore after 300ms
        self._flash_rejection()
        return  # Don't update status or history

    if status == 'tool_use' and metadata and 'intent' in metadata:
        self.tool_intent = metadata['intent']
    else:
        self.tool_intent = None

    # ... existing update_status logic ...

def _flash_rejection(self):
    """Brief visual indicator that audio was filtered."""
    self._rejection_flash = True
    self.queue_draw()
    GLib.timeout_add(300, self._clear_rejection_flash)

def _clear_rejection_flash(self):
    self._rejection_flash = False
    self.queue_draw()
    return False  # Don't repeat
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `no_speech_prob` only (threshold 0.6) | Multi-layer: no_speech_prob + logprob + compression_ratio + hallucination list | Research finding | Better non-speech rejection without over-filtering |
| Simple "Using Tool" status | Dynamic tool intent label ("Checking pipeline code") | This phase | Informative overlay during tool use |
| No pre-tool acknowledgment | Pre-recorded acknowledgment clips with gate | This phase | Natural feel during tool-use waits |
| Silent STT rejection | Brief visual flash on rejection | This phase | User knows audio was heard but filtered |

**Not deprecated, still valid:**
- Hallucination phrase list (`HALLUCINATION_PHRASES`) -- still useful as final safety net
- RMS-based speech energy gate -- catches low-energy noise before Whisper
- Filler gate pattern (300ms wait) -- reuse for acknowledgment timing

## Open Questions

1. **Exact no_speech_prob threshold value**
   - What we know: Default 0.6 is in place. Research suggests range 0.5-0.7 for balanced filtering. Community reports suggest 0.6 is reasonable but may need adjustment for specific environments.
   - What's unclear: The optimal value for the user's specific environment (quiet home office with burst noise). Throat clearing and coughs may have no_speech_prob around 0.3-0.5, which means they pass even with 0.6 threshold.
   - Recommendation: Start with 0.6 (current), add logprob and compression_ratio as supplementary filters. Log all rejection data for manual tuning. The user explicitly deferred threshold values to Claude's discretion during tuning.

2. **Acknowledgment clip variety and selection logic**
   - What we know: Pool of 10-15 phrases, mix of conversational ("let me check that") and brief ("one sec"). User wants clip factory to generate them.
   - What's unclear: Whether Piper TTS produces natural-sounding full phrases (vs. the hums/breaths it generates well for nonverbal clips). Longer phrases may sound robotic.
   - Recommendation: Generate clips using same Piper model. Apply same quality evaluation. If quality is poor, consider using OpenAI TTS (which the user has an API key for) as a one-time batch generation. Test during implementation.

3. **How to handle tool intent for non-MCP tools**
   - What we know: Current code disables all built-in tools (`--tools ""`). Only MCP tools are available. The TOOL_INTENT_MAP can be exhaustive.
   - What's unclear: If built-in tools are ever re-enabled, the map won't cover them.
   - Recommendation: Use the static map for known MCP tools. Fall back to `f"Using {tool_name.replace('_', ' ')}"` for unknown tools. This is sufficient.

4. **Long tool chain filler timing**
   - What we know: User wants nonverbal filler clips for 10+ second tool chains. Current filler_manager plays clips at 0.3s and 4.3s intervals.
   - What's unclear: Whether the existing filler manager should be reused during tool use, or if a separate tool-chain filler loop is needed.
   - Recommendation: Reuse existing filler mechanism. After acknowledgment clip plays, if tool use continues for >4s, play nonverbal clips at intervals. The current `_filler_manager` pattern already does this for LLM waits.

## Sources

### Primary (HIGH confidence)
- **Codebase analysis:** `live_session.py` (1917 lines), `indicator.py` (~2230 lines), `clip_factory.py` (325 lines), `pipeline_frames.py` (28 lines)
- **Anthropic Streaming Docs:** https://platform.claude.com/docs/en/build-with-claude/streaming -- Verified tool_use content_block_start includes `name` field
- **Whisper source code:** https://github.com/openai/whisper/blob/main/whisper/transcribe.py -- `no_speech_prob`, `avg_logprob`, `compression_ratio` all available per segment

### Secondary (MEDIUM confidence)
- **Whisper threshold tuning:** GitHub Discussion #679 (https://github.com/openai/whisper/discussions/679) -- Community-verified parameter combinations
- **Whisper non-speech behavior:** GitHub Discussion #29 (https://github.com/openai/whisper/discussions/29) -- Explains unreliability of no_speech_prob for non-silence sounds
- **Whisper Large V3 defaults:** https://huggingface.co/openai/whisper-large-v3 -- Tighter defaults (compression_ratio 1.35)

### Tertiary (LOW confidence)
- **Calm-Whisper (2025):** Research showing attention head calming reduces hallucinations by 84.5% -- interesting but requires model fine-tuning, not applicable here

## Metadata

**Confidence breakdown:**
- STT filtering approach: HIGH -- verified against Whisper source code, existing code already has the mechanism, just needs threshold tuning + supplementary filters
- Tool intent extraction: HIGH -- verified against official Anthropic streaming docs, content_block_start contains tool `name`
- Acknowledgment clips: MEDIUM -- clip factory pattern is proven for nonverbal, but verbal phrase quality via Piper is untested
- Overlay enrichment: HIGH -- codebase analysis confirms file-based status mechanism, JSON extension is straightforward
- Status history enrichment: HIGH -- overlay code is clear, HISTORY_MAX_VISIBLE and storage cap already exist

**Research date:** 2026-02-18
**Valid until:** Stable (60 days) -- all components are local/self-hosted, no API changes expected
