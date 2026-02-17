# Push-to-Talk

## What This Is

A Linux desktop tool providing global hotkey-driven voice input with multiple AI modes. The flagship "live" mode runs a real-time voice conversation with Claude via a 5-stage asyncio pipeline (audio capture → Whisper STT → Claude CLI → Piper TTS → playback). It can spawn and manage background Claude CLI tasks, learn from conversations, and display status through a floating overlay. Other modes include dictation, interview, and conversation with full tool access.

## Core Value

Natural, low-friction voice conversation with Claude that feels like talking to a person — fast recognition, intelligent responses, no jarring artifacts.

## Current Milestone: v1.1 Voice UX Polish

**Goal:** Make the live voice session feel natural — fix filler conflicts, reduce false STT triggers, suppress stale tool narration, enable barge-in interruption.

**Target features:**
- Filler system overhaul: non-verbal clips only + clip factory for generation/rotation
- STT reliability: Whisper no_speech_prob filtering to reject non-speech sounds
- Tool-use speech flow: only speak the final coherent response after all tool calls
- Barge-in: let user interrupt AI mid-speech
- Overlay polish: granular status states, expandable history

## Requirements

### Validated

- ✓ Mode rename ("live" dictation → "dictate") — v1.0 Phase 1
- ✓ Live voice session with 5-stage pipeline — v1.0 Phase 1
- ✓ Claude CLI integration via stream-json — v1.0 Phase 1
- ✓ MCP tool server for task management — v1.0 Phase 3
- ✓ TaskManager for async background tasks — v1.0 Phase 2
- ✓ Filler system (canned clips + smart generation) — v1.0 Phase 1
- ✓ Learner daemon for persistent memory — v1.0 Phase 3
- ✓ Live overlay with status, drag, model selection — v1.0 Phase 1
- ✓ Conversation logging (JSONL) — v1.0 Phase 3
- ✓ Granular status indicators (thinking, tool_use) — v1.1 pre-work
- ✓ Expandable status history panel — v1.1 pre-work
- ✓ Tool-use text suppression (post_tool_buffer) — v1.1 pre-work
- ✓ Whisper no_speech_prob segment filtering — v1.1 pre-work

### Active

- [ ] Non-verbal filler clips only (remove Ollama smart filler)
- [ ] Filler clip factory: subprocess generates/rotates clips, capped pool, natural evaluation
- [ ] STT false trigger tuning and verification
- [ ] Tool-use speech flow end-to-end verification
- [ ] Barge-in: user can interrupt AI mid-speech via voice activity detection
- [ ] Overlay and status polish: verify all states render correctly

### Out of Scope

- OpenAI TTS (sticking with local Piper for now) — latency vs quality tradeoff, revisit later
- Always-on listening without PTT — privacy, CPU, false positives
- Visual task dashboard — voice-first tool
- Persistent tasks across sessions — each session starts fresh

## Context

The live mode evolved from the original OpenAI Realtime API plan into a Claude CLI-based pipeline using local Whisper for STT and Piper for TTS. The Silero VAD model is already in the codebase (loaded by `_load_vad_model`) but only used for barge-in monitoring which is currently non-functional (mic is muted during playback via pactl). Barge-in requires gating STT instead of muting the physical mic source.

## Constraints

- **Platform**: Linux X11 with PipeWire — no cross-platform concerns
- **Architecture**: 5-stage asyncio pipeline in `live_session.py`, GTK overlay in `indicator.py`
- **TTS**: Piper local (22050Hz → 24000Hz resampling), fillers also via Piper
- **STT**: Local Whisper "small" model, blocking transcription in executor

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Claude CLI pipeline instead of OpenAI Realtime | Direct Claude access, no API key dependency for LLM, lower cost | ✓ Good |
| Local Whisper + Piper instead of cloud STT/TTS | Zero latency dependency on cloud, works offline | ✓ Good |
| Non-verbal fillers only | Semantic fillers (smart filler, "okay", "sure") conflict with LLM response | — Pending |
| Filler clip factory subprocess | Generate varied non-verbal clips, rotate pool, evaluate naturalness | — Pending |
| Gate STT for barge-in instead of mic mute | Mic must stay live for VAD to detect speech during playback | — Pending |

---
*Last updated: 2026-02-17 after v1.1 milestone start*
