# Push-to-Talk

## What This Is

A Linux desktop tool providing global hotkey-driven voice input with multiple AI modes. The flagship "live" mode runs a real-time voice conversation with Claude via a 5-stage asyncio pipeline (audio capture → Whisper STT → Claude CLI → Piper TTS → playback). It can spawn and manage background Claude CLI tasks, learn from conversations, and display status through a floating overlay. Other modes include dictation, interview, and conversation with full tool access.

## Core Value

Natural, low-friction voice conversation with Claude that feels like talking to a person — fast recognition, intelligent responses, no jarring artifacts.

## Completed Milestone: v1.1 Voice UX Polish ✓

**Goal:** Make the live voice session feel natural — fix filler conflicts, reduce false STT triggers, suppress stale tool narration, enable barge-in interruption.

**Status:** All 15 requirements satisfied, 4/4 phases passed. Audit passed 2026-02-18.

## Next Milestone: v1.2 Adaptive Quick Responses

**Goal:** Replace the dumb filler system with an AI-driven quick response library that learns what to say based on context — making the assistant feel socially aware, not just technically capable.

**Core idea:** The AI builds and maintains a library of situational quick responses (situation → pre-generated audio clip). When input arrives, the system instantly matches it to the right response and plays it while the full LLM response processes in the background.

**Target features:**
- Quick response library: growing cache of (situation → audio clip) pairs
- AI-chosen responses: the AI decides what phrases to use, not random selection
- Context-aware matching: classify input instantly and match to appropriate response
- Non-speech awareness: cough → "excuse you", sigh → empathetic response, etc.
- Library growth: after sessions, identify uncovered situations and generate new phrases
- Library pruning: phase out phrases that feel unnatural over time
- Always processing: no perceptible gate/delay, background pipeline feels seamless

## Requirements

### Validated

- ✓ Mode rename ("live" dictation → "dictate") — v1.0
- ✓ Live voice session with 5-stage pipeline — v1.0
- ✓ Claude CLI integration via stream-json — v1.0
- ✓ MCP tool server for task management — v1.0
- ✓ TaskManager for async background tasks — v1.0
- ✓ Filler system (acknowledgment clip factory with quality gating) — v1.0→v1.1
- ✓ Learner daemon for persistent memory — v1.0
- ✓ Live overlay with status, drag, model selection — v1.0
- ✓ Conversation logging (JSONL) — v1.0
- ✓ Barge-in interruption with VAD, context annotation, shortened silence — v1.1
- ✓ Multi-layer STT filtering (no_speech_prob, logprob, compression ratio) — v1.1
- ✓ Tool-use speech suppression (only final response spoken) — v1.1
- ✓ Dynamic overlay (tool intent labels, history coalescing, STT rejection flash) — v1.1

### Active

- [ ] Quick response library with situation → audio clip mapping
- [ ] AI-driven phrase selection based on input context
- [ ] Non-speech event awareness (coughs, sighs, laughter → appropriate responses)
- [ ] Background library growth and pruning across sessions
- [ ] Seamless background processing — no perceptible filler gate

### Out of Scope

- OpenAI TTS (sticking with local Piper for now) — latency vs quality tradeoff, revisit later
- Always-on listening without PTT — privacy, CPU, false positives
- Visual task dashboard — voice-first tool
- Persistent tasks across sessions — each session starts fresh

## Context

The live mode evolved from the original OpenAI Realtime API plan into a Claude CLI-based pipeline using local Whisper for STT and Piper for TTS. v1.1 added barge-in (VAD-based, STT gating instead of mic mute), multi-layer STT filtering, tool-use speech suppression, and acknowledgment phrase fillers. The current filler system picks random clips from a pool — it doesn't understand context, so task-oriented phrases ("let me take a look") play even for simple conversational questions ("what's your name?"). v1.2 replaces this with an AI-driven quick response system that understands what's happening and responds appropriately, including to non-speech events like coughs.

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
| Acknowledgment phrase fillers | Non-verbal (hums) failed with Piper TTS; verbal acknowledgments work well | ✓ Good |
| Filler clip factory subprocess | Generate varied clips, rotate pool, evaluate naturalness | ✓ Good |
| Gate STT for barge-in instead of mic mute | Mic must stay live for VAD to detect speech during playback | ✓ Good |
| AI-driven quick response library | Dumb random fillers sound wrong in context; AI should choose what to say | — v1.2 |

---
*Last updated: 2026-02-18 after v1.1 milestone*
