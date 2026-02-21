# Push-to-Talk

## What This Is

A Linux desktop AI voice assistant with always-on listening. The AI continuously monitors ambient audio via local Whisper STT, uses a local LLM (Ollama) to decide when to engage, and responds through configurable backends (Claude CLI for deep work, Ollama for quick responses). It proactively participates in conversations, answers questions, and assists with tasks — like having a knowledgeable colleague in the room. Also supports dictation, interview, and conversation modes.

## Core Value

An always-present AI that listens, understands context, and contributes when it has something useful to add — without requiring explicit activation.

## Completed Milestone: v1.1 Voice UX Polish ✓

**Goal:** Make the live voice session feel natural — fix filler conflicts, reduce false STT triggers, suppress stale tool narration, enable barge-in interruption.

**Status:** All 15 requirements satisfied, 4/4 phases passed. Audit passed 2026-02-18.

## Completed Milestone: v1.2 Adaptive Quick Responses (Partial) ✓

**Goal:** Replace the dumb filler system with an AI-driven quick response library that learns what to say based on context.

**Status:** Phases 8-9 completed (classifier + response library + semantic matching + StreamComposer). Phases 10-11 (library growth, non-speech) folded into v2.0.

## Current Milestone: v2.0 Always-On Observer

**Goal:** Transform from push-to-talk to always-on listening with the LLM as an autonomous observer that decides when and how to respond.

**Core idea:** Decouple input capture from LLM processing. Audio capture and STT run continuously as an independent stream. A local LLM (Ollama + Llama 3.2 3B) monitors this stream, builds context, and decides when the AI should respond. Response generation uses a configurable backend — Claude CLI for complex work, Ollama for quick responses — chosen automatically based on conditions (network, latency, complexity).

**Target features:**
- Always-on mic with continuous Whisper STT (no PTT activation)
- Local LLM monitoring layer (Ollama) that decides when to respond
- Configurable response backend (Claude CLI / Ollama) — system picks based on conditions
- Name-based interruption ("hey Russel") to cut the AI off mid-response
- Proactive participation — AI joins conversations when it has something to add
- Library growth (from v1.2): post-session curator expands quick response library
- Non-speech awareness (from v1.2): coughs, sighs, laughter get contextual responses

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
- ✓ Heuristic + semantic input classification (<10ms) — v1.2
- ✓ Categorized quick response library with situation → audio clip mapping — v1.2
- ✓ StreamComposer for unified audio queue with pre-buffering and cadence control — v1.2
- ✓ Configurable idle timeout (0 = always-on) — v1.2

### Active

- [ ] Always-on continuous listening (no PTT activation required)
- [ ] Independent input stream (audio capture + STT decoupled from LLM)
- [ ] Local LLM monitoring layer (Ollama + Llama 3.2 3B) decides when to respond
- [ ] Configurable response backend (Claude CLI / Ollama), auto-selected by conditions
- [ ] Name-based interruption ("hey Russel") to cut AI off mid-response
- [ ] Proactive AI participation in conversations
- [ ] Post-session library growth via curator daemon
- [ ] Non-speech event awareness (coughs, sighs, laughter → appropriate responses)

### Out of Scope

- OpenAI TTS (sticking with local Piper for now) — latency vs quality tradeoff, revisit later
- Visual task dashboard — voice-first tool
- Persistent tasks across sessions — each session starts fresh
- Cloud STT — local Whisper is sufficient and free
- Wake word detection hardware — using software-based name recognition instead

## Context

v1.0-v1.2 built a solid PTT-gated voice pipeline. The architecture proved that local-first (Whisper STT, Piper TTS, Claude CLI) works well. v1.2 added intelligent input classification and a StreamComposer for unified audio handling. But the fundamental model — user pushes a button, speaks, releases, AI responds — is reactive. v2.0 inverts this: the AI is always listening, building context from everything it hears, and contributing proactively. This requires decoupling the input stream from LLM processing, adding a local monitoring layer (Ollama), and making the response backend configurable so the system can respond quickly (Ollama) or deeply (Claude) depending on what's needed.

## Constraints

- **Platform**: Linux X11 with PipeWire — no cross-platform concerns
- **Architecture**: Evolving from 5-stage sequential pipeline to decoupled input stream + LLM observer
- **TTS**: Piper local (22050Hz → 24000Hz resampling)
- **STT**: Local Whisper "small" model, continuous operation
- **Monitoring LLM**: Ollama + Llama 3.2 3B (local, free, ~200ms inference)
- **Response LLM**: Configurable — Claude CLI or Ollama, auto-selected

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Claude CLI pipeline instead of OpenAI Realtime | Direct Claude access, no API key dependency for LLM, lower cost | ✓ Good |
| Local Whisper + Piper instead of cloud STT/TTS | Zero latency dependency on cloud, works offline | ✓ Good |
| Acknowledgment phrase fillers | Non-verbal (hums) failed with Piper TTS; verbal acknowledgments work well | ✓ Good |
| Filler clip factory subprocess | Generate varied clips, rotate pool, evaluate naturalness | ✓ Good |
| Gate STT for barge-in instead of mic mute | Mic must stay live for VAD to detect speech during playback | ✓ Good |
| AI-driven quick response library | Dumb random fillers sound wrong in context; AI should choose what to say | ✓ Good (v1.2) |
| Decouple inputs from LLM processing | PTT-gated sequential pipeline prevents always-on and proactive participation | — v2.0 |
| Ollama for monitoring layer | Free, local, ~200ms, fits local-first philosophy. Haiku comparable but costs money and needs network | — v2.0 |
| Configurable response backend | System picks Claude or Ollama based on network, latency, complexity. Self-healing when network drops | — v2.0 |
| PTT replaced by always-on | Always-on with name-based interruption. No more push-to-talk activation | — v2.0 |

---
*Last updated: 2026-02-21 after v2.0 milestone started*
