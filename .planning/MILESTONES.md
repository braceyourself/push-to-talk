# Project Milestones: Push-to-Talk

## v1.1 Voice UX Polish (Shipped: 2026-02-18)

**Delivered:** Natural voice conversation UX with barge-in interruption, intelligent filler management, multi-layer STT filtering, and tool-aware overlay status.

**Phases completed:** 4-7 (8 plans total)

**Key accomplishments:**
- Barge-in interruption: user can speak over AI with VAD detection, playback cancellation, context annotation, and shortened post-interrupt silence
- Acknowledgment clip factory: Piper TTS generates, evaluates, and rotates verbal filler clips with quality gating
- Multi-layer Whisper STT filtering: rejects coughs, noise, and hallucinated text with 3-layer segment analysis
- Tool-use speech flow: suppresses inter-tool narration, gated pre-tool acknowledgment, only speaks final response
- Dynamic overlay: tool intent labels, JSON status protocol, history coalescing, STT rejection flash

**Stats:**
- 59 files changed, +5754/-384 lines
- 8,149 LOC Python (core files)
- 4 phases, 8 plans
- 2 days (2026-02-17 → 2026-02-18)

**Git range:** `docs(04)` → `docs(07)`

**What's next:** v1.2 Adaptive Quick Responses — AI-driven situational response library that learns what to say based on context

---

## v1.0: Live Mode (Completed 2026-02-17)

**Goal:** Real-time voice-to-voice AI conversation with async Claude CLI task orchestration.

**What shipped:**
- Mode rename: "live" dictation → "dictate", new "live" mode for voice conversation
- 5-stage asyncio pipeline: audio capture → STT (Whisper) → LLM (Claude CLI) → TTS (Piper) → playback
- Claude CLI integration with stream-json protocol, MCP tool server
- TaskManager for async background task spawning/tracking
- Filler system (smart + canned clips)
- Learner daemon (watches conversations, writes persistent memories)
- Live overlay widget with status dot, drag, model selection
- Conversation logging (JSONL)

**Note:** Implementation diverged from original roadmap — used Claude CLI pipeline directly instead of OpenAI Realtime API. Phase 3 task orchestration requirements partially addressed through MCP tools and notification system.

**Phases:** 1-3 (3 phases, 5 plans)

---
*Archived: 2026-02-17*
