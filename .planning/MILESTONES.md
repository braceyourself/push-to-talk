# Milestones

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
