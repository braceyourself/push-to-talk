# Roadmap: Push-to-Talk

## Overview

This roadmap covers all milestones. v1.0 delivered the core live voice session with async task orchestration. v1.1 polishes the voice UX — overhauling the filler system, enabling barge-in interruption, and verifying STT/speech-flow pre-work.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): v1.0 milestone (complete)
- Phases 4-6: v1.1 Voice UX Polish
- Decimal phases (e.g., 4.1): Urgent insertions (marked with INSERTED)

### v1.0: Live Mode (Complete)

- [x] **Phase 1: Mode Rename and Live Voice Session** - Rename "live" to "dictate" and create a new live mode with real-time voice conversation
- [x] **Phase 2: Async Task Infrastructure** - Build TaskManager and ClaudeTask classes for non-blocking Claude CLI subprocess management
- [x] **Phase 3: Voice-Controlled Task Orchestration** - Wire task management into the live session via MCP tools and notification system

### v1.1: Voice UX Polish

- [x] **Phase 4: Filler System Overhaul** - Replace smart fillers with non-verbal clips and a clip factory subprocess
- [x] **Phase 5: Barge-in** - User can interrupt AI mid-speech via voice activity detection
- [x] **Phase 6: Polish & Verification** - Verify and tune all pre-work features end-to-end

## Phase Details

### Phase 1: Mode Rename and Live Voice Session (v1.0) ✓
**Goal**: User can select the new "live" dictation mode and have a real-time voice conversation with AI, with the old live mode cleanly renamed to "dictate"
**Requirements**: RENAME-01..04, LIVE-01..04
**Status**: Complete (2026-02-13)

Plans:
- [x] 01-01-PLAN.md -- Rename "live" dictation mode to "dictate" across codebase, config, and UI
- [x] 01-02-PLAN.md -- Implement new live mode with LiveSession, personality system, and overlay widget

### Phase 2: Async Task Infrastructure (v1.0) ✓
**Goal**: A TaskManager can spawn, track, query, and cancel isolated Claude CLI subprocesses without blocking the asyncio event loop
**Requirements**: INFRA-01..06
**Status**: Complete (2026-02-15)

Plans:
- [x] 02-01-PLAN.md -- TaskManager singleton and ClaudeTask with full async subprocess lifecycle and integration tests

### Phase 3: Voice-Controlled Task Orchestration (v1.0) ✓
**Goal**: User can manage Claude CLI tasks entirely by voice during a live session
**Requirements**: TASK-01..07, CTX-01..03
**Status**: Complete (2026-02-17)

Plans:
- [x] 03-01-PLAN.md -- Realtime API tool definitions, function call handler, and task orchestrator personality
- [x] 03-02-PLAN.md -- Task completion/failure notifications, queue-based delivery, and ambient task awareness

### Phase 4: Filler System Overhaul (v1.1) ✓
**Goal**: Replace Ollama smart filler generation with non-verbal audio clips managed by a clip factory subprocess that generates, evaluates, and rotates a capped pool of natural-sounding clips
**Depends on**: Nothing (independent of other v1.1 phases)
**Requirements**: FILL-01, FILL-02, FILL-03, FILL-04, FILL-05
**Status**: Complete (2026-02-17)
**Success Criteria** (what must be TRUE):
  1. No Ollama/LLM-generated filler text is spoken during live sessions
  2. Fillers are exclusively non-verbal audio clips (breaths, hums, etc.)
  3. A background subprocess generates new clips via Piper TTS
  4. The clip pool has a configurable size cap and rotates old clips out
  5. Generated clips are evaluated for naturalness before being added to the pool
**Plans:** 2 plans

Plans:
- [x] 04-01-PLAN.md -- Create clip factory daemon with generation, evaluation, and pool rotation
- [x] 04-02-PLAN.md -- Remove smart filler code, simplify filler system, wire clip factory, clean up

### Phase 5: Barge-in (v1.1) ✓
**Goal**: User can interrupt AI mid-speech by speaking, which cancels current TTS playback and queued audio, allowing the conversation to continue naturally
**Depends on**: Nothing (independent of other v1.1 phases)
**Requirements**: BARGE-01, BARGE-02, BARGE-03, BARGE-04
**Status**: Complete (2026-02-17)
**Success Criteria** (what must be TRUE):
  1. User can speak while AI is talking and be heard (mic not muted during playback)
  2. VAD detects user speech during AI playback
  3. Detection cancels current TTS playback and all queued audio frames
  4. Interrupted (unspoken) text is either excluded from context or marked as interrupted
  5. Conversation continues naturally after interruption
**Plans:** 2 plans

Plans:
- [x] 05-01-PLAN.md -- STT gating, VAD detection in STT stage, barge-in trigger with fade-out and cooldown
- [x] 05-02-PLAN.md -- Sentence tracking, interruption context annotation, post-interrupt silence tuning

### Phase 6: Polish & Verification (v1.1) ✓
**Goal**: Verify and tune all pre-work features (STT filtering, tool-use speech suppression, overlay states) end-to-end to ensure they work correctly in real usage
**Depends on**: Phase 4, Phase 5 (run after new features are stable)
**Requirements**: STT-01, STT-02, FLOW-01, FLOW-02, OVL-01, OVL-02
**Status**: Complete (2026-02-18)
**Success Criteria** (what must be TRUE):
  1. Whisper no_speech_prob filtering correctly rejects throat clearing, coughs, ambient noise
  2. STT false trigger rate is acceptably low in real usage
  3. Only the final post-tool response is spoken; inter-tool narration is discarded
  4. All overlay status states (listening, thinking, tool_use, speaking, idle, muted) render correctly
  5. Status history panel shows transitions with timestamps
**Plans:** 3 plans

Plans:
- [x] 06-01-PLAN.md -- Multi-layer Whisper filtering (logprob + compression ratio) and STT rejection flash in overlay
- [x] 06-02-PLAN.md -- Acknowledgment clip factory, gated pre-tool playback, tool intent map, JSON status metadata
- [x] 06-03-PLAN.md -- Dynamic tool-use overlay with intent labels, history enrichment, end-to-end verification

## Progress

**Execution Order:**
v1.0: 1 → 2 → 3 (complete)
v1.1: 4 & 5 (parallel) → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Mode Rename and Live Voice Session | 2/2 | Complete | 2026-02-13 |
| 2. Async Task Infrastructure | 1/1 | Complete | 2026-02-15 |
| 3. Voice-Controlled Task Orchestration | 2/2 | Complete | 2026-02-17 |
| 4. Filler System Overhaul | 2/2 | Complete | 2026-02-17 |
| 5. Barge-in | 2/2 | Complete | 2026-02-17 |
| 6. Polish & Verification | 3/3 | Complete | 2026-02-18 |
