# Roadmap: Push-to-Talk

## Milestones

- v1.0 Live Mode (Phases 1-3) -- shipped 2026-02-17
- v1.1 Voice UX Polish (Phases 4-7) -- shipped 2026-02-18
- v1.2 Adaptive Quick Responses (Phases 8-11) -- partial, phases 10-11 folded into v2.0
- **v2.0 Always-On Observer (Phases 12-16)** -- in progress

## Phases

<details>
<summary>v1.0 Live Mode (Phases 1-3) - SHIPPED 2026-02-17</summary>

See MILESTONES.md for details.

</details>

<details>
<summary>v1.1 Voice UX Polish (Phases 4-7) - SHIPPED 2026-02-18</summary>

See MILESTONES.md for details.

</details>

<details>
<summary>v1.2 Adaptive Quick Responses (Phases 8-9) - SHIPPED 2026-02-19 (partial)</summary>

Phases 8-9 completed (classifier + response library + semantic matching + StreamComposer).
Phases 10-11 (library growth, non-speech) folded into v2.0 as PLSH-01/02/03.

### Phase 8: Core Classification + Response Library
**Goal**: User hears contextually appropriate quick responses instead of random acknowledgments
**Plans:** 3/3 complete

### Phase 9: Semantic Matching + Pipeline Polish
**Goal**: Classification handles paraphrased and ambiguous inputs gracefully
**Plans:** 3/3 complete

</details>

### v2.0 Always-On Observer (Refreshed)

**Milestone Goal:** Transform from push-to-talk to always-on listening. The AI continuously monitors ambient audio via Deepgram streaming STT, decides when to engage via a local Llama 3.1 8B decision model, and responds through the appropriate backend -- like having a knowledgeable colleague in the room.

- [ ] **Phase 12: Deepgram Streaming Infrastructure** - Continuous cloud STT with lifecycle management, echo suppression, cost control, and Whisper fallback
- [ ] **Phase 13: Decision Engine + Name Activation** - Local LLM monitors transcripts and decides when to respond, with deterministic name-based activation
- [ ] **Phase 14: Response Backend + Integration** - Dual backend routing (Ollama quick / Claude CLI deep), barge-in rework, graceful degradation, end-to-end stability
- [ ] **Phase 15: Proactive Participation** - Unsolicited AI contributions with attention signals, interruptibility detection, and conversation balance
- [ ] **Phase 16: Polish + Enrichment** - Non-speech awareness, library growth curator, library pruning, multi-speaker discrimination

## Phase Details

### Phase 12: Deepgram Streaming Infrastructure
**Goal**: Always-on audio capture produces a clean, bounded, cost-controlled transcript stream via Deepgram Nova-3 -- without feedback loops, runaway billing, or data loss during network blips
**Depends on**: Phase 9 (v1.2 complete)
**Requirements**: STT-01, STT-02, STT-03, STT-04, STT-05, STT-06, RSRC-02, RSRC-04
**Success Criteria** (what must be TRUE):
  1. User speaks naturally and sees real-time transcripts appearing without pressing any button -- Deepgram streams words as they are spoken
  2. AI speech does not appear in the transcript stream -- echo cancellation (PipeWire AEC + transcript fingerprinting) prevents the system from hearing itself
  3. Deepgram WebSocket transitions between active (streaming audio), idle (KeepAlive, free), and sleep (disconnected) based on speech activity -- idle and sleep periods cost nothing
  4. When network drops mid-sentence, system reconnects and no audio is permanently lost -- Whisper loads as fallback if Deepgram stays unreachable
  5. A full 8-hour session with ~30 min actual speech stays under $0.30 Deepgram cost and the transcript buffer stays bounded (no memory growth)
**Plans:** 5 plans
Plans:
- [ ] 12-01-PLAN.md — TDD: DeepgramSTT core class (WebSocket, VAD lifecycle, transcript accumulation)
- [ ] 12-02-PLAN.md — Config/requirements update (SDK pin, API key wiring)
- [ ] 12-03-PLAN.md — TDD: Echo suppression (transcript fingerprinting)
- [ ] 12-04-PLAN.md — Live session integration (rewrite _stt_stage, wire DeepgramSTT)
- [ ] 12-05-PLAN.md — Integration tests, pipeline diagram, deployment verification

### Phase 13: Decision Engine + Name Activation
**Goal**: A local LLM monitors the transcript stream and reliably decides when to respond -- always answering when addressed by name, never responding to background noise or TV dialogue
**Depends on**: Phase 12
**Requirements**: DCSN-01, DCSN-02, DCSN-03, DCSN-04, DCSN-05
**Success Criteria** (what must be TRUE):
  1. User says "hey Russel, what time is it?" and the system always decides to respond -- name activation is deterministic regardless of confidence threshold
  2. TV dialogue or background conversation plays and the system stays silent -- no false positive responses to non-addressed speech
  3. Decision engine outputs structured JSON (should_respond, confidence, response_type, tone, reasoning) after each completed utterance, using transcript buffer context with speaker attribution
  4. Confidence threshold is configurable -- lowering it makes the AI respond to more ambient conversation, raising it restricts responses to direct address only
**Plans**: TBD

### Phase 14: Response Backend + Integration
**Goal**: Decisions flow through to the right LLM backend and produce spoken responses -- the full always-on pipeline works end-to-end with automatic backend selection and graceful degradation
**Depends on**: Phase 13
**Requirements**: RESP-01, RESP-02, RESP-03, RESP-04, RESP-05, RESP-06, RSRC-01, RSRC-03
**Success Criteria** (what must be TRUE):
  1. Simple question ("what's 2+2?") routes to Ollama and responds in under 2 seconds; complex question ("read my config file and explain it") routes to Claude CLI -- user never chooses the backend
  2. User says "Russel" while the AI is speaking and playback stops immediately -- name-based barge-in replaces PTT-based interruption
  3. Response tone matches conversation context -- technical answers in work discussions, casual replies in banter, supportive when user sounds frustrated
  4. System runs for 8+ hours straight without memory leaks, GPU exhaustion, or degraded response quality -- Ollama Llama 3.1 8B stays within RTX 3070 8GB VRAM budget
  5. When network drops, system falls back to Ollama-only; when Ollama is down, heuristic classifier + Claude CLI; when Deepgram is down, local Whisper loads automatically
**Plans**: TBD

### Phase 15: Proactive Participation
**Goal**: The AI contributes to conversations on its own when it has something relevant to add -- without being annoying, dominating, or interrupting at bad times
**Depends on**: Phase 14
**Requirements**: PRCT-01, PRCT-02, PRCT-03, PRCT-04
**Success Criteria** (what must be TRUE):
  1. User discusses a technical problem aloud and the AI volunteers a relevant suggestion without being asked -- proactive contribution based on transcript context
  2. Before any unsolicited response, user hears a brief attention signal (verbal cue or chime) so the AI does not startle them
  3. User says "quiet mode" and proactive responses stop; extended silence (deep work) also suppresses proactive behavior automatically
  4. AI does not respond more than once per ~3 conversational turns unprompted -- it participates without dominating
**Plans**: TBD

### Phase 16: Polish + Enrichment
**Goal**: Non-speech events get contextual responses, the quick response library grows and prunes automatically, and the system distinguishes the user from other audio sources
**Depends on**: Phase 14 (can start in parallel with Phase 15)
**Requirements**: PLSH-01, PLSH-02, PLSH-03, PLSH-04
**Success Criteria** (what must be TRUE):
  1. User coughs or laughs and hears a contextual response (e.g., "bless you", playful comment) instead of silence
  2. After a session ends, the curator daemon generates new response clips for situations that had no library coverage -- library grows over time
  3. Clips that get frequently interrupted or produce no engagement are deprioritized or removed -- library quality improves over time
  4. System responds to the user's voice but stays silent when TV dialogue or other people talk in the room
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 12 -> 13 -> 14 -> 15 -> 16
(Phase 16 can start in parallel with Phase 15 after Phase 14 is complete.)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 8. Core Classification + Response Library | v1.2 | 3/3 | Complete | 2026-02-19 |
| 9. Semantic Matching + Pipeline Polish | v1.2 | 3/3 | Complete | 2026-02-20 |
| 10. Library Growth + Pruning | v1.2 | -- | Deferred to v2.0 (PLSH-02/03) | -- |
| 11. Non-Speech Awareness | v1.2 | -- | Deferred to v2.0 (PLSH-01) | -- |
| 12. Deepgram Streaming Infrastructure | v2.0 | 0/5 | Planned | - |
| 13. Decision Engine + Name Activation | v2.0 | 0/TBD | Not started | - |
| 14. Response Backend + Integration | v2.0 | 0/TBD | Not started | - |
| 15. Proactive Participation | v2.0 | 0/TBD | Not started | - |
| 16. Polish + Enrichment | v2.0 | 0/TBD | Not started | - |
