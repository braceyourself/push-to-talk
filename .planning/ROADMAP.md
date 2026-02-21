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
Phases 10-11 (library growth, non-speech) folded into v2.0 as NSPL-01/02/03.

### Phase 8: Core Classification + Response Library
**Goal**: User hears contextually appropriate quick responses instead of random acknowledgments
**Plans:** 3/3 complete

### Phase 9: Semantic Matching + Pipeline Polish
**Goal**: Classification handles paraphrased and ambiguous inputs gracefully
**Plans:** 3/3 complete

</details>

### v2.0 Always-On Observer

**Milestone Goal:** Transform from push-to-talk to always-on listening. The AI continuously monitors ambient audio, decides when to engage via a local LLM, and responds through the appropriate backend -- like having a knowledgeable colleague in the room.

- [ ] **Phase 12: Infrastructure + Safety Net** - Continuous audio capture, STT, echo cancellation, transcript buffer, VRAM validation
- [ ] **Phase 13: Decision Engine** - Ollama monitoring loop with name activation, structured decisions, speaker awareness
- [ ] **Phase 14: Response Backend + Integration** - Dual backend routing, pipeline rewire, barge-in rework, stability
- [ ] **Phase 15: Proactive Participation** - Unsolicited contributions, attention signals, interruptibility, conversation balance
- [ ] **Phase 16: Polish + Library Growth** - Non-speech awareness, curator daemon, library pruning, multi-speaker discrimination

## Phase Details

### Phase 12: Infrastructure + Safety Net
**Goal**: Always-on audio capture produces a clean, bounded, low-hallucination transcript stream that the monitoring LLM can trust -- without VRAM exhaustion or feedback loops
**Depends on**: Phase 9 (v1.2 complete)
**Requirements**: CSTR-01, CSTR-02, CSTR-03, CSTR-05, RSRC-02, RSRC-03
**Success Criteria** (what must be TRUE):
  1. System captures and transcribes speech continuously without any button press -- user speaks normally and sees transcripts appearing in the log
  2. AI speech does not appear in the transcript stream -- echo cancellation prevents the AI from hearing and responding to itself
  3. Whisper and Ollama run simultaneously on the RTX 3070 without OOM crashes during a 30-minute stress test
  4. Transcript buffer holds ~5 minutes of context and older entries drop off automatically -- no unbounded memory growth
  5. Hallucination rate on ambient audio (keyboard, HVAC, silence) stays below 5% of transcribed segments
**Plans**: TBD

### Phase 13: Decision Engine
**Goal**: A local LLM monitors the transcript stream and reliably decides when to respond -- always responding when addressed by name, never responding to background noise or TV dialogue
**Depends on**: Phase 12
**Requirements**: CSTR-04, MNTR-01, MNTR-02, MNTR-04, MNTR-05
**Success Criteria** (what must be TRUE):
  1. User says "hey Russel, what time is it?" and the system always decides to respond (name activation is deterministic)
  2. TV dialogue or background conversation plays and the system stays silent (no false positive responses)
  3. Decision engine outputs structured JSON (should_respond, confidence, response_type, reasoning) after each user utterance
  4. Confidence threshold is configurable -- lowering it makes the AI more willing to respond, raising it makes it more conservative
  5. Transcript segments include speaker attribution (user vs AI vs other) that the decision engine uses for context
**Plans**: TBD

### Phase 14: Response Backend + Integration
**Goal**: Decisions from the monitor flow through to the right LLM backend and produce spoken responses -- the full pipeline works end-to-end as an always-on assistant
**Depends on**: Phase 13
**Requirements**: RESP-01, RESP-02, RESP-03, RESP-04, MNTR-03, RSRC-01, RSRC-04
**Success Criteria** (what must be TRUE):
  1. Simple question ("what's 2+2?") routes to Ollama and responds in under 2 seconds; complex question ("read my config file and explain it") routes to Claude CLI
  2. User says "Russel" while the AI is speaking and playback stops immediately (name-based barge-in replaces PTT-based)
  3. Response tone matches conversation context -- technical answers in work discussions, casual replies in banter
  4. System runs for 8+ hours without memory leaks, GPU exhaustion, or degraded response quality
  5. When network drops, system falls back to Ollama-only responses; when Ollama is down, falls back to Claude CLI with heuristic classifier
**Plans**: TBD

### Phase 15: Proactive Participation
**Goal**: The AI contributes to conversations on its own when it has something relevant to add -- without being annoying, dominating, or interrupting at bad times
**Depends on**: Phase 14
**Requirements**: PRCT-01, PRCT-02, PRCT-03, PRCT-04
**Success Criteria** (what must be TRUE):
  1. User discusses a technical problem aloud and the AI volunteers a relevant suggestion without being asked (proactive contribution)
  2. Before any unsolicited response, user hears a brief attention signal (verbal cue or chime) so the AI doesn't startle them
  3. User says "quiet mode" and proactive responses stop; long silence (deep work) also suppresses proactive behavior
  4. AI does not respond more than once per ~3 conversational turns unprompted -- it participates, not dominates
**Plans**: TBD

### Phase 16: Polish + Library Growth
**Goal**: Non-speech events get contextual responses, the quick response library grows automatically, and the system distinguishes the user from other audio sources
**Depends on**: Phase 14 (can start in parallel with Phase 15)
**Requirements**: NSPL-01, NSPL-02, NSPL-03, SPKR-01, SPKR-02
**Success Criteria** (what must be TRUE):
  1. User coughs or laughs and hears a playful contextual response (e.g., "bless you") instead of silence
  2. After a session, the curator daemon generates new response clips for situations that had no coverage -- library grows over time
  3. Clips that get frequently interrupted or feel wrong are deprioritized or removed -- library quality improves over time
  4. System responds to the user's voice but ignores TV dialogue and other people talking in the room
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 12 -> 13 -> 14 -> 15 -> 16
(Phase 16 can start in parallel with Phase 15 after Phase 14 is complete.)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 8. Core Classification + Response Library | v1.2 | 3/3 | Complete | 2026-02-19 |
| 9. Semantic Matching + Pipeline Polish | v1.2 | 3/3 | Complete | 2026-02-20 |
| 10. Library Growth + Pruning | v1.2 | — | Deferred to v2.0 (NSPL-02/03) | — |
| 11. Non-Speech Awareness | v1.2 | — | Deferred to v2.0 (NSPL-01) | — |
| 12. Infrastructure + Safety Net | v2.0 | 0/TBD | Not started | - |
| 13. Decision Engine | v2.0 | 0/TBD | Not started | - |
| 14. Response Backend + Integration | v2.0 | 0/TBD | Not started | - |
| 15. Proactive Participation | v2.0 | 0/TBD | Not started | - |
| 16. Polish + Library Growth | v2.0 | 0/TBD | Not started | - |
