# Roadmap: Push-to-Talk

## Milestones

- v1.0 Live Mode (Phases 1-3) -- shipped 2026-02-17
- v1.1 Voice UX Polish (Phases 4-7) -- shipped 2026-02-18
- **v1.2 Adaptive Quick Responses (Phases 8-11)** -- in progress

## Phases

<details>
<summary>v1.0 Live Mode (Phases 1-3) - SHIPPED 2026-02-17</summary>

See MILESTONES.md for details.

</details>

<details>
<summary>v1.1 Voice UX Polish (Phases 4-7) - SHIPPED 2026-02-18</summary>

See MILESTONES.md for details.

</details>

### v1.2 Adaptive Quick Responses

**Milestone Goal:** Replace random filler selection with an AI-driven quick response library that understands context, learns across sessions, and responds to non-speech events.

- [ ] **Phase 8: Core Classification + Response Library** - Heuristic classifier and categorized clip library replace random filler selection
- [ ] **Phase 9: Semantic Matching + Pipeline Polish** - model2vec fallback improves accuracy; barge-in and clip-to-LLM transitions work seamlessly
- [ ] **Phase 10: Library Growth + Pruning** - Curator daemon expands and refines the library after each session
- [ ] **Phase 11: Non-Speech Awareness** - Coughs, sighs, and laughter get contextual responses instead of silence

## Phase Details

### Phase 8: Core Classification + Response Library
**Goal**: User hears contextually appropriate quick responses instead of random acknowledgments -- task commands get "on it", questions get "hmm", greetings get "hey"
**Depends on**: Phase 7 (v1.1 complete)
**Requirements**: CLAS-01, RLIB-01, RLIB-02, RLIB-03, RLIB-04, PIPE-01
**Success Criteria** (what must be TRUE):
  1. User says a question and hears a question-appropriate filler (not a task-oriented one)
  2. User says a command and hears a task-appropriate filler (not a conversational one)
  3. System launches with a working seed library of 30-40 clips across all categories on first use
  4. Classification completes within the existing 500ms filler gate with no perceptible added latency
  5. If classification fails or confidence is low, user hears a generic acknowledgment (never silence, never a wrong-category clip)
**Plans:** 3 plans
Plans:
- [ ] 08-01-PLAN.md -- Classifier daemon + response library modules
- [ ] 08-02-PLAN.md -- Seed phrase list + clip generation
- [ ] 08-03-PLAN.md -- Pipeline integration + end-to-end verification

### Phase 9: Semantic Matching + Pipeline Polish
**Goal**: Classification handles paraphrased and ambiguous inputs gracefully, quick response clips integrate cleanly with barge-in and LLM playback transitions
**Depends on**: Phase 8
**Requirements**: CLAS-02, CLAS-03, PIPE-02, PIPE-03
**Success Criteria** (what must be TRUE):
  1. User says something that doesn't match keyword patterns (e.g., "could you take a peek at this") and still gets a task-appropriate response via semantic matching
  2. User says a trivial input ("yes", "ok", "mhm") and hears natural silence instead of an unnecessary filler
  3. User barges in during a quick response clip and the clip stops cleanly without leaving audio artifacts or confusing the LLM context
  4. Quick response clip transitions smoothly to LLM TTS response with no overlap, gap, or audio glitch
**Plans**: TBD

### Phase 10: Library Growth + Pruning
**Goal**: The response library improves automatically across sessions -- new situations get coverage, ineffective clips get phased out
**Depends on**: Phase 8
**Requirements**: LMGT-01, LMGT-02, LMGT-03
**Success Criteria** (what must be TRUE):
  1. After a session where the user said things with no matching clip, new clips appear in the library for those situations before the next session
  2. Clips that are frequently interrupted or feel wrong get deprioritized or removed over time
  3. Curator runs as a background subprocess (like learner.py) and notifies the system when new clips are ready
**Plans**: TBD

### Phase 11: Non-Speech Awareness
**Goal**: Non-speech vocalizations get playful contextual responses instead of being silently ignored
**Depends on**: Phase 8
**Requirements**: NSPC-01, NSPC-02, NSPC-03
**Success Criteria** (what must be TRUE):
  1. User coughs and hears "excuse you" (or similar) instead of silence
  2. User sighs and hears an empathetic acknowledgment instead of silence
  3. Non-speech detection has a configurable confidence threshold so false positives (normal speech misidentified as cough) can be tuned down
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 8 -> 9 -> 10 -> 11
(Phases 10 and 11 both depend on Phase 8 only, but 10 runs first because library growth infrastructure is useful for non-speech clip generation.)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 8. Core Classification + Response Library | v1.2 | 0/3 | Planning complete | - |
| 9. Semantic Matching + Pipeline Polish | v1.2 | 0/TBD | Not started | - |
| 10. Library Growth + Pruning | v1.2 | 0/TBD | Not started | - |
| 11. Non-Speech Awareness | v1.2 | 0/TBD | Not started | - |
