# Requirements: Push-to-Talk v1.2

**Defined:** 2026-02-18
**Core Value:** Natural, low-friction voice conversation with Claude that feels like talking to a person

## v1 Requirements

Requirements for v1.2 Adaptive Quick Responses milestone.

### Classification

- [ ] **CLAS-01**: System classifies STT output into coarse categories (question, command, agreement, emotional, greeting, farewell) via heuristic pattern matching in under 1ms
- [ ] **CLAS-02**: Semantic similarity matching via model2vec serves as fallback when heuristic classifier produces no confident match
- [ ] **CLAS-03**: Short or trivial inputs receive silence-as-response (natural pause) instead of a filler clip

### Response Library

- [ ] **RLIB-01**: Response library stores situation-to-audio-clip mappings organized by input category
- [ ] **RLIB-02**: Seed library ships with 30-40 pre-generated Piper TTS clips across all categories, available on first use
- [ ] **RLIB-03**: Category-aware filler selection replaces random clip selection in `_filler_manager()`
- [ ] **RLIB-04**: Each clip tracks usage count and contextual accuracy metrics for pruning decisions

### Non-Speech Awareness

- [ ] **NSPC-01**: System detects non-speech vocalizations (coughs, sighs, laughter) from STT rejection metadata and VAD signals
- [ ] **NSPC-02**: Non-speech events trigger contextual responses (cough -> "excuse you", sigh -> empathetic acknowledgment)
- [ ] **NSPC-03**: Non-speech detection uses configurable confidence threshold to manage false positive rate

### Pipeline Integration

- [ ] **PIPE-01**: Classification and response selection complete within existing 500ms filler gate with no added perceptible latency
- [ ] **PIPE-02**: Quick response clips integrate with barge-in system via sentence tracking awareness
- [ ] **PIPE-03**: Playback handles collision-free transition from quick response clip to full LLM TTS response (sink-side frame draining)

### Library Management

- [ ] **LMGT-01**: Post-session curator daemon analyzes conversation to identify uncovered situations and generates new clips
- [ ] **LMGT-02**: Library pruning removes or deprioritizes clips with low effectiveness based on usage tracking
- [ ] **LMGT-03**: Curator daemon follows learner.py pattern (subprocess, signal file notification)

## v2 Requirements

Deferred to future milestones.

### Advanced Classification

- **CLAS-04**: Multi-turn context tracking (response depends on conversation history, not just current input)
- **CLAS-05**: Prosodic analysis (tone, pitch, speaking rate inform category selection)

### Advanced Library

- **RLIB-05**: Dynamic TTS generation (generate clips on-the-fly instead of pre-cached, requires faster TTS)
- **RLIB-06**: Multiple voice/personality profiles with distinct response libraries
- **LMGT-04**: A/B effectiveness comparison between response variants

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cloud API classification | Latency incompatible with <50ms budget, defeats local-first design |
| Full NLU pipeline | Over-engineering for 5-6 coarse categories on single sentences |
| 30+ category taxonomy | Research shows accuracy drops to ~60%, worse than random generic fillers |
| ML classifier for v1 | Pattern matching handles 90%+ of cases in <1ms with zero dependencies |
| Category-dependent gate timing | Premature optimization, uniform gate sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLAS-01 | Phase 8 | Pending |
| CLAS-02 | Phase 9 | Pending |
| CLAS-03 | Phase 9 | Pending |
| RLIB-01 | Phase 8 | Pending |
| RLIB-02 | Phase 8 | Pending |
| RLIB-03 | Phase 8 | Pending |
| RLIB-04 | Phase 8 | Pending |
| NSPC-01 | Phase 11 | Pending |
| NSPC-02 | Phase 11 | Pending |
| NSPC-03 | Phase 11 | Pending |
| PIPE-01 | Phase 8 | Pending |
| PIPE-02 | Phase 9 | Pending |
| PIPE-03 | Phase 9 | Pending |
| LMGT-01 | Phase 10 | Pending |
| LMGT-02 | Phase 10 | Pending |
| LMGT-03 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-02-18*
*Last updated: 2026-02-18 after roadmap creation*
