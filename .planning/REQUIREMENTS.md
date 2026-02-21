# Requirements: Push-to-Talk v2.0

**Defined:** 2026-02-21
**Core Value:** An always-present AI that listens, understands context, and contributes when it has something useful to add

## v2.0 Requirements

Requirements for v2.0 Always-On Observer milestone.

### Continuous Input Stream

- [ ] **CSTR-01**: Audio capture runs continuously without PTT activation, gated by Silero VAD to only process speech segments
- [ ] **CSTR-02**: Whisper STT transcribes speech segments continuously, producing a stream of timestamped transcripts
- [ ] **CSTR-03**: Rolling transcript buffer maintains a sliding window of recent transcripts (configurable, ~5-10 minutes) as the monitoring LLM's context
- [ ] **CSTR-04**: Transcript buffer includes speaker attribution where detectable (user vs AI vs other)
- [ ] **CSTR-05**: Audio feedback loop prevention via PipeWire echo cancellation prevents the AI from hearing and responding to its own speech

### Monitoring Decision Engine

- [ ] **MNTR-01**: Ollama (Llama 3.2 3B) evaluates the transcript buffer after each speech segment and outputs a structured JSON decision (should_respond, confidence, response_type, tone)
- [ ] **MNTR-02**: Name-based activation ("hey Russel" / "Russell" / "Russ") bypasses confidence threshold — AI always responds when addressed by name
- [ ] **MNTR-03**: Name detection during AI playback triggers barge-in interruption (replacing PTT-based interruption)
- [ ] **MNTR-04**: Decision engine considers addressee detection, relevance, urgency, and interruption cost when deciding whether to respond
- [ ] **MNTR-05**: Configurable confidence threshold controls how aggressively the AI participates (low = proactive, high = conservative)

### Response Backend

- [ ] **RESP-01**: System automatically selects response backend (Claude CLI or Ollama) based on query complexity, network availability, and expected latency
- [ ] **RESP-02**: Ollama generates quick conversational responses (~200ms) for simple interactions and proactive contributions
- [ ] **RESP-03**: Claude CLI handles complex queries, tool-using requests, and multi-step reasoning (existing pipeline)
- [ ] **RESP-04**: Response tone/style adapts to conversation context — technical in work discussions, casual in banter, supportive when user sounds frustrated

### Proactive Participation

- [ ] **PRCT-01**: AI proactively contributes to conversations when it has relevant information, even without being addressed
- [ ] **PRCT-02**: Attention signal (brief verbal cue or audio chime) plays before unsolicited proactive responses
- [ ] **PRCT-03**: Interruptibility detection suppresses proactive responses when user appears busy (long silence = deep work, explicit "quiet mode" command)
- [ ] **PRCT-04**: Conversation balance tracking prevents the AI from dominating the conversation

### Resource Management

- [ ] **RSRC-01**: System runs continuously for 8+ hours without memory leaks, GPU exhaustion, or CPU runaway
- [ ] **RSRC-02**: Transcript buffer is bounded (ring buffer) with configurable size, older entries dropped or summarized
- [ ] **RSRC-03**: GPU VRAM manages concurrent Whisper + Ollama inference within RTX 3070 8GB budget
- [ ] **RSRC-04**: Graceful degradation chain: Ollama down → heuristic classifier + Claude CLI; network down → Ollama only; GPU pressure → downgrade Whisper model

### Non-Speech & Library (from v1.2)

- [ ] **NSPL-01**: Non-speech vocalizations (coughs, sighs, laughter) detected from STT rejection metadata trigger contextual responses
- [ ] **NSPL-02**: Post-session curator daemon analyzes conversations, identifies response gaps, and generates new quick response clips
- [ ] **NSPL-03**: Library pruning removes or deprioritizes clips with low effectiveness based on usage tracking

### Multi-Speaker Awareness

- [ ] **SPKR-01**: System distinguishes primary user voice from other audio sources (TV, guests, other people) using energy profiles and proximity heuristics
- [ ] **SPKR-02**: Speaker attribution labels in transcript buffer improve decision engine accuracy (respond to user, not TV dialogue)

## v3 Requirements

Deferred to future milestones.

### Advanced Proactivity

- **PRCT-05**: Learning user schedule/routines for time-based proactive offers
- **PRCT-06**: Multi-turn conversation memory across sessions (persistent context)

### Advanced Speaker

- **SPKR-03**: Speaker diarization via pyannote.audio for accurate multi-speaker tracking
- **SPKR-04**: Voice enrollment — user registers their voice for personalized recognition

### Advanced Response

- **RESP-05**: Dynamic TTS fillers generated on-the-fly with contextual text
- **RESP-06**: Multiple voice/personality profiles with distinct response libraries

## Out of Scope

| Feature | Reason |
|---------|--------|
| Hardware wake word detection (Picovoice/openWakeWord) | Whisper STT already runs continuously — name detection on transcripts is simpler and cheaper |
| Cloud-based monitoring LLM | Privacy concern (always-on mic), adds latency, costs money, violates local-first philosophy |
| Full conversation transcription storage | Surveillance feature — rolling buffer is ephemeral by design |
| Speech-to-speech model (GPT-4o Realtime) | Cloud-only, expensive, designed for 1:1 conversation not ambient monitoring |
| Multi-room / multi-device mesh | Massive scope expansion — one device, one room, one mic |
| Routine-based proactivity | Different product — Russel participates in conversations, doesn't initiate from silence |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CSTR-01 | TBD | Pending |
| CSTR-02 | TBD | Pending |
| CSTR-03 | TBD | Pending |
| CSTR-04 | TBD | Pending |
| CSTR-05 | TBD | Pending |
| MNTR-01 | TBD | Pending |
| MNTR-02 | TBD | Pending |
| MNTR-03 | TBD | Pending |
| MNTR-04 | TBD | Pending |
| MNTR-05 | TBD | Pending |
| RESP-01 | TBD | Pending |
| RESP-02 | TBD | Pending |
| RESP-03 | TBD | Pending |
| RESP-04 | TBD | Pending |
| PRCT-01 | TBD | Pending |
| PRCT-02 | TBD | Pending |
| PRCT-03 | TBD | Pending |
| PRCT-04 | TBD | Pending |
| RSRC-01 | TBD | Pending |
| RSRC-02 | TBD | Pending |
| RSRC-03 | TBD | Pending |
| RSRC-04 | TBD | Pending |
| NSPL-01 | TBD | Pending |
| NSPL-02 | TBD | Pending |
| NSPL-03 | TBD | Pending |
| SPKR-01 | TBD | Pending |
| SPKR-02 | TBD | Pending |

**Coverage:**
- v2.0 requirements: 27 total
- Mapped to phases: 0 (awaiting roadmap)
- Unmapped: 27

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-02-21 after initial definition*
