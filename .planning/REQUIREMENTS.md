# Requirements: Push-to-Talk v2.0 (Refreshed)

**Defined:** 2026-02-22
**Core Value:** An always-present AI that listens, understands context, and contributes when it has something useful to add

## v2.0 Requirements

Requirements for v2.0 Always-On Observer milestone (Deepgram streaming + local decision model architecture).

### Streaming STT Infrastructure

- [ ] **STT-01**: Deepgram Nova-3 WebSocket streams real-time transcripts as the user speaks, producing TranscriptSegment objects in the existing TranscriptBuffer
- [ ] **STT-02**: Silero VAD manages Deepgram connection lifecycle — connect on first speech detection, KeepAlive during silence, disconnect after extended quiet (not per-chunk audio gating)
- [ ] **STT-03**: KeepAlive messages maintain WebSocket during silence without incurring Deepgram audio billing
- [ ] **STT-04**: Utterance boundaries detected via dual trigger (`speech_final` + `utterance_end_ms`) for reliable sentence detection in noisy environments
- [ ] **STT-05**: Echo suppression prevents AI from hearing its own speech — PipeWire AEC as primary, transcript fingerprinting against recent AI speech as secondary
- [ ] **STT-06**: Local Whisper loads on demand as fallback when Deepgram is unavailable (network down, API error)

### Decision Engine

- [ ] **DCSN-01**: Llama 3.1 8B via Ollama evaluates the transcript buffer after each utterance and outputs a structured JSON decision (should_respond, confidence, response_type, tone, reasoning)
- [ ] **DCSN-02**: Name-based activation ("hey Russel" / "Russell" / "Russ" and fuzzy variants) bypasses confidence threshold — AI always responds when addressed by name
- [ ] **DCSN-03**: Decision engine considers addressee detection, relevance, urgency, and interruption cost when deciding whether to respond
- [ ] **DCSN-04**: Configurable confidence threshold controls how aggressively the AI participates (low = proactive, high = conservative)
- [ ] **DCSN-05**: Transcript segments include speaker attribution (user vs AI) that the decision engine uses for context

### Response Backend

- [ ] **RESP-01**: System automatically selects response backend (Claude CLI or Ollama) based on query complexity, network availability, and expected latency
- [ ] **RESP-02**: Ollama generates quick conversational responses (~200ms) for simple interactions
- [ ] **RESP-03**: Claude CLI handles complex queries, tool-using requests, and multi-step reasoning (existing pipeline)
- [ ] **RESP-04**: Response tone adapts to conversation context — technical in work discussions, casual in banter, supportive when user sounds frustrated
- [ ] **RESP-05**: Name spoken during AI playback triggers barge-in interruption (replacing PTT-based interruption)
- [ ] **RESP-06**: Graceful degradation chain: Deepgram down → load local Whisper; Ollama down → heuristic classifier + Claude CLI; network down → Ollama only

### Proactive Participation

- [ ] **PRCT-01**: AI proactively contributes to conversations when it has relevant information, even without being addressed
- [ ] **PRCT-02**: Attention signal (brief verbal cue or audio chime) plays before unsolicited proactive responses
- [ ] **PRCT-03**: Interruptibility detection suppresses proactive responses when user appears busy (long silence = deep work, explicit "quiet mode" command)
- [ ] **PRCT-04**: Conversation balance tracking prevents the AI from dominating — no more than one unsolicited response per ~3 conversational turns

### Polish + Enrichment

- [ ] **PLSH-01**: Non-speech vocalizations (coughs, sighs, laughter) detected and trigger contextual responses
- [ ] **PLSH-02**: Post-session curator daemon analyzes conversations, identifies response gaps, and generates new quick response clips
- [ ] **PLSH-03**: Library pruning removes or deprioritizes clips with low effectiveness based on usage tracking
- [ ] **PLSH-04**: System distinguishes primary user voice from other audio sources (TV, guests) using Deepgram diarization or energy heuristics

### Resource Management

- [ ] **RSRC-01**: System runs continuously for 8+ hours without memory leaks, GPU exhaustion, or degraded quality
- [ ] **RSRC-02**: TranscriptBuffer is bounded (ring buffer) with configurable size, older entries dropped automatically
- [ ] **RSRC-03**: GPU VRAM manages Ollama Llama 3.1 8B within RTX 3070 8GB budget
- [ ] **RSRC-04**: Deepgram API cost stays under $0.30/day for typical usage (8hr session, ~30 min actual speech)

## v3 Requirements

Deferred to future milestones.

### Advanced Proactivity

- **PRCT-05**: Learning user schedule/routines for time-based proactive offers
- **PRCT-06**: Multi-turn conversation memory across sessions (persistent context)

### Advanced Speaker

- **SPKR-03**: Speaker diarization via pyannote.audio for accurate multi-speaker tracking
- **SPKR-04**: Voice enrollment — user registers their voice for personalized recognition

### Advanced Response

- **RESP-07**: Dynamic TTS fillers generated on-the-fly with contextual text
- **RESP-08**: Multiple voice/personality profiles with distinct response libraries

## Out of Scope

| Feature | Reason |
|---------|--------|
| Fully local STT (no cloud) | Deepgram streaming provides 10x better latency than local Whisper for always-on use. Local Whisper retained as fallback only. |
| Hardware wake word detection (Picovoice/openWakeWord) | Deepgram transcribes continuously — name detection on transcripts is simpler |
| Speech-to-speech model (GPT-4o Realtime) | Too expensive for always-on ($5-20+/hr with context accumulation) |
| Full conversation storage | Surveillance concern — rolling buffer is ephemeral by design |
| Multi-room / multi-device | One device, one room, one mic |
| OpenAI Realtime API as observer | Context accumulation makes it cost-prohibitive for ambient listening |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STT-01 | TBD | Pending |
| STT-02 | TBD | Pending |
| STT-03 | TBD | Pending |
| STT-04 | TBD | Pending |
| STT-05 | TBD | Pending |
| STT-06 | TBD | Pending |
| DCSN-01 | TBD | Pending |
| DCSN-02 | TBD | Pending |
| DCSN-03 | TBD | Pending |
| DCSN-04 | TBD | Pending |
| DCSN-05 | TBD | Pending |
| RESP-01 | TBD | Pending |
| RESP-02 | TBD | Pending |
| RESP-03 | TBD | Pending |
| RESP-04 | TBD | Pending |
| RESP-05 | TBD | Pending |
| RESP-06 | TBD | Pending |
| PRCT-01 | TBD | Pending |
| PRCT-02 | TBD | Pending |
| PRCT-03 | TBD | Pending |
| PRCT-04 | TBD | Pending |
| PLSH-01 | TBD | Pending |
| PLSH-02 | TBD | Pending |
| PLSH-03 | TBD | Pending |
| PLSH-04 | TBD | Pending |
| RSRC-01 | TBD | Pending |
| RSRC-02 | TBD | Pending |
| RSRC-03 | TBD | Pending |
| RSRC-04 | TBD | Pending |

**Coverage:**
- v2.0 requirements: 29 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 29

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-22 after architectural pivot*
