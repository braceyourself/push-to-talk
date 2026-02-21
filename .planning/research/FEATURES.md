# Feature Landscape: Always-On Voice Assistant with Proactive AI

**Domain:** Desktop always-on voice assistant with autonomous participation
**Researched:** 2026-02-21
**Confidence:** MEDIUM-HIGH (strong ecosystem research, Inner Thoughts framework well-documented, Gemini Proactive Audio validates the pattern; specific Llama 3.2 3B latency for this task is LOW confidence -- needs benchmarking)

## Existing Foundation

The system already has:
- 5-stage asyncio pipeline: Audio Capture -> STT (Whisper) -> LLM (Claude CLI) -> TTS (Piper) -> Playback
- Heuristic + semantic input classification (6 categories, <10ms)
- Categorized quick response library with situation-matched audio clips
- StreamComposer for unified audio queue with pre-buffering
- Barge-in interruption with VAD detection
- Configurable idle timeout (0 = always-on, already implemented)
- Personality system (Russel: direct, opinionated, helpful colleague)
- Event bus for cross-component communication (JSONL)
- Tool use (run_command, read_file, spawn_task via Claude CLI + MCP)

The v2.0 transformation keeps all of this intact and adds three new layers: (1) continuous input stream decoupled from LLM, (2) a local LLM monitoring/decision layer, and (3) configurable response backend selection.

---

## Table Stakes

Features the system MUST have for always-on to work at all. Without any one of these, the system is broken or unusable.

### TS-1: Continuous Audio Capture with VAD Gating

**What:** Microphone stays open permanently. Audio flows through Silero VAD to detect speech segments. Only speech segments are sent to Whisper for transcription. Silence/noise is discarded without STT processing.

**Why Required:** Running Whisper continuously on all audio (including silence) would consume 100% GPU and produce hallucinated transcripts from noise. VAD gating is how every always-on system works -- Alexa, Google, Siri all use a lightweight detector before the expensive STT model. The existing barge-in VAD (Silero, already loaded in live_session.py line 1306) proves this works. Silero VAD uses <0.5% CPU for real-time processing (RTF 0.004 on AMD CPU, MIT license, 2MB model).

**Complexity:** LOW -- Silero VAD already exists in the codebase for barge-in. Repurpose it as the always-on gate.

**Dependencies:** None new. Extends existing `_load_vad_model()` and `_run_vad()`.

**Confidence:** HIGH (Silero VAD performance verified via GitHub docs and Picovoice comparison benchmarks)

---

### TS-2: Rolling Transcript Buffer (Context Window)

**What:** Maintain a sliding window of recent transcripts (last N minutes or N tokens) that the monitoring LLM reads to make response decisions. Older transcripts get summarized or dropped. The buffer is the LLM's "working memory" of what's been said.

**Why Required:** The monitoring LLM cannot make good decisions about when to respond without knowing what's been discussed. A single utterance in isolation ("yeah that's true") is meaningless -- the LLM needs the preceding context to know whether it has something to add. Every ambient AI system maintains a conversation buffer: whisper.cpp's sliding window mode keeps context tokens across chunks, LiveKit agents maintain rolling chat history trimmed to token limits, and the Inner Thoughts framework (CHI 2025) uses a full conversation log for thought generation.

**Implementation approach:**
- Ring buffer of (timestamp, speaker, text) tuples
- Configurable window: 5-10 minutes or ~2000 tokens (Llama 3.2 3B context is 128K, but shorter context = faster inference)
- Summarize or drop entries older than the window
- Include speaker attribution where detectable (user vs AI vs "other" if multi-party)

**Complexity:** MEDIUM -- The data structure is simple but tuning the window size and deciding what to summarize vs drop requires experimentation.

**Dependencies:** TS-1 (needs continuous transcripts to buffer)

**Confidence:** HIGH (pattern validated across multiple systems)

---

### TS-3: Response Decision Engine (The "Should I Respond?" Classifier)

**What:** After each speech segment is transcribed, the local LLM (Ollama + Llama 3.2 3B) evaluates the rolling transcript buffer and decides: (a) should Russel respond, and (b) if yes, what kind of response (quick acknowledgment, substantive answer, proactive contribution). Returns a structured JSON decision.

**Why Required:** This is the core of the always-on system. Without it, the AI either responds to everything (annoying) or nothing (useless). The existing input classifier (heuristic + model2vec) classifies WHAT was said but not WHETHER to respond. The monitoring LLM adds the "should I speak?" judgment that incorporates full conversation context.

**Decision factors (derived from Inner Thoughts framework and Gemini Proactive Audio):**
1. **Addressee detection:** Was the utterance directed at Russel? (Name mention, question directed at AI, or generic room conversation)
2. **Relevance score:** Does Russel have something useful to add based on conversation context?
3. **Urgency:** Is this time-sensitive? (e.g., "what time is my meeting" vs casual chat)
4. **Interruption cost:** Is someone else speaking? Is the user focused?
5. **Confidence threshold:** How sure is the LLM that responding would be helpful?

**Output format (structured JSON from Ollama):**
```json
{
  "should_respond": true,
  "confidence": 0.85,
  "response_type": "substantive",
  "reason": "User asked a factual question directed at the room",
  "suggested_approach": "answer directly"
}
```

**Complexity:** HIGH -- This is the hardest feature. The prompt engineering for the monitoring LLM is critical. Too aggressive = annoying. Too conservative = useless. Needs extensive tuning.

**Dependencies:** TS-1 (continuous transcripts), TS-2 (context buffer)

**Confidence:** MEDIUM (Inner Thoughts framework validates the concept, but Llama 3.2 3B's ability to do this reliably at <500ms is unverified -- needs benchmarking)

---

### TS-4: Name-Based Activation ("Hey Russel")

**What:** The AI monitors transcripts for its name ("Russel", "Russell", "hey Russel") as a high-confidence activation signal. Name mentions bypass the confidence threshold -- if you say the name, the AI always responds. Also serves as the interruption mechanism (say "Russel" to cut the AI off mid-response, replacing the current PTT-based barge-in).

**Why Required:** Users need a guaranteed way to get the AI's attention. Proactive participation is probabilistic (the AI might not speak up), but name-based activation must be deterministic. Every always-on assistant has this: "Alexa", "Hey Siri", "OK Google", "Hey Copilot". Research shows users strongly prefer an explicit activation path even when proactive features exist (CHI 2024 study: participants favored utterance starter "Hey, are you available?" before VA initiated conversation).

**Implementation approach:**
- Simple string matching on transcripts (no wake word model needed -- STT is already running)
- Fuzzy match for common misspellings/misheard variants: "Russell", "Rusel", "hey Russ"
- Name detection feeds directly into TS-3 decision engine as a forced-respond signal
- For interruption: name detected during AI playback triggers existing barge-in flow

**Complexity:** LOW -- String matching on existing transcript stream. The barge-in infrastructure already exists.

**Dependencies:** TS-1 (continuous transcripts). Extends existing barge-in system.

**Confidence:** HIGH (trivial string matching, well-understood pattern)

---

### TS-5: Configurable Response Backend (Claude CLI / Ollama)

**What:** After the decision engine says "respond," the system picks which LLM generates the actual response: Claude CLI for complex/tool-using queries, Ollama for quick conversational responses. Selection is automatic based on query complexity, network availability, and expected latency.

**Why Required:** Claude CLI (via the existing pipeline) is powerful but slow (2-5s to first token) and requires network. Ollama is fast (~200ms for short responses) but less capable. Always-on proactive responses need to be fast -- if someone makes a joke and Russel wants to quip back, waiting 3 seconds kills the moment. But for "refactor the auth module," Claude CLI's tool access is essential.

**Selection heuristics:**
- Name mention + simple question -> Ollama (speed)
- Task/tool request -> Claude CLI (capability)
- Proactive contribution to casual conversation -> Ollama (speed, low stakes)
- Network down -> Ollama (only option)
- Explicit "hey Russel, use Claude for this" -> Claude CLI (user override)

**Complexity:** MEDIUM -- The routing logic is straightforward. The complexity is in maintaining two separate response pipelines and ensuring the TTS/playback path works identically for both.

**Dependencies:** TS-3 (decision engine determines response type, which informs backend selection)

**Confidence:** HIGH (Ollama API is well-documented, Claude CLI pipeline already exists)

---

### TS-6: Resource Management for Continuous Operation

**What:** The system must run indefinitely without memory leaks, GPU exhaustion, or CPU runaway. This means: bounded transcript buffer, Whisper model kept warm but not running during silence, Ollama connection pooling, audio buffer overflow protection.

**Why Required:** The current PTT system runs for minutes at a time. Always-on runs for hours or days. Memory leaks that are invisible in a 5-minute session become OOM kills after 8 hours. Whisper on GPU + Ollama on GPU + Piper TTS on CPU is a significant resource footprint.

**Key concerns:**
- Whisper "small" model: ~461MB GPU VRAM (already loaded). Note: PROJECT.md says "small" but push-to-talk.py line 215 uses "large-v3" -- need to confirm which model for always-on.
- Ollama Llama 3.2 3B: ~2GB VRAM (new)
- Total GPU: ~2.5GB VRAM concurrent. Must fit alongside desktop GPU usage.
- Transcript buffer must be bounded (ring buffer, not unbounded list)
- Audio capture buffer overflow detection (existing pattern: whisper.cpp drops audio if buffer > 2x expected)

**Complexity:** MEDIUM -- Mostly engineering discipline (bounded buffers, proper cleanup), but GPU memory management across Whisper + Ollama needs testing.

**Dependencies:** All other features (this is a cross-cutting concern)

**Confidence:** MEDIUM (known patterns, but specific GPU sharing between Whisper large-v3 and Llama 3.2 3B needs testing on the actual hardware)

---

### TS-7: Graceful Degradation

**What:** When components fail, the system degrades gracefully rather than crashing. Ollama down -> fall back to Claude CLI only. Network down -> Ollama only. Whisper overloaded -> drop frames, don't queue indefinitely. GPU OOM -> restart the affected model.

**Why Required:** An always-on system that crashes is worse than no system. The user will lose trust after the second crash and disable it. Cloud assistants (Alexa, Google) handle degradation through massive redundancy. A local system handles it through intelligent fallback chains.

**Fallback chain:**
1. Full system: Silero VAD -> Whisper STT -> Ollama decision -> Claude/Ollama response -> Piper TTS
2. Network down: Silero VAD -> Whisper STT -> Ollama decision -> Ollama response -> Piper TTS
3. Ollama down: Silero VAD -> Whisper STT -> heuristic classifier (existing) -> Claude CLI response -> Piper TTS
4. GPU pressure: Switch Whisper from large-v3 to small or tiny model, reduce quality for survival
5. Total failure: Mute mic, show error in overlay, wait for manual restart

**Complexity:** MEDIUM -- Each fallback path is simple, but testing all combinations is thorough work.

**Dependencies:** TS-5 (backend selection handles part of this)

**Confidence:** HIGH (well-understood engineering patterns)

---

## Differentiators

Features that make this system better than Alexa/Siri/Google/Copilot for its specific use case (desktop developer companion). Not required for basic operation, but these are what make users prefer this.

### D-1: Proactive Conversation Participation

**What:** Russel joins conversations even when not addressed. If two people are discussing a code problem and Russel knows the answer, he speaks up. If someone makes a factual error, Russel corrects it. If someone asks a question to the room, Russel offers his take.

**Value Proposition:** This is the core differentiator from every existing voice assistant. Alexa/Siri/Google only respond when addressed. They are reactive. Russel is a participant -- "like having a knowledgeable colleague in the room." No mainstream consumer product does this. Google's Proactive Audio (Gemini Live API) is the closest, but it only proactively responds to device-directed queries -- it explicitly does NOT respond to background conversation.

**The Inner Thoughts Framework (CHI 2025):**
The most directly relevant research. Defines three participation modes:
- **Overt proactivity:** How often the AI participates (system-1 probability, 0-1)
- **Covert proactivity:** How much motivation it needs to speak (imThreshold, 1-5)
- **Tonal proactivity:** How assertively it speaks

For Russel's "aggressive participation" goal, tune toward:
- High overt proactivity (speak frequently)
- Low covert threshold (low bar to contribute)
- Moderate tonal proactivity (confident but not domineering)

**Eight scoring heuristics for whether to speak:**
1. Relevance to current topic
2. Information gap (does the AI know something others don't?)
3. Expected impact of contributing
4. Urgency
5. Coherence with conversation flow
6. Originality (not just restating what was said)
7. Balance (has the AI been hogging the conversation?)
8. Dynamics (is now a natural entry point?)

**Complexity:** HIGH -- This is the most ambitious feature. The prompt engineering for the Ollama monitoring layer must encode these heuristics. Too aggressive and Russel becomes that coworker who won't shut up. Too conservative and he's just another Alexa.

**Dependencies:** TS-2 (context buffer), TS-3 (decision engine), TS-5 (fast Ollama backend for quick contributions)

**Confidence:** MEDIUM (concept validated in research, but implementing it with a 3B local model vs the large models used in research papers introduces quality risk)

---

### D-2: Conversation-Aware Response Calibration

**What:** The response style and length vary based on conversation context. In a focused work discussion, Russel gives precise technical answers. In casual banter, Russel quips or jokes. If the user sounds frustrated, Russel is measured and empathetic. If the user is excited, Russel matches the energy.

**Value Proposition:** Alexa has one mode. Siri has one mode. Russel adapts. The personality system already encodes this ("Match the user's energy" in 04-voice.md), but currently the classifier only operates on single utterances. With the rolling transcript buffer, the monitoring LLM can detect conversation MOOD and REGISTER over multiple turns.

**Implementation approach:**
- The Ollama decision engine includes a "tone" field in its output: `"tone": "casual"` / `"focused"` / `"supportive"` / `"excited"`
- This tone instruction is passed to the response backend (Claude or Ollama) as a system prompt modifier
- The existing personality files provide the base; tone is a runtime overlay

**Complexity:** LOW-MEDIUM -- Mostly prompt engineering. The infrastructure (decision engine -> response backend) handles the plumbing.

**Dependencies:** TS-3 (decision engine provides tone assessment)

**Confidence:** MEDIUM (prompt engineering quality determines success)

---

### D-3: Non-Speech Event Awareness

**What:** Detect coughs, sighs, laughter, throat clearing, typing sounds, and respond appropriately. A cough gets "bless you" or concerned silence. Laughter might get the AI joining in or asking "what's funny?" A long sigh after a frustrating conversation gets "rough day?"

**Value Proposition:** This is what makes Russel feel like a real presence in the room, not just a chatbot with a microphone. No consumer assistant does this. The existing Whisper STT already detects these events (they show up as rejected segments with high no_speech_prob or specific hallucinated patterns like "[laughter]", "[cough]"). The v1.2 research identified SenseVoice (FunAudioLLM) as capable of explicit non-speech event classification, but the existing Whisper metadata approach is sufficient and avoids a second model.

**Carried forward from v1.2 scope.** This was part of the original v1.2 plan but folded into v2.0.

**Complexity:** MEDIUM -- Leverages existing STT rejection metadata. Needs mapping from rejection patterns to response categories. Integration with the response library.

**Dependencies:** TS-1 (continuous audio), existing STT filtering pipeline, existing response library

**Confidence:** HIGH (existing codebase already detects these events, just currently discards them)

---

### D-4: Post-Session Library Growth (Curator Daemon)

**What:** After a session ends (or during long idle periods), a curator process reviews conversation logs, identifies new response phrases that would have been useful, generates them via Piper TTS, evaluates quality, and adds them to the response library.

**Value Proposition:** The response library grows organically based on actual usage patterns. Over time, Russel's quick responses become more varied and more appropriate to the specific user's communication style. This is the self-improving loop that no consumer assistant has.

**Carried forward from v1.2 scope.** Phase 10 of v1.2 was "library growth." Folded into v2.0.

**Complexity:** MEDIUM -- The clip factory infrastructure already exists. The curator needs conversation analysis logic (which phrases were missing, what should be generated).

**Dependencies:** Existing clip_factory.py, response_library.py, event bus conversation logs

**Confidence:** HIGH (infrastructure exists, this is a new consumer of existing systems)

---

### D-5: Multi-Speaker Awareness

**What:** Distinguish between different speakers in the room. Know when the user is talking vs a guest vs TV audio. Attribute transcript segments to speakers so the context buffer has speaker labels. This enables better "should I respond?" decisions (e.g., respond to user's questions but not to TV dialogue).

**Value Proposition:** Dramatically reduces false activations and inappropriate responses. The biggest complaint about always-on assistants is responding to TV/radio/other people. Amazon's device-directed speech detection uses LSTM classifiers on acoustic + ASR features to achieve 5.2% EER. Gemini Proactive Audio filters out "external chatter" explicitly.

**Implementation approach (tiered):**
- **Tier 1 (simple):** Use Whisper's VAD energy levels + speaking patterns to distinguish "primary user" (consistent voice, close to mic) from "other" (different energy profile, farther away)
- **Tier 2 (better):** Speaker diarization via pyannote.audio or similar -- assigns speaker IDs to segments
- **Tier 3 (best):** Voice enrollment -- user registers their voice, system recognizes them specifically

**Complexity:** LOW for Tier 1, MEDIUM for Tier 2, HIGH for Tier 3

**Dependencies:** TS-1 (continuous audio), TS-2 (context buffer needs speaker labels)

**Confidence:** MEDIUM (Tier 1 is straightforward heuristic, Tier 2 requires pyannote.audio which adds dependencies, Tier 3 is research-grade)

---

### D-6: Attention Signals Before Proactive Responses

**What:** Before Russel proactively interjects, play a brief "attention signal" -- a subtle sound or short verbal cue like "hey" or "actually" or a soft chime. This gives the user a moment to register that Russel is about to speak, preventing the jarring experience of an AI suddenly talking over a conversation.

**Value Proposition:** Research strongly supports this. The CHI 2024 "Better to Ask Than Assume" study found that participants favored the "utterance starter" approach where the VA says "Hey, are you available?" before initiating. For a casual companion like Russel, a full "are you available?" is too formal -- a brief verbal marker or audio cue is more natural.

**Implementation options:**
- Short verbal: "Hey," / "Actually," / "Oh," / "(throat clear)" -- played from the quick response library
- Audio cue: Subtle notification sound (200-500ms) from a pre-generated clip
- Contextual: Vary the signal based on urgency. Low urgency = soft sound. High urgency = verbal "hey"

**Complexity:** LOW -- Uses existing StreamComposer and response library infrastructure. Just needs a new "attention signal" category in the library.

**Dependencies:** D-1 (proactive participation triggers the signal), existing StreamComposer

**Confidence:** HIGH (well-researched pattern, simple implementation)

---

### D-7: Interruptibility Detection

**What:** Detect when the user is likely busy/focused and suppress proactive responses. Signals: user hasn't spoken in a while (deep work), keyboard typing sounds detected, user explicitly said "quiet mode" or "give me a minute."

**Value Proposition:** The biggest risk with proactive AI is annoying the user. Research on proactive VAs (CUI 2022 "Proactivity Dilemma") identifies interruptibility as the key factor: personal context (busyness, mood) matters more than the content being offered. Interestingly, one study found 30 proactive interactions in 2.5 hours caused zero annoyance -- but that was in a cooperative setting. A developer in deep focus would react very differently.

**Implementation approach:**
- **Time-based:** If user hasn't spoken in >10 minutes, assume deep work. Raise the confidence threshold for proactive responses.
- **Explicit:** User says "quiet mode" / "shut up for a bit" -> suppress all non-addressed responses for N minutes.
- **Ambient:** Detect sustained keyboard typing via audio features -> assume coding focus -> raise threshold.
- **OS integration:** Check if a fullscreen app is running (gaming, presentations) -> suppress.

**Complexity:** LOW-MEDIUM -- Time-based and explicit are trivial. Ambient and OS integration add complexity.

**Dependencies:** TS-3 (decision engine respects interruptibility signals)

**Confidence:** HIGH for time-based/explicit, LOW for ambient/OS integration

---

## Anti-Features

Features to explicitly NOT build. These seem valuable but create problems that outweigh benefits.

### AF-1: Hardware Wake Word Detection (Picovoice Porcupine / openWakeWord)

**Why It Seems Good:** Dedicated wake word models are optimized for this exact task. Picovoice Porcupine can create custom wake words "in seconds." openWakeWord runs 15-20 models simultaneously on a Raspberry Pi.

**Why Problematic:** Whisper STT is ALREADY running continuously. Adding a separate wake word model creates two parallel audio processing paths -- one for wake word, one for STT. The wake word model is solving a problem that doesn't exist: "detect speech before running expensive STT." But we're already running STT on everything. Name detection on the transcript ("hey Russel") is simpler, cheaper, and more flexible (handles variations naturally via STT's language model).

**Do This Instead:** TS-4 (name-based activation via transcript string matching). Zero additional models, zero additional CPU/GPU, instant implementation.

**When to revisit:** Only if Whisper continuous STT proves too expensive and we need to gate it behind a wake word.

---

### AF-2: Cloud-Based Monitoring LLM

**Why It Seems Good:** Claude Haiku or GPT-4o-mini would be far more capable at the "should I respond?" decision than Llama 3.2 3B. Higher accuracy means fewer false positives and false negatives.

**Why Problematic:** The monitoring LLM evaluates EVERY speech segment -- potentially dozens per minute in an active conversation. At $0.25/1M input tokens (Haiku), monitoring 8 hours of conversation (~50K tokens/hour) costs ~$0.10/day. Not expensive, but: (1) adds network latency to every decision (200-500ms round-trip), (2) requires internet for basic operation, (3) sends all ambient conversation to cloud servers -- a serious privacy concern for an always-on mic. The local-first philosophy is a core value of this project.

**Do This Instead:** TS-3 (Ollama + Llama 3.2 3B local). If 3B proves inadequate, try Llama 3.2 8B or Mistral 7B locally before going to cloud. The system already runs on a desktop with a capable GPU.

**When to revisit:** Only if local models prove fundamentally incapable of the task after extensive prompt tuning.

---

### AF-3: Full Conversation Transcription Storage / Searchable History

**Why It Seems Good:** "Record everything Russel hears, make it searchable later." Like an always-on meeting recorder.

**Why Problematic:** This is a surveillance feature that happens to be technically easy. Storing continuous transcripts of ambient room audio crosses a privacy line that even the user themselves may not realize until it's too late. Guests don't know they're being recorded. Family members don't know. Even for a single-user system, having a searchable record of every word spoken near your desk is a liability, not a feature.

**Do This Instead:** The rolling transcript buffer (TS-2) is ephemeral -- it holds the last N minutes for context, then drops it. Conversation logs (existing JSONL) only capture DELIBERATE interactions (where the user chose to engage). No ambient logging.

**When to revisit:** Never. This is a values decision, not a technical one.

---

### AF-4: Continuous Full-Duplex Conversation (Speech-to-Speech Model)

**Why It Seems Good:** Models like GPT-4o Realtime and Gemini Live API offer native audio-to-audio with natural turn-taking, backchannels, and overlapping speech. NVIDIA PersonaPlex handles interruptions and conversational rhythm natively.

**Why Problematic:** These are cloud-only, expensive ($5-15/hr for continuous connection), and fundamentally designed for 1:1 conversation -- not ambient monitoring. They assume the model IS the conversation partner, not an observer who occasionally joins. The architecture is incompatible with "listen to a room and decide when to participate." Also: cloud dependency for always-on = reliability nightmare.

**Do This Instead:** The cascaded pipeline (STT -> LLM -> TTS) with the Ollama monitoring layer. Less "natural" turn-taking but compatible with the observer model and fully local.

**When to revisit:** When open-source speech-to-speech models (like Moshi or similar) mature enough to run locally and support observer/monitoring mode.

---

### AF-5: Learning User Schedule / Routine-Based Proactivity

**Why It Seems Good:** Alexa Hunches learns daily patterns (user always asks about weather at 7am) and proactively offers information. The system could learn "Ethan always reviews PRs at 9am" and proactively offer a summary.

**Why Problematic:** This is a different product. Russel is a conversational companion, not a scheduling assistant. Routine-based proactivity requires persistent state across days/weeks, pattern detection algorithms, and a fundamentally different interaction model. It also requires the system to initiate conversation from nothing -- not "I heard something relevant and want to contribute" but "it's 9am so I should say something." That's much harder to get right and much easier to get annoying.

**Do This Instead:** Russel participates proactively in CONVERSATIONS that are happening. He doesn't initiate conversations from silence. If the user wants a morning briefing, they say "hey Russel, what's up today?" The proactivity is reactive-proactive (responds within conversation context), not schedule-proactive (initiates based on time patterns).

**When to revisit:** After the core always-on conversation participation is solid and users request it.

---

### AF-6: Multi-Room / Multi-Device Mesh

**Why It Seems Good:** Run Russel on multiple devices, share context across rooms.

**Why Problematic:** Massive scope expansion. Network synchronization, conflict resolution, echo cancellation across devices, and distributed state management. This is a product category change, not a feature.

**Do This Instead:** One device, one room, one mic. The desktop. Keep scope tight.

---

## Feature Dependencies

```
TS-1: Continuous Audio + VAD
  |
  +---> TS-2: Rolling Transcript Buffer
  |       |
  |       +---> TS-3: Response Decision Engine (Ollama)
  |       |       |
  |       |       +---> TS-5: Backend Selection (Claude/Ollama)
  |       |       |       |
  |       |       |       +---> D-1: Proactive Participation
  |       |       |       |       |
  |       |       |       |       +---> D-6: Attention Signals
  |       |       |       |
  |       |       |       +---> D-2: Response Calibration
  |       |       |
  |       |       +---> D-7: Interruptibility Detection
  |       |
  |       +---> D-5: Multi-Speaker Awareness
  |
  +---> TS-4: Name-Based Activation ("Hey Russel")
  |
  +---> D-3: Non-Speech Event Awareness
  |
  +---> TS-6: Resource Management (cross-cutting)
  |
  +---> TS-7: Graceful Degradation (cross-cutting)

D-4: Library Growth (independent -- post-session, uses existing infrastructure)
```

## MVP Recommendation

For MVP (first shippable version of always-on), prioritize:

1. **TS-1: Continuous Audio + VAD** -- Foundation, everything depends on it
2. **TS-4: Name-Based Activation** -- Deterministic interaction path, low complexity
3. **TS-2: Rolling Transcript Buffer** -- Minimal version: last 3 minutes, no summarization
4. **TS-3: Response Decision Engine** -- Core intelligence, even if initial prompts are rough
5. **TS-5: Backend Selection** -- Start simple: Ollama for everything the decision engine flags, Claude CLI only on explicit name-mention task requests
6. **TS-6: Resource Management** -- Bounded buffers from day one, not retrofitted

Defer to post-MVP:
- **D-1 (Proactive Participation):** Start with respond-when-addressed only. Add proactive gradually by lowering thresholds once the decision engine is proven reliable.
- **D-5 (Multi-Speaker):** Start without speaker diarization. Add if false activations from other people/TV become a problem.
- **D-3 (Non-Speech Events):** Nice-to-have, not blocking.
- **D-4 (Library Growth):** Independent track, can ship whenever.
- **TS-7 (Graceful Degradation):** Basic fallbacks in MVP, comprehensive chain post-MVP.

**The critical path is TS-1 -> TS-2 -> TS-3 -> TS-5.** This is the minimum chain that transforms PTT into always-on.

## Phasing Recommendation

**Phase 1: Infrastructure** (TS-1, TS-2, TS-6)
- Decouple audio capture from LLM processing
- Continuous VAD-gated STT running independently
- Rolling transcript buffer with bounded memory
- System runs but makes no autonomous decisions yet
- Existing PTT interaction still works alongside

**Phase 2: Decision Engine** (TS-3, TS-4)
- Ollama integration for monitoring
- Name-based activation via transcript matching
- Decision engine evaluates transcripts, initially conservative (high threshold)
- Addresses only direct name mentions at first

**Phase 3: Response Backend** (TS-5, TS-7)
- Ollama as response generator for quick responses
- Claude CLI for complex/tool-using requests
- Backend selection logic
- Basic fallback chains

**Phase 4: Proactive Participation** (D-1, D-2, D-6, D-7)
- Lower decision thresholds for proactive contributions
- Attention signals before unsolicited responses
- Conversation-aware tone calibration
- Interruptibility detection

**Phase 5: Polish** (D-3, D-4, D-5)
- Non-speech event awareness
- Library growth curator
- Multi-speaker awareness (if needed)

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Llama 3.2 3B too slow/dumb for decision engine | Blocks TS-3, the critical feature | Benchmark early. Fallback: Llama 3.2 8B, Mistral 7B, or Phi-3 mini. Or: heuristic classifier as fast path with LLM only for ambiguous cases. |
| GPU VRAM exhaustion (Whisper large-v3 + Llama 3.2 3B) | System crashes or slows to crawl | Profile actual VRAM usage. Fallback: Whisper small instead of large-v3 for continuous STT. Or: CPU inference for Ollama. |
| Proactive responses are annoying | Users disable the feature | Start conservative (high threshold). Let user control aggressiveness. D-7 interruptibility detection as safety valve. |
| Whisper hallucinations in continuous mode | False transcripts trigger false responses | Existing multi-layer STT filtering helps. Add minimum speech duration threshold. VAD gating prevents silence hallucinations. |
| Latency budget blown | Responses feel laggy | Budget: VAD (1ms) + Whisper (500ms) + Ollama decision (200ms) + Ollama/Claude response (200ms-3s) + Piper TTS (200ms). Total: ~1-4s. Acceptable for proactive but tight for reactive. Optimize the critical path. |

## Sources

**Academic / Research:**
- [Inner Thoughts: Proactive Conversational Agents (CHI 2025)](https://arxiv.org/html/2501.00383v2) -- Core framework for proactive AI participation with motivation thresholds
- [Better to Ask Than Assume (CHI 2024)](https://dl.acm.org/doi/10.1145/3613904.3642193) -- User preference for VA communication strategies
- [Proactivity Dilemma (CUI 2022)](https://dl.acm.org/doi/10.1145/3543829.3543834) -- When proactive VA behavior is desirable vs annoying
- [Device-Directed Utterance Detection (Amazon Science)](https://arxiv.org/abs/1808.02504) -- Classifier for determining if speech is directed at assistant
- [Proactive Conversational AI: Comprehensive Survey (ACM TOIS)](https://dl.acm.org/doi/10.1145/3715097) -- Broad survey of proactive CA approaches

**Industry / Product:**
- [Gemini Proactive Audio (Google Cloud)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/proactive-audio) -- Google's approach to model-decides-when-to-respond
- [Alexa Hunches](https://www.amazon.com/gp/help/customer/display.html?nodeId=G7F5F7K93GKSLC4F) -- Amazon's pattern-based proactive assistance
- [Microsoft Copilot Voice ("Hey Copilot")](https://support.microsoft.com/en-us/topic/using-copilot-voice-with-microsoft-copilot-efad42fc-d593-49c6-98bf-5ed94c881c32) -- Desktop voice activation with on-device wake word

**Technical / Libraries:**
- [Silero VAD](https://github.com/snakers4/silero-vad) -- Pre-trained VAD, MIT license, <1ms per chunk, 2MB model
- [openWakeWord](https://github.com/dscripka/openWakeWord) -- Open-source wake word detection (noted as anti-feature for this project)
- [Whisper continuous mode (whisper.cpp)](https://github.com/ggml-org/whisper.cpp/issues/304) -- Sliding window context management for continuous STT
- [Ollama Llama 3.2](https://ollama.com/library/llama3.2) -- Local LLM with 128K context, structured output support
- [Building a Fully Local LLM Voice Assistant (Towards AI)](https://towardsai.net/p/machine-learning/building-a-fully-local-llm-voice-assistant-a-practical-architecture-guide) -- Architecture patterns for local voice assistant

**Voice UX:**
- [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) -- Natural conversational AI with backchanneling and interruption handling
- [Wake Word Detection Guide 2026 (Picovoice)](https://picovoice.ai/blog/complete-guide-to-wake-word/) -- FAR/FRR tradeoffs, wake word vs always-listening
- [Turn-Based Voice AI Agents (MLPills)](https://mlpills.substack.com/p/issue-120-turn-based-voice-ai-agents) -- Turn-taking architecture comparison
