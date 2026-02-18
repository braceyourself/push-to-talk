# Feature Research: Adaptive Quick Responses

**Domain:** Context-aware acknowledgment/backchannel system for voice AI assistant
**Researched:** 2026-02-18
**Confidence:** MEDIUM-HIGH (well-researched domain in voice UX; novel for local desktop assistant context)

## Current State

The existing system has a single-category filler pool: 10-15 pre-generated Piper TTS clips of task-oriented phrases ("let me check that", "one sec", "looking into that"). Selection is random with no-repeat guard. A 500ms gate skips the filler if the LLM responds fast. A 300ms gate exists for tool-use acknowledgments.

**The problem:** Every filler sounds task-oriented. "Let me take a look" plays even when the user asks "how are you?" or coughs into the mic. The system has no awareness of *what* the user said or *why* the processing gap exists.

## Feature Landscape

### Table Stakes (Must Have)

Features required for adaptive responses to feel correct. Without these, the system is worse than random selection because users notice *wrong* responses more than *generic* ones.

| Feature | Why Required | Complexity | Dependencies |
|---------|------------|------------|--------------|
| **Intent-based category routing** | Core mechanic: classify user input into categories (question, command, conversational, emotional) and pick fillers from the matching category | MEDIUM | Requires transcript text (already available from STT stage). Needs a classifier. |
| **Multi-category clip pools** | Separate clip sets for different response types: task-oriented ("on it"), conversational ("hmm, good question"), emotional ("oh wow"), etc. | LOW | Extends existing `clip_factory.py` with per-category prompts and directories. |
| **Graceful fallback to random** | If classification fails or takes too long, fall back to current random behavior rather than silence | LOW | None -- this is the existing behavior, just needs to be preserved as fallback. |
| **Classification speed < 200ms** | Filler must play within the 500ms gate window. If classification takes longer than the gate, the user hears silence then a filler -- worse than random. Budget: 200ms for classification, leaving 300ms headroom. | CRITICAL | Constrains classifier choice. Rules out API calls or large models. |
| **Clip variety within categories** | Each category needs 8-15 clips to avoid repetition fatigue. Users notice patterns fast in audio. | LOW | Extends clip_factory.py per-category generation. More prompt templates. |

**Research basis:** Voice UX research consistently shows that awkward or mismatched acknowledgments are worse than no acknowledgment at all. Users have a ~200ms perception threshold for conversational timing (Source: [AssemblyAI latency research](https://www.assemblyai.com/blog/low-latency-voice-ai), [Sierra voice latency](https://sierra.ai/blog/voice-latency)). Google Duplex specifically found that contextually appropriate fillers ("so" to signal information coming, "umm" for hesitation) are perceived as meaningfully different from random insertion (Source: [Google Duplex research blog](https://research.google/blog/google-duplex-an-ai-system-for-accomplishing-real-world-tasks-over-the-phone/)).

### Differentiators (What Makes This Feel Special)

Features that elevate the experience from "functional" to "this assistant gets me." Not strictly required, but these are what make users prefer this over alternatives.

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| **Non-speech event responses** | Detect coughs, sighs, laughter via STT rejection metadata and respond playfully ("excuse you", "you okay?", sympathetic acknowledgment) | MEDIUM | Leverages existing multi-layer STT filtering (no_speech_prob, logprob thresholds). Currently these events are silently rejected -- instead, classify the rejection reason and play appropriate audio. |
| **Emotional tone matching** | Detect excitement, frustration, or seriousness in user text and match filler energy ("oh awesome!", calm "I see", measured "let me look into that") | MEDIUM | Needs sentiment analysis on transcript. Could be a lightweight step in the classifier or a separate heuristic (exclamation marks, keywords like "amazing"/"frustrated"/"help"). |
| **Barge-in trailing fillers by context** | Current barge-in plays a random acknowledgment after interruption. Context-aware version: if user interrupted to correct, play "oh right" / "got it"; if user interrupted because bored, play "sure" / shorter response | LOW-MEDIUM | Barge-in annotation already captures what the AI was saying when interrupted and what the user said. Classification of the interruption reason. |
| **Conversational vs. task mode detection** | "What's your name?" gets "hmm" not "checking now". "Refactor the auth module" gets "on it" not "interesting question" | LOW | This is a natural output of intent classification. The categories themselves encode this. |
| **Learned response preferences** | Track which fillers get interrupted (user talks over them = too long/annoying) and which flow smoothly. Down-weight frequently-interrupted clips. | MEDIUM | Needs to log: which clip played, whether it was cancelled early by barge-in or LLM response. Simple frequency tracking over sessions. |
| **Dynamic TTS fillers** | Instead of only pre-generated clips, occasionally generate a contextual filler on-the-fly: "hmm, let me think about [topic keyword]..." | HIGH | Piper TTS latency (~200-400ms) may blow the timing budget. Would need to fire speculatively before classification completes, or only use for responses where higher latency is acceptable (complex questions). |
| **Silence as a valid response** | For very short inputs ("yes", "no", "ok"), no filler at all -- just fast processing. The 500ms gate partially handles this, but explicit classification of "no-filler-needed" inputs is better. | LOW | Intent classifier outputs a "minimal" category that maps to no filler. |

**Research basis:** Backchanneling research shows that generic vs. specific acknowledgments are perceived differently. "Mm-hm" is generic backchannel; "oh wow" requires surprising content to feel appropriate (Source: [Retell AI on backchanneling](https://www.retellai.com/blog/how-backchanneling-improves-user-experience-in-ai-powered-voice-agents), [Wikipedia: Backchannel (linguistics)](https://en.wikipedia.org/wiki/Backchannel_(linguistics))). NVIDIA PersonaPlex trains on 7,303 real conversations to learn contextual backchannel timing and content (Source: [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/)). Non-speech event detection is well-supported by models like SenseVoice which detect coughing, sneezing, laughter alongside speech (Source: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)).

### Anti-Features (Do NOT Build)

Features that seem appealing but create over-engineering problems or degrade the experience.

| Anti-Feature | Why It Seems Good | Why Problematic | Do This Instead |
|--------------|-------------------|-----------------|-----------------|
| **Full NLU pipeline for classification** | "Proper" intent detection with entity extraction, slot filling, multi-intent support | Massive over-engineering for filler selection. You are not building a dialogue manager -- you are picking from ~6 audio categories. A BERT model or NLU framework adds dependency weight, GPU requirements, and maintenance burden for a problem that needs 5 categories. | Simple heuristic classifier: keyword matching + regex patterns + optional lightweight LLM call as tiebreaker. Google Duplex uses separate small models for different aspects -- don't build one monolithic NLU. |
| **User-configurable response categories** | Let users define their own filler categories and prompts | UI complexity explosion. Users don't think in categories -- they think "make it sound more natural." Configuration surface area that nobody touches after day one. | Opinionated defaults that work well. One toggle: fillers on/off (already exists). Maybe one preference: "personality" slider (formal/casual). |
| **Real-time prosodic/pitch analysis** | Analyze audio features (pitch contour, speaking rate, energy) for emotion detection | Requires audio feature extraction pipeline before STT, adding latency. The transcript text plus STT metadata (confidence, no_speech_prob) already carry sufficient signal. Prosodic analysis is research-grade complexity for marginal gain over text-based heuristics. | Use STT metadata (confidence scores, no_speech_prob) as proxy for audio quality/speech type. Use text-based sentiment heuristics. |
| **Per-user response style profiles** | Learn that "Ethan prefers casual acknowledgments" vs formal | Single-user desktop app. There is one user. Personalization infrastructure for one person is configuration, not ML. | Hardcode the personality in the prompt templates. If the user wants to change it, they edit the personality file (already exists in `personality/`). |
| **A/B testing framework for fillers** | Systematically test which fillers users prefer | You are the only user. Ask yourself. A/B testing a single-user app is absurd overhead. | Manual tuning: listen to sessions, remove clips that sound bad, add ones that sound good. The clip factory already supports pool rotation. |
| **Streaming audio classification** | Classify the audio itself (not transcript) to detect paralinguistic events in real-time | Adds a parallel processing path before STT, duplicating audio handling. The existing Whisper/Deepgram pipeline already extracts the signals needed (no_speech_prob, rejected segments). Building a second audio classifier is redundant. | Classify the STT output and metadata. For non-speech events, use the existing rejection pipeline's metadata (what was rejected and why). |
| **Multi-turn context for filler selection** | Track conversation history to pick fillers based on conversation flow (e.g., "we've been technical for 5 turns, use technical fillers") | The filler plays for 0.5-2 seconds. Nobody notices or cares whether the acknowledgment tracks multi-turn context. The LLM response itself handles conversational context. Over-investing intelligence in a throwaway audio clip. | Single-turn classification is sufficient. The user's most recent utterance is the only context that matters for filler selection. |
| **Cloud API for classification** | Use Claude/GPT API to classify intent for highest accuracy | Adds network latency (200-500ms minimum), API cost per utterance, and a failure mode (API down = no fillers). Classification needs to complete in <200ms. Cloud round-trip alone may exceed this. | Local-only classification. Keyword heuristics, regex, or a tiny local model. Zero network dependency for filler selection. |

## Recommended Categories

Based on research into backchannel linguistics and the existing codebase's needs, here are the recommended filler categories.

### Category Taxonomy

| Category | Trigger | Example Clips | Count Target |
|----------|---------|---------------|--------------|
| **task** | Commands, requests to do something ("refactor this", "run the tests", "find the file") | "On it.", "Working on it.", "Let me take a look.", "Checking now." | 10-12 clips |
| **question** | Questions seeking information ("what is...", "how does...", "why did...") | "Hmm, good question.", "Let me think about that.", "Hmm.", "Oh, interesting." | 10-12 clips |
| **conversational** | Greetings, small talk, opinions ("how are you", "what do you think about", "tell me about yourself") | "Hmm.", "Well...", "Oh.", "Let's see..." | 8-10 clips |
| **emotional** | Excitement, frustration, sharing news ("I just got promoted!", "this is so frustrating", "I love this") | "Oh!", "Oh wow.", "Hmm.", "Ah." | 8-10 clips |
| **acknowledgment** | Confirmations, agreements, "yes/no" responses, short inputs | *No filler* (silence -- input is too simple to warrant acknowledgment, LLM responds fast) | 0 (gate handles this) |
| **non-speech** | Coughs, sighs, throat clears, laughter (detected via STT rejection) | "Excuse you.", "You okay?", "Ha.", *sympathetic hum* | 4-6 clips |

### Classification Strategy

**Recommended approach: Tiered heuristic classifier (no ML required)**

The classification problem is small enough (5-6 categories, single sentence input) that a rule-based approach with keyword matching handles 90%+ of cases. This runs in <1ms, has zero dependencies, and is trivially debuggable.

```
Tier 1: Pattern matching (~0ms, handles 80% of inputs)
  - Starts with question word (what/how/why/when/where/who/can/could/would/is/are/do/does) → question
  - Contains action verbs (refactor/run/find/fix/check/build/deploy/create/update/delete) → task
  - Greeting patterns (hi/hello/hey/good morning/how are you) → conversational
  - Emotional markers (!/amazing/love/hate/frustrated/excited/happy/sad/ugh) → emotional
  - Very short (<3 words, no question mark) → acknowledgment (no filler)

Tier 2: Structural analysis (~0ms, handles 15% more)
  - Ends with "?" → question
  - Imperative sentence structure (starts with verb) → task
  - First/second person + opinion verb (I think/I feel/you should) → conversational

Tier 3: Fallback → task (current behavior, safe default)
```

**Why not ML:** The categories are coarse-grained and the input is a single spoken sentence. Keyword heuristics achieve sufficient accuracy. The risk of misclassification is low-consequence (wrong filler category is mildly awkward, not system-breaking). ML adds latency, dependencies, and maintenance burden for marginal improvement.

**Future upgrade path:** If heuristics prove insufficient, the classifier interface can be swapped to use a local embedding model (sentence-transformers + cosine similarity, <1ms per classification per [this approach](https://medium.com/@durgeshrathod.777/intent-classification-in-1ms-how-we-built-a-lightning-fast-classifier-with-embeddings-db76bfb6d964)) without changing the rest of the system.

## Feature Dependencies

```
Existing Features (already built)
  |
  +-- STT transcript text ──────────> Intent classifier (NEW)
  |                                      |
  +-- STT rejection metadata ──────> Non-speech detector (NEW)
  |                                      |
  +-- clip_factory.py ──────────────> Multi-category clip pools (NEW)
  |                                      |
  +-- _filler_manager() ───────────> Category-aware filler selection (NEW)
  |                                      |
  +-- _pick_filler(category) ──────> Already supports categories (EXTEND)
  |                                      |
  +-- barge-in annotation ─────────> Context-aware trailing filler (NEW)
```

Key dependency chain:
1. Multi-category clip pools must exist before category routing works
2. Intent classifier must be fast enough to fit within the 500ms filler gate
3. Non-speech detection hooks into the existing STT rejection path
4. Barge-in context fillers hook into the existing barge-in annotation system

## MVP Recommendation

**Phase 1 (Core):** Build the minimum to make fillers context-appropriate.

1. **Intent classifier** -- Tiered heuristic (keyword + pattern matching), <1ms
2. **Multi-category clip pools** -- Extend clip_factory.py with category-specific prompts and directories (task, question, conversational)
3. **Category-aware filler selection** -- Modify `_filler_manager()` to classify transcript, then call `_pick_filler(category)`
4. **Clip generation** -- Generate initial pools for each category (10+ clips each)

This addresses the core problem: task-oriented fillers no longer play for conversational questions.

**Phase 2 (Polish):** Add the features that make it feel alive.

5. **Non-speech event responses** -- Hook into STT rejection path, play playful/empathetic clips
6. **Emotional tone matching** -- Add sentiment heuristics to classifier
7. **Silence-as-response** -- Explicit "no filler" for very short/simple inputs
8. **Barge-in context fillers** -- Classify interruption type for trailing filler selection

**Defer to post-milestone:**
- Dynamic TTS fillers (high complexity, marginal value given pre-generated pools work well)
- Learned response preferences (needs session logging infrastructure, premature optimization)
- Multi-turn context (over-engineering for a 1-second audio clip)

## UX Expectations: What Makes Quick Responses Feel Natural

Research synthesis of what users perceive as "natural" vs "robotic" in voice AI acknowledgments.

### Natural Feels Like

| Quality | How to Achieve | Source |
|---------|---------------|--------|
| **Timing matches human conversation** | 200-500ms response window. Filler plays within 500ms of user finishing. Current 500ms gate is well-calibrated. | [AssemblyAI](https://www.assemblyai.com/blog/low-latency-voice-ai) |
| **Variety without randomness** | Same category, different clips. Not the same "let me check" every time, but also not wildly different energy levels between turns. | Voice UX design principles |
| **Appropriate energy level** | Excited input gets excited acknowledgment. Calm input gets calm acknowledgment. Mismatched energy is jarring ("OH WOW" in response to "what time is it"). | [Retell AI](https://www.retellai.com/blog/how-backchanneling-improves-user-experience-in-ai-powered-voice-agents) |
| **Brevity for simple inputs** | "Yes" doesn't need a 2-second "let me think about that." Short inputs get short/no fillers. | [ACM study](https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517684) |
| **Fillers convey meaning** | "So" signals information is coming. "Hmm" signals thinking. "On it" signals action. Not just noise -- each filler sets expectations for what comes next. | [Google Duplex](https://research.google/blog/google-duplex-an-ai-system-for-accomplishing-real-world-tasks-over-the-phone/) |

### Robotic Feels Like

| Quality | What Goes Wrong | Prevention |
|---------|----------------|------------|
| **Wrong context** | "Let me take a look" when user said "how are you" | Intent classification (core feature) |
| **Same clip twice in a row** | User notices the pattern immediately | No-repeat guard (already implemented) |
| **Filler longer than the response** | 3-second filler clip, then 1-second LLM answer | Keep clips short (0.5-2s). The clip_factory already enforces 0.3-4s duration. |
| **Filler that contradicts the response** | "Let me check" then "I don't know" (didn't actually check anything) | Use vague fillers for uncertain categories. "Hmm" is always safe. |
| **Filler after immediate response** | LLM responds in 200ms but filler still plays | 500ms gate (already implemented). Filler cancellation on first LLM text (already implemented). |

## Latency Budget Analysis

Current pipeline timing from user speech end to first audio:

```
User stops speaking
  |
  +-- STT processing: ~100-300ms (Deepgram streaming)
  |
  +-- [NEW: Intent classification]: ~1ms (heuristic) or ~5-50ms (embedding model)
  |
  +-- Filler gate: 500ms wait
  |     |
  |     +-- If LLM responds within 500ms: no filler (skip)
  |     +-- If LLM still processing: play category-appropriate filler
  |
  +-- Filler audio playback: ~500-2000ms
  |
  +-- LLM first token: ~800-2000ms (Claude CLI)
  |
  +-- Filler cancellation on first LLM text
  |
  +-- TTS + playback of actual response
```

The intent classification step must complete before the filler gate opens. With heuristic classification at ~1ms, this adds zero perceptible latency. Even an embedding-based classifier at 50ms fits comfortably.

## Sources

### High Confidence (Official Documentation, Research)
- [Google Duplex Research Blog](https://research.google/blog/google-duplex-an-ai-system-for-accomplishing-real-world-tasks-over-the-phone/) -- Filler design, latency management, context-aware disfluencies
- [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) -- Backchannel training on 7,303 real conversations
- [Wikipedia: Backchannel (linguistics)](https://en.wikipedia.org/wiki/Backchannel_(linguistics)) -- Linguistic framework for generic vs. specific backchannels
- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) -- Non-speech event detection (cough, laugh, sneeze)
- [Deepgram Audio Intelligence](https://deepgram.com/product/audio-intelligence) -- STT-level sentiment, intent, filler word detection

### Medium Confidence (Industry Analysis, Multiple Sources Agree)
- [AssemblyAI: The 300ms Rule](https://www.assemblyai.com/blog/low-latency-voice-ai) -- Latency perception thresholds
- [Sierra: Engineering Low-Latency Voice Agents](https://sierra.ai/blog/voice-latency) -- Filler audio as latency masking
- [Retell AI: Backchanneling](https://www.retellai.com/blog/how-backchanneling-improves-user-experience-in-ai-powered-voice-agents) -- Context-aware acknowledgment selection
- [ACM: Voice Assistant Response Behavior](https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517684) -- User preference for short responses to commands
- [Vaanix: Backchanneling in AI Voice Agents](https://vaanix.ai/blog/what-is-backchanneling-in-ai-voice-agents) -- Technical challenges
- [Intent Classification in <1ms](https://medium.com/@durgeshrathod.777/intent-classification-in-1ms-how-we-built-a-lightning-fast-classifier-with-embeddings-db76bfb6d964) -- Embedding-based fast classification

### Low Confidence (Single Source, Needs Validation)
- [LIDSNet: Lightweight On-Device Intent Detection](https://arxiv.org/pdf/2110.15717) -- Siamese network approach, not verified for this use case
- [Trillet: The High Cost of Silence](https://www.trillet.ai/blogs/high-cost-of-latency) -- Business impact claims (call center context, not desktop app)
