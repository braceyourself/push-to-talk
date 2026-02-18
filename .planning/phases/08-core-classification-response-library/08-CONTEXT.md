# Phase 8: Core Classification + Response Library - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace random filler selection with heuristic classification + categorized clip library. Users hear contextually appropriate quick responses instead of random acknowledgments — task commands get "on it", questions get "hmm", greetings get "hey". Classification runs as its own daemon process. Seed library generated on first launch.

</domain>

<decisions>
## Implementation Decisions

### Category taxonomy
- 6 categories: task, question, conversational, social, emotional, acknowledgment
- Emotional category has sentiment sub-pools: frustration, excitement, gratitude, sadness
- Acknowledgment is dual-purpose: matches short affirmations ("yes", "ok", "got it") AND serves as fallback for low-confidence classifications
- Social category handles greetings/farewells separately from conversational (casual chat)

### Clip phrases & tone
- Overall personality: chill assistant — relaxed, casual, like a friend helping out
- Task clips: mix of brief acknowledgment ("gotcha", "got it") and action-confirming ("on it", "sure thing", "let me take a look")
- Emotional clips: keep short — single words or very short phrases ("ugh", "nice", "wow", "aw thanks") where flat Piper TTS delivery sounds natural
- Emotional enhancement (expressive TTS, longer emotional phrases) deferred to future milestone

### Classifier architecture
- Classification runs as its own daemon process, not inline in `_filler_manager()`
- Receives STT text at the same time as other pipeline daemons
- IPC mechanism: Claude's discretion — must fit within the 500ms filler gate
- Best-guess matching: use top category even at moderate confidence, don't require high confidence to pick a category-specific clip
- Monitor and tune thresholds over time via Phase 10's curator daemon (same daemon handles both library growth and classification tuning)

### Classification logging
- Full classification trace logged from day one (Phase 8, not deferred to Phase 10)
- Every classification records: input text, category chosen, confidence score, which clip played, whether user barged in
- Logs live in the session log file (~/Audio/push-to-talk/sessions/<timestamp>/) alongside other session data
- Data accumulates so Phase 10's curator has real usage data to work with

### Seed library
- 5-8 clips per category (~40 total)
- Generated on first launch via Piper (not committed as WAV files to repo) — only phrase list committed
- All emotional sub-pools seeded (frustration, excitement, gratitude, sadness — 2-3 clips each)
- Start fresh — do NOT migrate existing ack_pool.json clips. Generate consistent new clips across all categories.

### Claude's Discretion
- IPC mechanism for classifier daemon (pipes, sockets, shared memory — whatever fits the timing constraint)
- Exact confidence threshold values (tune based on testing)
- Specific phrase lists for each category (follow the chill assistant personality)
- Classification pattern/keyword design for heuristic matching
- How the classifier daemon integrates with the existing asyncio pipeline

</decisions>

<specifics>
## Specific Ideas

- Chill assistant vibe throughout — "sure thing", "hmm", "hey", "gotcha", not "absolutely" or "certainly"
- Emotional clips should be very short since Piper can't do emotional expressivity — word choice carries the emotion, not prosody
- Classifier daemon should follow the same subprocess patterns as other daemons in the codebase (like learner.py)

</specifics>

<deferred>
## Deferred Ideas

- Emotional TTS enhancement (expressive voice for emotional clips) — future milestone
- Classification tuning automation — Phase 10 (curator daemon)
- Semantic matching via model2vec — Phase 9
- Silence-as-response for trivial inputs — Phase 9

</deferred>

---

*Phase: 08-core-classification-response-library*
*Context gathered: 2026-02-18*
