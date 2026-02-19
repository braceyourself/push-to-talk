---
phase: 09-semantic-matching-pipeline-polish
plan: 01
subsystem: classification
tags: [model2vec, semantic-similarity, cosine-similarity, trivial-detection, backchannel, classifier-daemon]

# Dependency graph
requires:
  - phase: 08
    provides: "Heuristic classifier daemon with regex patterns and IPC protocol"
provides:
  - "SemanticFallback class with model2vec potion-base-8M embeddings"
  - "is_trivial() function with ai_asked_question context override"
  - "category_exemplars.json with 6 categories of exemplar phrases"
  - "Enhanced IPC response with match_type and trivial fields"
affects: [09-02, 09-03, 10]

# Tech tracking
tech-stack:
  added: [model2vec, pysbd]
  patterns: ["Background thread model loading for graceful degradation", "Cosine similarity with confidence normalization"]

key-files:
  created: [category_exemplars.json]
  modified: [input_classifier.py, requirements.txt]

key-decisions:
  - "Heuristic tiebreak: 'could you'/'can you'/'would you' framing prefers task over question"
  - "Confidence normalization: cosine>=0.6->0.8-0.9, 0.4-0.6->0.5-0.7, <0.4->0.2-0.4"
  - "CLASSIFIER_READY emitted before semantic model loads (graceful degradation via background thread)"
  - "TRIVIAL_PATTERNS as frozenset for O(1) lookup"

patterns-established:
  - "Background thread model loading: emit ready signal first, load heavy model in daemon thread"
  - "Confidence normalization: map cosine similarity range to heuristic-comparable confidence scale"

# Metrics
duration: 4min
completed: 2026-02-19
---

# Phase 9 Plan 1: Semantic Fallback + Trivial Detection Summary

**model2vec semantic fallback with cosine similarity against category exemplar embeddings, trivial backchannel detection with ai_asked_question context override**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T03:50:40Z
- **Completed:** 2026-02-19T03:55:07Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- SemanticFallback class classifies pattern-free inputs via model2vec potion-base-8M embeddings with cosine similarity against pre-computed category exemplar embeddings
- is_trivial() detects ~30 backchannel phrases ("yes", "mhm", "ok cool") with ai_asked_question context override
- Classifier daemon starts in ~100ms (CLASSIFIER_READY), loads semantic model in background thread (~2-3s)
- IPC response now includes match_type ("heuristic" or "semantic") and trivial boolean fields
- Confidence normalization maps cosine similarity range to heuristic-comparable scale

## Task Commits

Each task was committed atomically:

1. **Task 1: Add semantic fallback + trivial detection** - `8e124dc` (feat)

## Files Created/Modified
- `category_exemplars.json` - 5-10 exemplar phrases per 6 categories for semantic similarity matching
- `input_classifier.py` - Added SemanticFallback class, is_trivial() function, match_type field, background model loading
- `requirements.txt` - Added model2vec and pysbd dependencies

## Decisions Made
- Heuristic tiebreak for "could you"/"can you"/"would you" framing: when question and task patterns tie, prefer task (polite request framing is a directive, not an interrogative)
- Confidence normalization uses linear mapping in three ranges: cosine >= 0.6 maps to 0.8-0.9, 0.4-0.6 maps to 0.5-0.7, < 0.4 maps to 0.2-0.4
- Semantic model loads in daemon background thread after CLASSIFIER_READY, not blocking startup
- TRIVIAL_PATTERNS as frozenset (O(1) lookup) rather than list

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed heuristic question/task tiebreak for polite request framing**
- **Found during:** Task 1 verification
- **Issue:** "could you take a peek at this" matched both question (via "could") and task (via "could you") patterns with equal score. Dict insertion order caused question to win the tie, producing wrong category for a clear task request.
- **Fix:** Added tiebreak logic: when question and task tie and input starts with "can you"/"could you"/"would you", prefer task.
- **Files modified:** input_classifier.py
- **Verification:** "could you take a peek at this" now returns category: task
- **Committed in:** 8e124dc (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix necessary for correctness of the plan's own must_have truth. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Semantic fallback and trivial detection are ready for integration
- StreamComposer (09-02) can proceed independently
- Pipeline integration (09-03) will wire these into live_session.py

---
*Phase: 09-semantic-matching-pipeline-polish*
*Completed: 2026-02-19*
