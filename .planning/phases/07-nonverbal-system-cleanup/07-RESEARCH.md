# Phase 7: Nonverbal System Cleanup - Research

**Researched:** 2026-02-18
**Domain:** Code cleanup, dead code removal, documentation correction
**Confidence:** HIGH

## Summary

This phase is a targeted cleanup operation, not a feature build. Phase 6 dropped the nonverbal filler system (Piper TTS can't produce natural hums/breaths) and replaced it with acknowledgment phrases, but left behind orphaned code, unused assets, and one broken consumer. The milestone audit identified all the specific locations.

The changes are entirely mechanical: one function call argument fix, removal of dead constants/functions/files, docstring/comment updates, and a requirement documentation correction. No new libraries, no architecture changes, no design decisions needed.

**Primary recommendation:** Execute as a single plan with 4 discrete tasks (fix barge-in, clean clip_factory.py, delete nonverbal assets, update docs), each independently verifiable.

## Standard Stack

No new libraries or tools needed. This phase modifies existing Python files and deletes unused files.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.x (existing) | All modified files are Python | Already in use |
| git | (existing) | File deletion tracking, commit | Already in use |

### Supporting
None needed.

### Alternatives Considered
None -- this is a cleanup phase, not a technology choice.

## Architecture Patterns

### Change Map

The changes touch 4 locations in 3 files plus filesystem deletion and doc updates:

```
Code Changes:
├── live_session.py
│   ├── Line 260: docstring says "non-verbal" → update to "acknowledgment"
│   └── Line 1830: _pick_filler("nonverbal") → _pick_filler("acknowledgment")
├── clip_factory.py
│   ├── Lines 2-9: Module docstring mentions "Non-verbal fillers" → remove
│   ├── Line 36: CLIP_DIR constant (nonverbal path) → REMOVE
│   ├── Line 37: POOL_META constant (pool.json path) → REMOVE
│   ├── Lines 39-40: POOL_SIZE_CAP, MIN_POOL_SIZE → REMOVE
│   ├── Lines 44-45: PROMPTS list (nonverbal prompts) → REMOVE
│   ├── Lines 81-88: random_synthesis_params() → REMOVE
│   ├── Lines 140-178: evaluate_clip() default category param → change to "acknowledgment"
│   ├── Lines 219-221: save_clip() (nonverbal-specific) → REMOVE
│   ├── Lines 245-252: load_pool_meta/save_pool_meta (nonverbal) → REMOVE
│   ├── Lines 277-279: rotate_pool() (nonverbal-specific) → REMOVE
│   ├── Lines 362-365: top_up_pool() → REMOVE
│   ├── Lines 378-386: daemon_mode() → remove top_up_pool() call
│   ├── Lines 394-417: main() → remove top_up_pool() call, update description
│   └── Rename constants: ACK_CLIP_DIR/ACK_POOL_META become primary names
└── .planning/REQUIREMENTS.md
    └── FILL-02: Update text to reflect acknowledgment phrase behavior

File Deletions:
├── audio/fillers/nonverbal/  (10 WAV files, 268KB, git-tracked)
└── audio/fillers/pool.json   (git-tracked metadata for nonverbal pool)
```

### Pattern: Surgical Dead Code Removal

**What:** Remove orphaned nonverbal code while preserving the working acknowledgment system
**When to use:** When a subsystem has been replaced but the old code was not fully cleaned up
**Key principle:** Each removal must be independently verifiable -- after removing nonverbal constants and functions, the acknowledgment system must still work identically.

### Anti-Patterns to Avoid
- **Over-generalizing the cleanup:** Do NOT refactor the acknowledgment system while cleaning up. The acknowledgment system works correctly. Touch it only where the nonverbal code intersects it (e.g., daemon_mode calls both top_up_pool and top_up_ack_pool).
- **Renaming ACK_* constants to generic names:** Tempting but creates unnecessary diff noise. Keep ACK_CLIP_DIR, ACK_POOL_META etc. as-is. They are clear and descriptive.

## Don't Hand-Roll

Not applicable -- this phase removes code, it doesn't build anything.

## Common Pitfalls

### Pitfall 1: Breaking the Acknowledgment System During Cleanup
**What goes wrong:** Accidentally removing shared code that both nonverbal and acknowledgment systems use.
**Why it happens:** clip_factory.py has generic helpers (_top_up, _load_meta, _save_meta, _rotate_pool, generate_clip, evaluate_clip, _next_filename, save_clip_to) used by both systems.
**How to avoid:** Only remove functions/constants that are EXCLUSIVELY nonverbal:
- REMOVE: `CLIP_DIR`, `POOL_META`, `POOL_SIZE_CAP`, `MIN_POOL_SIZE`, `PROMPTS`, `random_synthesis_params()`, `save_clip()`, `load_pool_meta()`, `save_pool_meta()`, `rotate_pool()`, `top_up_pool()`
- KEEP: `_top_up()`, `_load_meta()`, `_save_meta()`, `_rotate_pool()`, `generate_clip()`, `evaluate_clip()`, `_next_filename()`, `save_clip_to()`, `MAX_CONSECUTIVE_FAILURES`, `PIPER_CMD`, `PIPER_MODEL`, `SAMPLE_RATE`
**Warning signs:** `top_up_ack_pool()` failing after cleanup = shared helper was removed.

### Pitfall 2: Forgetting to Update daemon_mode() and main()
**What goes wrong:** `top_up_pool()` is removed but daemon_mode() and main() still call it.
**Why it happens:** These functions call both `top_up_pool()` and `top_up_ack_pool()`.
**How to avoid:** Update daemon_mode() (line 383) and main() (line 412) to only call `top_up_ack_pool()`.
**Warning signs:** NameError on `top_up_pool` when clip factory runs.

### Pitfall 3: Git-tracked Nonverbal Files Left Behind
**What goes wrong:** Files deleted from disk but not staged for git deletion, or pool.json is gitignored but nonverbal WAVs are tracked.
**Why it happens:** The .gitignore has `!audio/fillers/**/*.wav` which explicitly tracks filler WAVs.
**How to avoid:** Use `git rm -r audio/fillers/nonverbal/` and `git rm audio/fillers/pool.json` to both delete and stage.
**Warning signs:** `git status` still showing the files after cleanup.

### Pitfall 4: evaluate_clip() Default Parameter
**What goes wrong:** `evaluate_clip()` has `category: str = "nonverbal"` as default. After cleanup, "nonverbal" is meaningless but the function is still used by the generic `_top_up()` helper which passes category explicitly.
**Why it happens:** The function signature has a stale default.
**How to avoid:** Change the default to `"acknowledgment"` since that is now the only category.
**Warning signs:** Not a runtime issue (callers pass category explicitly) but a maintenance hazard.

## Code Examples

### Fix 1: Barge-in Trailing Filler (live_session.py:1830)

**Before:**
```python
# 9. Play a trailing non-verbal filler clip for naturalness
if self.fillers_enabled:
    clip = self._pick_filler("nonverbal")
```

**After:**
```python
# 9. Play a trailing acknowledgment filler clip for naturalness
if self.fillers_enabled:
    clip = self._pick_filler("acknowledgment")
```

### Fix 2: _spawn_clip_factory Docstring (live_session.py:260)

**Before:**
```python
def _spawn_clip_factory(self):
    """Spawn the clip factory to top up the non-verbal filler pool."""
```

**After:**
```python
def _spawn_clip_factory(self):
    """Spawn the clip factory to top up the acknowledgment filler pool."""
```

### Fix 3: clip_factory.py After Cleanup

The file should retain only:
- Module docstring: updated to describe acknowledgment clip generation only
- Shared constants: PIPER_CMD, PIPER_MODEL, SAMPLE_RATE, MAX_CONSECUTIVE_FAILURES
- ACK constants: ACK_CLIP_DIR, ACK_POOL_META, ACK_POOL_SIZE_CAP, ACK_MIN_POOL_SIZE, ACKNOWLEDGMENT_PROMPTS
- Shared functions: generate_clip(), evaluate_clip() (default="acknowledgment"), _next_filename(), save_clip_to(), _load_meta(), _save_meta(), _rotate_pool()
- ACK functions: random_ack_params(), top_up_ack_pool()
- Entry points: daemon_mode() (calling only top_up_ack_pool), main()

### Fix 4: FILL-02 Requirement Update

**Before:**
```markdown
- [x] **FILL-02**: Live session uses only non-verbal canned audio clips as fillers (breaths, hums, etc.)
```

**After:**
```markdown
- [x] **FILL-02**: Live session uses only acknowledgment phrase audio clips as fillers (e.g., "let me check that", "one sec")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Nonverbal hums/breaths via Piper | Acknowledgment phrases via Piper | Phase 6 (2026-02-18) | Piper can't produce natural nonverbal interjections with lessac-medium model |
| Two filler categories (nonverbal + acknowledgment) | Single category (acknowledgment) | Phase 6 (2026-02-18) | Nonverbal code is orphaned, wastes CPU on session start |
| `_pick_filler("nonverbal")` in barge-in | Returns None (broken) | Phase 6 (2026-02-18) | Trailing filler silently skipped, interruptions sound abrupt |

## Inventory of Changes

Complete enumeration of every change needed, verified against codebase:

### live_session.py (2 changes)
1. **Line 260:** Docstring `"non-verbal"` -> `"acknowledgment"`
2. **Line 1828-1830:** Comment + `_pick_filler("nonverbal")` -> `_pick_filler("acknowledgment")`

### clip_factory.py (remove nonverbal-only items)
Constants to REMOVE:
1. `CLIP_DIR` (line 36)
2. `POOL_META` (line 37)
3. `POOL_SIZE_CAP` (line 39)
4. `MIN_POOL_SIZE` (line 40)
5. `PROMPTS` (lines 44-45)

Functions to REMOVE:
6. `random_synthesis_params()` (lines 81-88)
7. `save_clip()` (lines 219-221)
8. `load_pool_meta()` (lines 245-247)
9. `save_pool_meta()` (lines 250-252)
10. `rotate_pool()` (lines 277-279)
11. `top_up_pool()` (lines 362-365)

Functions to UPDATE:
12. `evaluate_clip()` (line 140): default param `"nonverbal"` -> `"acknowledgment"`, remove nonverbal branch from docstring
13. `daemon_mode()` (line 383): remove `top_up_pool()` call
14. `main()` (line 412): remove `top_up_pool()` call, update argparse description

Docstrings/comments to UPDATE:
15. Module docstring (lines 2-9): remove nonverbal references

### Filesystem deletions (git rm)
16. `audio/fillers/nonverbal/` directory (10 WAV files, all git-tracked)
17. `audio/fillers/pool.json` (git-tracked)

### Documentation updates
18. `.planning/REQUIREMENTS.md` line 52: FILL-02 text update
19. `.planning/STATE.md`: Update decisions to remove nonverbal references

## Open Questions

None. All changes are well-defined and verified against the codebase. The milestone audit document provides a complete inventory that has been cross-referenced with the actual code.

## Sources

### Primary (HIGH confidence)
- `clip_factory.py` -- Read in full, all line numbers verified
- `live_session.py` -- Read in full, line 1830 confirmed as `_pick_filler("nonverbal")`
- `v1.1-MILESTONE-AUDIT.md` -- Complete gap inventory verified against code
- `.gitignore` -- Confirmed `!audio/fillers/**/*.wav` means nonverbal WAVs are git-tracked
- `git ls-files` -- Confirmed all 10 nonverbal WAVs and pool.json are tracked
- `.planning/REQUIREMENTS.md` -- FILL-02 current wording confirmed
- `audio/fillers/` directory listing -- Confirmed 10 nonverbal clips (268KB) + pool.json exist

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no external dependencies, pure code cleanup
- Architecture: HIGH -- changes are surgical and well-enumerated from milestone audit
- Pitfalls: HIGH -- verified shared vs. nonverbal-only code by reading clip_factory.py in full

**Research date:** 2026-02-18
**Valid until:** Indefinite (cleanup of existing code, no external dependencies to go stale)
