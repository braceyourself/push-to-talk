# Phase 1: Mode Rename and Live Voice Session - Research

**Researched:** 2026-02-13
**Domain:** OpenAI Realtime API, GTK3 overlay widgets, Python async voice session management
**Confidence:** HIGH (codebase analysis) / MEDIUM (API specifics)

## Summary

Phase 1 has two distinct workstreams: (A) renaming the existing "live" dictation mode to "dictate" across all code, config, UI, and voice commands, and (B) building a new "live" mode that establishes an OpenAI Realtime API voice session with PTT-driven conversation, session memory, overlay widget, and personality system.

The rename is a straightforward search-and-replace across three files (`push-to-talk.py`, `indicator.py`, and the website pages per the SOP). The new live mode builds on the existing `RealtimeSession` class in `openai_realtime.py` but requires significant rework: stripping tools (Phase 1 is pure voice), adding conversation context management, building a personality system prompt from multiple files, creating a GTK3 overlay widget, and implementing session lifecycle (auto-start on mode select, idle timeout, reconnect on PTT).

**Primary recommendation:** Rewrite `openai_realtime.py` as a dedicated `LiveSession` class that manages the WebSocket connection, conversation state (with sliding window + summarization), and personality prompt. Build the overlay as a new GTK3 window class in `indicator.py`. Use `semantic_vad` for turn detection. Use the `gpt-realtime` model endpoint.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| websockets | latest | WebSocket connection to OpenAI Realtime API | Already in use, required for Realtime API |
| openai | latest | Summarization calls (gpt-4o-mini for context compaction) | Already in use for TTS |
| GTK 3 (gi.repository) | 3.x | Overlay widget with waveform visualization | Already the UI framework for indicator.py |
| Cairo | system | Custom drawing for waveform and status dot | Already used for indicator dot rendering |
| numpy | latest | Audio level detection for waveform visualization | Already in requirements.txt |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| asyncio | stdlib | Event loop for WebSocket + audio I/O | Core of Realtime session management |
| json | stdlib | Conversation state persistence to disk | Saving/loading session history between reconnects |
| threading | stdlib | Running async session from synchronous PushToTalk | Same pattern as existing realtime mode |
| pathlib | stdlib | Session storage directories | File path management |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw websockets | OpenAI Agents SDK RealtimeSession | Agents SDK adds abstraction but hides control needed for PTT timing, mic muting, and manual audio buffer management. Raw websockets match the existing codebase pattern. |
| gpt-4o-mini for summarization | In-session summarization via Realtime API | Using a separate cheap model for context compaction is more cost-effective than burning Realtime API tokens on summarization |
| semantic_vad | server_vad | semantic_vad understands sentence completion, reducing interruptions. Use semantic_vad with eagerness "medium" for conversational flow |

**Installation:**
No new dependencies required. All libraries already in `requirements.txt`.

## Architecture Patterns

### Recommended File Changes
```
push-to-talk.py          # Rename "live" -> "dictate", add new "live" mode routing
openai_realtime.py       # Rewrite: LiveSession class (no tools), ConversationState
indicator.py             # Add LiveOverlayWidget, update Settings combo, update voice commands
personality/             # NEW directory: multi-file personality system
  core.md                # Base personality traits (modeled after clawdbot SOUL.md)
  voice-style.md         # Concise, punchy, fillers allowed
  context.md             # Auto-populated with recent session summaries
```

### Pattern 1: LiveSession Class (replaces RealtimeSession for new mode)
**What:** A new class managing the Realtime API WebSocket with conversation persistence, personality injection, and no tool support.
**When to use:** When user selects "Live" mode in settings or PTT triggers in live mode.
**Example:**
```python
# Architecture based on existing RealtimeSession pattern + OpenAI Cookbook context summarization
class ConversationState:
    """Tracks conversation turns for context management."""
    def __init__(self):
        self.history: list[dict] = []  # {role, item_id, text}
        self.summary_count: int = 0
        self.latest_tokens: int = 0
        self.summarizing: bool = False

class LiveSession:
    """OpenAI Realtime voice session with personality and memory."""

    def __init__(self, api_key, voice="ash", on_status=None):
        self.api_key = api_key
        self.voice = voice
        self.on_status = on_status or (lambda s: None)
        self.ws = None
        self.running = False
        self.conversation = ConversationState()
        self.personality_prompt = self._build_personality()

    def _build_personality(self):
        """Load personality from multiple .md files."""
        parts = []
        personality_dir = Path(__file__).parent / "personality"
        for md_file in sorted(personality_dir.glob("*.md")):
            parts.append(md_file.read_text())
        return "\n\n".join(parts)

    async def connect(self):
        """Connect and configure session with personality."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-realtime",
            additional_headers=headers,
            ping_interval=20,
            max_size=None
        )
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.personality_prompt,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "medium",
                    "interrupt_response": True
                },
                "tools": [],  # No tools in Phase 1
                "tool_choice": "none"
            }
        }))
```

### Pattern 2: Conversation Context Seeding on Reconnect
**What:** When reconnecting after idle timeout, replay conversation summary as a system message.
**When to use:** On every reconnection or session start that has prior context.
**Example:**
```python
# Source: OpenAI Cookbook - Context Summarization with Realtime API
async def seed_context(self):
    """Inject conversation summary as first message on connect."""
    summary = self._get_session_summary()
    if summary:
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "previous_item_id": "root",
            "item": {
                "id": f"summary_{self.conversation.summary_count}",
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": summary}]
            }
        }))
```

### Pattern 3: Personality System (Multi-File, Modeled After Clawdbot)
**What:** Multiple markdown files compose the AI's personality, loaded and concatenated into the session instructions.
**When to use:** On every session connect.
**Details from clawdbot investigation:**

The clawdbot system uses a workspace at `~/clawd/` with these personality files:
- `IDENTITY.md` - Name, creature type, vibe, emoji (short, 10 lines)
- `SOUL.md` - Core behavioral rules, boundaries, execution patterns (long, detailed)
- `USER.md` - Information about the human (name, working style, preferences)
- `MEMORY.md` - SOPs and learned behaviors

For push-to-talk's live mode, adapt this to a simpler structure:
```
personality/
  core.md        # Identity + soul (who you are, how you behave)
  voice-style.md # Voice-specific rules (concise, fillers OK, no markdown)
  context.md     # Auto-managed: recent session summaries, learned preferences
```

The `core.md` should encode the decisions from CONTEXT.md:
- Direct and opinionated
- Dry humor and wit
- Task-oriented memory (connects dots between topics)
- Friendly but efficient, stays on topic
- Short punchy responses, few sentences max
- Fillers allowed (signals listening/processing)

### Pattern 4: Overlay Widget
**What:** A floating, draggable GTK3 window showing live session status.
**When to use:** Visible whenever live mode is active.
**Implementation approach:**
```python
class LiveOverlayWidget(Gtk.Window):
    """Floating overlay for live session status."""

    def __init__(self):
        super().__init__(type=Gtk.WindowType.TOPLEVEL)
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_accept_focus(False)
        self.set_type_hint(Gdk.WindowTypeHint.DOCK)

        # Enable transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)
        self.set_app_paintable(True)

        # Status: green=listening, blue=speaking, gray=idle/disconnected
        self.status = "idle"
        self.waveform_data = []  # Rolling buffer of audio levels

        # Drawing area for waveform
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect("draw", self.on_draw)
        self.add(self.drawing_area)

        # Default size: compact bar
        self.set_default_size(200, 50)
```

Cairo-based waveform: Draw a simple oscillating line using audio RMS levels from the input buffer. Update at 30fps via `GLib.timeout_add(33, self.queue_draw)`. The existing codebase already uses Cairo for the status dot, so this pattern is proven.

### Anti-Patterns to Avoid
- **Don't use server_vad for conversation:** `server_vad` triggers on silence duration alone and interrupts during natural pauses. Use `semantic_vad` which understands sentence completion.
- **Don't send audio while AI is speaking:** The existing codebase has echo cancellation via mic muting. Preserve this pattern but use the Realtime API's built-in `interrupt_response: true` instead of manual cancel logic.
- **Don't store conversation as audio:** Store text transcripts only. Audio is ephemeral (streamed via aplay). Disk-backed history uses text.
- **Don't build a single monolithic prompt:** Use the multi-file personality system. Easier to tune individual aspects without rewriting the whole prompt.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Context window management | Manual token counting | Track `response.usage.total_tokens` from `response.done` events | The API tells you exactly how many tokens are used |
| Context summarization | Regex-based transcript compaction | gpt-4o-mini API call to summarize older turns | LLM summarization preserves meaning; regex loses nuance |
| Turn detection | Manual silence detection timer | `semantic_vad` mode in Realtime API | Semantic understanding of when user is done speaking |
| Audio playback | Custom audio pipeline | `aplay -r 24000 -f S16_LE -t raw -q` (existing pattern) | Already proven in codebase, handles PCM16 format |
| Mic echo cancellation | Software echo cancellation | `pactl set-source-mute` (existing pattern) + API's `interrupt_response` | Hardware-level muting is more reliable than software filtering |
| Reconnection backoff | Custom timer logic | asyncio sleep with exponential backoff (2^n seconds, max 30s) | Standard pattern, simple to implement |

**Key insight:** The OpenAI Realtime API handles turn detection, audio encoding/decoding, and response generation. The client only needs to manage the WebSocket connection, audio I/O, and conversation state persistence.

## Common Pitfalls

### Pitfall 1: Session Timeout Without Graceful Reconnect
**What goes wrong:** Realtime API sessions have a 60-minute maximum duration. If the session quietly dies (network blip, API timeout), the user gets silence with no feedback.
**Why it happens:** WebSocket connections can drop without triggering the `ConnectionClosed` exception if the network degrades slowly.
**How to avoid:** Implement a heartbeat check: if no server event received in 30 seconds, send a ping. On disconnect, set status to "disconnected" (gray dot), attempt reconnect with backoff on next PTT press.
**Warning signs:** `playing_audio` flag stuck true, no events received, aplay subprocess exits.

### Pitfall 2: Context Window Overflow Mid-Conversation
**What goes wrong:** After many turns, the 28,672-token automatic truncation drops important earlier context without warning. The AI suddenly "forgets" what was discussed.
**Why it happens:** The Realtime API drops oldest messages when the window fills. No notification is sent.
**How to avoid:** Track `response.usage.total_tokens` from `response.done` events. When it exceeds ~20,000 tokens, trigger proactive summarization: summarize all but the last 3 turns into a system message, delete the summarized turns.
**Warning signs:** AI repeats questions or contradicts earlier statements.

### Pitfall 3: Mic Muting Race Condition
**What goes wrong:** AI audio bleeds into mic input, creating echo or feedback. Or mic unmutes too early and captures the tail of AI speech.
**Why it happens:** The existing codebase uses a 1.5s delayed unmute after `response.done`, but speaker audio may still be playing (buffered in aplay).
**How to avoid:** Use `response.audio.done` event (not `response.done`) as the signal that audio finished playing. Add a small additional buffer (0.5s) after the last audio chunk before unmuting.
**Warning signs:** AI responds to its own speech, conversation loops.

### Pitfall 4: Config Migration Breaking Existing Users
**What goes wrong:** Renaming `"live"` to `"dictate"` in the default config breaks users who have `"dictation_mode": "live"` saved in their `config.json`. After update, they get an unrecognized mode.
**Why it happens:** The default merging in `load_config()` doesn't migrate old values.
**How to avoid:** Add a migration step in `load_config()`: if `dictation_mode` is `"live"`, automatically change it to `"dictate"` and save.
**Warning signs:** Users report dictation stopped working after update.

### Pitfall 5: GTK Thread Safety Violation
**What goes wrong:** Updating the overlay widget from the asyncio thread (which runs the Realtime session) crashes GTK.
**Why it happens:** GTK is not thread-safe. Widget updates must happen on the main GTK thread.
**How to avoid:** Use `GLib.idle_add()` to schedule widget updates from the async thread. The existing codebase already uses `set_status()` which writes to a file read by GTK poll -- but the overlay needs direct updates for waveform rendering.
**Warning signs:** Segfaults, "Gtk-CRITICAL" warnings in journal, frozen UI.

### Pitfall 6: Model Endpoint Using Stale Preview
**What goes wrong:** The existing code uses `gpt-4o-realtime-preview-2024-12-17` which is a preview model. It may be deprecated.
**Why it happens:** The codebase was written before GA release.
**How to avoid:** Use the GA model endpoint: `wss://api.openai.com/v1/realtime?model=gpt-realtime`. This is the production-grade, generally available model.
**Warning signs:** 404 or deprecation warnings from the API.

## Code Examples

### Rename All "live" References in Dictation Mode
```python
# push-to-talk.py - load_config() defaults
# BEFORE:
"dictation_mode": "live",  # "live", "prompt", or "stream"

# AFTER:
"dictation_mode": "dictate",  # "dictate", "prompt", or "stream"

# Migration in load_config():
config = {**default, **stored}
if config.get('dictation_mode') == 'live':
    config['dictation_mode'] = 'dictate'
    save_config(config)
```

### Voice Commands Update
```python
# push-to-talk.py - voice command section
# BEFORE:
if text_lower in ['go live', 'live mode', 'going live']:
    self.config['dictation_mode'] = 'live'

# AFTER:
if text_lower in ['dictate mode', 'go dictate', 'dictation mode']:
    self.config['dictation_mode'] = 'dictate'

# NEW: "live mode" now activates the new live voice session
if text_lower in ['live mode', 'go live', 'going live']:
    self.config['ai_mode'] = 'live'
    save_config(self.config)
    # Trigger live session start
```

### Settings UI Combo Box Update
```python
# indicator.py - SettingsWindow.create_general_tab()
# Dictation Mode combo:
# BEFORE:
self.mode_combo.append("live", "Live (instant typing)")

# AFTER:
self.mode_combo.append("dictate", "Dictate (instant typing)")

# AI Mode combo - add Live option:
self.ai_mode_combo.append("live", "Live (Voice Conversation)")
```

### Session Lifecycle in PushToTalk
```python
# push-to-talk.py - on_ai_mode_changed or on_press handler
# Session auto-starts when Live mode selected in settings
def start_live_session(self):
    """Start a live voice conversation session."""
    api_key = get_openai_api_key()
    if not api_key:
        prompt_api_key()
        return

    self.live_session = LiveSession(
        api_key=api_key,
        voice=self.config.get('openai_voice', 'ash'),
        on_status=set_status
    )

    def run_session():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.live_session.run())
        finally:
            loop.close()
            self.live_session = None

    self.live_thread = threading.Thread(target=run_session, daemon=True)
    self.live_thread.start()

def stop_live_session(self):
    """Stop the live session cleanly."""
    if self.live_session:
        self.live_session.stop()
        # Session says "Session ended" via TTS before disconnecting
```

### Context Summarization Trigger
```python
# Source: OpenAI Cookbook context_summarization_with_realtime_api
SUMMARY_TRIGGER = 20000  # tokens
KEEP_LAST_TURNS = 3

async def maybe_summarize(self):
    """Summarize older conversation turns if approaching token limit."""
    if self.conversation.summarizing:
        return
    if self.conversation.latest_tokens < SUMMARY_TRIGGER:
        return

    self.conversation.summarizing = True
    try:
        # Keep last N turns, summarize the rest
        to_summarize = self.conversation.history[:-KEEP_LAST_TURNS]
        if not to_summarize:
            return

        text = "\n".join(f"{t['role']}: {t['text']}" for t in to_summarize if t['text'])

        # Use gpt-4o-mini for cheap summarization
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Summarize this conversation concisely, preserving key facts, decisions, and context the assistant needs to continue naturally."
            }, {
                "role": "user",
                "content": text
            }],
            max_tokens=500
        )
        summary = response.choices[0].message.content

        # Inject summary as system message at root
        self.conversation.summary_count += 1
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "previous_item_id": "root",
            "item": {
                "id": f"summary_{self.conversation.summary_count}",
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": f"[Conversation summary]: {summary}"}]
            }
        }))

        # Delete summarized turns from server
        for turn in to_summarize:
            await self.ws.send(json.dumps({
                "type": "conversation.item.delete",
                "item_id": turn['item_id']
            }))

        # Update local state
        self.conversation.history = self.conversation.history[-KEEP_LAST_TURNS:]
    finally:
        self.conversation.summarizing = False
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `gpt-4o-realtime-preview-2024-12-17` model | `gpt-realtime` (GA model) | Aug 2025 | Production-grade, 60-min sessions, no session limits |
| `server_vad` only | `semantic_vad` available | Late 2024 | Better turn detection, fewer interruptions |
| 6 voices (alloy, echo, fable, onyx, nova, shimmer) | 10+ voices including ash, ballad, coral, sage, verse, cedar, marin | 2024-2025 | Ash voice available as requested |
| 30-minute session limit | 60-minute session limit | 2025 | Longer uninterrupted conversations |

**Deprecated/outdated:**
- `gpt-4o-realtime-preview-2024-12-17`: Preview model, should be replaced with `gpt-realtime`
- Manual `response.cancel` for interruption: Use `interrupt_response: true` in turn_detection config instead

## Codebase-Specific Findings

### Files Requiring Rename Changes
All locations where "live" refers to the dictation mode (not the new live voice session):

**push-to-talk.py:**
- Line 228: `"dictation_mode": "live"` in default config
- Lines 1006-1012: Voice command `'go live', 'live mode', 'going live'` -> set dictation_mode to 'live'
- Lines 1217-1223: Stream mode voice command same pattern
- Lines 1008, 1219: `self.config['dictation_mode'] = 'live'`

**indicator.py:**
- Line 67: `"dictation_mode": "live"` in default config
- Line 403: `self.mode_combo.append("live", "Live (instant typing)")`
- Line 1025-1027: `mode_names = {'live': 'Manual', ...}` display names
- Lines 1387-1389: QuickControlWindow modes list `('live', 'Manual', 'types after release')`
- Line 1467: `mode_names = {'live': 'Manual', ...}`

**Website files (per SOP - must stay in sync):**
- `/home/ethan/code/braceyourself/bys-website/resources/views/livewire/push-to-talk-page.blade.php` line 648: `<span>= Dictate</span>` (already says Dictate, but verify)
- `/home/ethan/code/braceyourself/bys-website/resources/views/landing/push-to-talk.blade.php`: No "live mode" references found

### New AI Mode Registration
The `ai_mode` config currently supports: `"claude"`, `"realtime"`, `"interview"`, `"conversation"`. The new live mode should be added as `ai_mode: "live"` to this list. This requires:
- Adding to `load_config()` defaults comment
- Adding to Settings combo box in `indicator.py`
- Adding routing in `PushToTalk.on_press()` (around line 2100)
- The existing `"realtime"` mode should remain as-is (it has tools, different behavior)

### Session Auto-Start Pattern
Decision: "Session auto-starts when user selects Live mode in settings combo box." This means the `on_ai_mode_changed()` handler in `indicator.py` (line 801) needs to trigger the session start in `push-to-talk.py`. Current inter-process communication is via `config.json` file -- `push-to-talk.py` reloads config on each PTT press (line 898). For auto-start, `push-to-talk.py` needs to watch config changes or the indicator needs to signal via the status file.

**Recommended approach:** Add a config watcher in `push-to-talk.py` that polls `config.json` modification time every 500ms. When `ai_mode` changes to `"live"`, auto-start the session. When it changes away from `"live"`, stop the session. This is consistent with the existing architecture (file-based IPC between indicator and main service).

### Clawdbot Personality System Investigation
**Location investigated:** `laptop:~/clawd/` via SSH

The clawdbot system (named "Russel") uses 8+ markdown files to compose its personality:
- `IDENTITY.md`: Name, creature type, vibe (10 lines)
- `SOUL.md`: Core behavioral rules, execution triggers, contradiction handling (200+ lines)
- `USER.md`: Human's name, timezone, working style, preferences, notification prefs
- `MEMORY.md`: SOPs and learned behaviors
- `AGENTS.md`: Sub-agent delegation rules
- `HEARTBEAT.md`: Periodic check-in behavior
- `TOOLS.md`: Available tool descriptions
- `TROUBLESHOOTING.md`: Known issues and fixes

**Key patterns to adopt:**
1. **SOUL.md is the core**: "Be genuinely helpful, not performatively helpful. Skip filler, pleasantries, and narration." This matches the Phase 1 personality decisions.
2. **USER.md personalizes**: Knowing the user's name, working style, and communication preferences makes interactions feel natural.
3. **Contradiction handling**: "Newest wins, explicit > implicit, narrow > global" -- useful for session context management.
4. **Continuity pattern**: "Each session, you wake up fresh. These files are your memory." -- exactly the pattern needed for session context loading.

**Adaptation for push-to-talk live mode:**
- Simpler than clawdbot (fewer files, no sub-agents)
- `personality/core.md`: Identity + soul (combined IDENTITY + SOUL)
- `personality/voice-style.md`: Voice-specific rules (concise, spoken language, fillers OK)
- `personality/context.md`: Auto-managed file with recent session summaries

### Learnings System Investigation
**Location:** `/home/ethan/.claude/learnings/`

The learnings system runs hourly via cron (`cron-evaluate.sh` -> `evaluate.py`):
- Extracts learnings from Claude CLI conversation history
- Categorizes into: Preferences, Gotchas, Procedures, Patterns
- Generates `SUMMARY.md` with top entries per category
- Injected into Claude sessions via `inject-learnings.sh` hook

**Relevance to Phase 1:** The learnings system can provide context for the live session's personality. The `context.md` personality file could include a curated subset of learnings relevant to general conversation. However, this is more of a Phase 3 concern (when tools are available). For Phase 1, the personality files are hand-written and static.

## Open Questions

1. **Overlay widget position persistence**
   - What we know: The existing indicator dot saves position to config.json. The overlay should do the same.
   - What's unclear: Should the overlay replace the status dot during live mode, or appear alongside it?
   - Recommendation: Show overlay INSTEAD of the status dot during live mode. Revert to dot when live mode ends. This avoids two competing status indicators.

2. **Voice configuration for Realtime API**
   - What we know: The decision says "configurable in settings, default to Ash." The existing voice options in settings are for OpenAI TTS (alloy, echo, fable, onyx, nova, shimmer).
   - What's unclear: The Realtime API has additional voices (ash, ballad, coral, sage, verse, cedar, marin). Should these be a separate setting or unified?
   - Recommendation: Add all Realtime API voices to the existing voice combo. The Realtime API accepts all TTS voices plus its additional ones. Use a single setting.

3. **Idle timeout duration**
   - What we know: Decision says "Claude's Discretion" for timeout duration.
   - What's unclear: Optimal value for desktop use.
   - Recommendation: 2 minutes of no PTT activity. The Realtime API's `idle_timeout_ms` feature on `server_vad` can be leveraged, but since we use `semantic_vad`, implement a client-side timer. After 2 minutes idle, disconnect WebSocket but keep `ConversationState` in memory. Reconnect on next PTT press.

4. **"Connected" / "Session ended" announcements**
   - What we know: Decision says voice announcement on connect/disconnect.
   - What's unclear: Should this use the Realtime API voice or the local TTS system?
   - Recommendation: Use the Realtime API itself. Send a `conversation.item.create` with text "Connected" immediately after session setup, followed by a `response.create` to make the AI speak it naturally. For "Session ended," use local TTS (since the WebSocket is being torn down).

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `push-to-talk.py`, `openai_realtime.py`, `indicator.py` -- direct file reading
- Clawdbot personality system: `laptop:~/clawd/` -- SSH investigation of SOUL.md, IDENTITY.md, USER.md, MEMORY.md
- Learnings system: `/home/ethan/.claude/learnings/` -- direct file reading

### Secondary (MEDIUM confidence)
- [OpenAI Realtime API Cookbook - Context Summarization](https://developers.openai.com/cookbook/examples/context_summarization_with_realtime_api) - ConversationState pattern, conversation.item.create payload format
- [OpenAI Realtime API WebSocket docs](https://platform.openai.com/docs/guides/realtime-websocket) - Session configuration, audio format
- [OpenAI Realtime API VAD docs](https://platform.openai.com/docs/guides/realtime-vad) - semantic_vad configuration, eagerness parameter
- [OpenAI gpt-realtime model page](https://platform.openai.com/docs/models/gpt-realtime) - GA model endpoint
- [OpenAI Realtime voices announcement](https://community.openai.com/t/new-realtime-api-voices-and-cache-pricing/998238) - Ash and other new voices confirmed

### Tertiary (LOW confidence)
- [GTK3 transparent window gist](https://gist.github.com/KurtJacobson/374c8cb83aee4851d39981b9c7e2c22c) - Pattern for transparent overlay windows
- [audioviz-desk GitHub](https://github.com/gephaistos/audioviz-desk) - GTK audio visualization reference

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use, no new dependencies
- Architecture: HIGH - Based on direct codebase analysis + established OpenAI patterns
- Rename scope: HIGH - Exhaustive grep of all files, every reference identified
- Personality system: MEDIUM - Based on clawdbot investigation, not yet implemented in this codebase
- Overlay widget: MEDIUM - GTK3 patterns well-understood, waveform specifics need implementation
- Pitfalls: MEDIUM - Based on existing codebase bugs + API documentation

**Research date:** 2026-02-13
**Valid until:** 2026-03-13 (30 days - APIs and voices stable at GA)
