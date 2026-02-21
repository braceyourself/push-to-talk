"""
Native GTK3 dashboard for push-to-talk live sessions.

Reads events by tailing events.jsonl, sends commands by appending to it.
Zero HTTP — all communication through the event bus JSONL file.

Launched from indicator.py right-click menu or auto-opened on live start.
"""

import json
import math
import subprocess
import time
from pathlib import Path

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Pango

from event_bus import BusEvent, EventBusWriter


# Status dot colors (matches indicator.py DOT_COLORS)
DOT_COLORS = {
    'listening':    (0.29, 0.85, 0.50),   # green
    'speaking':     (0.38, 0.65, 0.98),   # blue
    'hearing':      (0.56, 0.75, 0.99),   # lighter blue
    'processing':   (1.0,  0.8,  0.0),    # yellow
    'idle':         (0.42, 0.45, 0.50),   # gray
    'disconnected': (0.42, 0.45, 0.50),   # gray
    'error':        (1.0,  0.2,  0.2),    # red
    'muted':        (0.8,  0.4,  0.0),    # orange
    'thinking':     (1.0,  0.8,  0.0),    # yellow
    'tool_use':     (0.95, 0.55, 0.0),    # orange-amber
}

STATUS_LABELS = {
    'listening':    'Listening',
    'speaking':     'Speaking',
    'hearing':      'Hearing you...',
    'processing':   'Processing',
    'idle':         'Idle',
    'disconnected': 'Disconnected',
    'error':        'Error',
    'muted':        'Muted',
    'thinking':     'Thinking',
    'tool_use':     'Using Tool',
}

# Pipeline stage names and the events that activate them
PIPELINE_STAGES = ['Mic', 'STT', 'LLM', 'Composer', 'Playback']

# Which event types activate which pipeline stage (index into PIPELINE_STAGES)
STAGE_ACTIVATE = {
    'stt_start': 1,
    'stt_complete': 1,
    'llm_send': 2,
    'llm_first_token': 2,
    'llm_complete': 2,
    'tts_start': 3,
    'tts_complete': 3,
    'status': None,  # handled specially
}

# High-frequency events to suppress in "All" filter mode
_SUPPRESS_ALL = {'audio_rms', 'queue_depths', 'snapshot', 'llm_text_delta'}

# Event log filter categories
_FILTER_PIPELINE = {'stt_start', 'stt_complete', 'llm_send', 'llm_first_token',
                    'llm_complete', 'tts_start', 'tts_complete', 'filler_played'}
_FILTER_TURNS = {'user', 'assistant'}
_FILTER_ERRORS = {'error', 'task_failed'}

_CSS = """
window {
    background: #1a1a2e;
    color: #e0e0e0;
}
.status-bar {
    background: #16213e;
    padding: 8px 12px;
}
.status-label {
    font-weight: bold;
    font-size: 14px;
}
.meta-label {
    color: #888;
    font-size: 11px;
}
.controls {
    padding: 8px 12px;
}
.ctrl-btn {
    background: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 0;
}
.ctrl-btn:hover {
    background: #3a3a5a;
}
.ctrl-btn.danger {
    color: #ff6b6b;
}
.ctrl-btn.muted {
    background: #ff4444;
    color: white;
}
.metrics {
    padding: 4px 12px;
}
.metric-label {
    color: #888;
    font-size: 11px;
}
.metric-value {
    font-family: monospace;
    font-weight: bold;
    font-size: 13px;
}
.pipeline-box {
    padding: 8px 12px;
}
.pipeline-label {
    font-size: 10px;
    color: #888;
}
.pipeline-label-active {
    font-size: 10px;
    color: #e0e0e0;
    font-weight: bold;
}
.queue-label {
    font-size: 9px;
    color: #666;
    font-family: monospace;
}
.section-label {
    font-weight: bold;
    font-size: 12px;
    padding: 6px 12px 2px 12px;
    color: #aaa;
}
.conversation-view {
    background: #0d1117;
    padding: 4px;
}
.turn-user {
    color: #58a6ff;
    font-size: 12px;
    padding: 2px 8px;
}
.turn-assistant {
    color: #a78bfa;
    font-size: 12px;
    padding: 2px 8px;
}
.event-log {
    font-family: monospace;
    font-size: 11px;
    background: #0d1117;
    color: #8b949e;
}
.separator {
    background: #333;
    min-height: 1px;
}
"""


class DashboardWindow(Gtk.Window):
    """Native GTK3 dashboard for live session monitoring and control."""

    MAX_CONVERSATION_TURNS = 30
    MAX_EVENT_LOG_LINES = 200

    def __init__(self):
        super().__init__(title="Push-to-Talk Dashboard")
        self.set_default_size(500, 700)
        self.set_type_hint(Gdk.WindowTypeHint.NORMAL)

        # Apply CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(_CSS.encode())
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        # State
        self._bus_writer = None
        self._bus_path = None
        self._bus_file = None
        self._status = 'idle'
        self._gen_id = 0
        self._model = ''
        self._pipeline_active = [False] * len(PIPELINE_STAGES)
        self._pipeline_timeouts = [None] * len(PIPELINE_STAGES)
        self._queue_depths = [0] * len(PIPELINE_STAGES)
        self._conversation_count = 0
        self._event_log_lines = 0
        self._event_filter = 'All'

        self._active_session_path = Path.home() / ".local/share/push-to-talk/active_session"
        self._status_file = Path(__file__).parent / "status"

        # Build UI
        self._build_ui()

        # Start polling
        GLib.timeout_add(100, self._poll_events)
        GLib.timeout_add(1000, self._check_session)

        # Try to connect immediately
        self._check_session()

    def _build_ui(self):
        """Build the dashboard layout."""
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(vbox)

        # ── Status bar ──
        self._status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._status_bar.get_style_context().add_class('status-bar')

        self._status_dot = Gtk.DrawingArea()
        self._status_dot.set_size_request(14, 14)
        self._status_dot.connect('draw', self._draw_status_dot)
        self._status_bar.pack_start(self._status_dot, False, False, 0)

        self._status_label = Gtk.Label(label="Idle")
        self._status_label.get_style_context().add_class('status-label')
        self._status_bar.pack_start(self._status_label, False, False, 0)

        spacer = Gtk.Label()
        self._status_bar.pack_start(spacer, True, True, 0)

        self._gen_label = Gtk.Label(label="")
        self._gen_label.get_style_context().add_class('meta-label')
        self._status_bar.pack_start(self._gen_label, False, False, 0)

        self._model_label = Gtk.Label(label="")
        self._model_label.get_style_context().add_class('meta-label')
        self._status_bar.pack_start(self._model_label, False, False, 0)

        vbox.pack_start(self._status_bar, False, False, 0)
        vbox.pack_start(self._make_separator(), False, False, 0)

        # ── Controls ──
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.get_style_context().add_class('controls')

        self._mute_btn = self._make_button("Mute", self._on_mute)
        controls.pack_start(self._mute_btn, False, False, 0)

        self._interrupt_btn = self._make_button("Interrupt", self._on_interrupt)
        controls.pack_start(self._interrupt_btn, False, False, 0)

        spacer2 = Gtk.Label()
        controls.pack_start(spacer2, True, True, 0)

        self._restart_btn = self._make_button("Restart", self._on_restart)
        controls.pack_start(self._restart_btn, False, False, 0)

        self._stop_btn = self._make_button("Stop", self._on_stop)
        self._stop_btn.get_style_context().add_class('danger')
        controls.pack_start(self._stop_btn, False, False, 0)

        vbox.pack_start(controls, False, False, 0)
        vbox.pack_start(self._make_separator(), False, False, 0)

        # ── Metrics ──
        metrics = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        metrics.get_style_context().add_class('metrics')

        self._stt_metric = self._make_metric("STT", "—")
        metrics.pack_start(self._stt_metric, False, False, 0)

        self._ttft_metric = self._make_metric("TTFT", "—")
        metrics.pack_start(self._ttft_metric, False, False, 0)

        self._tts_metric = self._make_metric("TTS", "—")
        metrics.pack_start(self._tts_metric, False, False, 0)

        vbox.pack_start(metrics, False, False, 0)
        vbox.pack_start(self._make_separator(), False, False, 0)

        # ── Pipeline ──
        pipeline_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        pipeline_box.get_style_context().add_class('pipeline-box')

        stages_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self._pipeline_dots = []
        self._pipeline_labels = []
        self._queue_labels = []

        for i, name in enumerate(PIPELINE_STAGES):
            stage_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            stage_box.set_halign(Gtk.Align.CENTER)

            # Dot + name row
            dot_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            dot = Gtk.DrawingArea()
            dot.set_size_request(10, 10)
            dot.connect('draw', self._draw_pipeline_dot, i)
            self._pipeline_dots.append(dot)
            dot_row.pack_start(dot, False, False, 0)

            label = Gtk.Label(label=name)
            label.get_style_context().add_class('pipeline-label')
            self._pipeline_labels.append(label)
            dot_row.pack_start(label, False, False, 0)

            stage_box.pack_start(dot_row, False, False, 0)

            # Queue depth
            q_label = Gtk.Label(label="")
            q_label.get_style_context().add_class('queue-label')
            self._queue_labels.append(q_label)
            stage_box.pack_start(q_label, False, False, 0)

            stages_row.pack_start(stage_box, True, True, 0)

            # Arrow between stages
            if i < len(PIPELINE_STAGES) - 1:
                arrow = Gtk.Label(label="\u2192")
                arrow.get_style_context().add_class('pipeline-label')
                stages_row.pack_start(arrow, False, False, 4)

        pipeline_box.pack_start(stages_row, False, False, 0)
        vbox.pack_start(pipeline_box, False, False, 0)
        vbox.pack_start(self._make_separator(), False, False, 0)

        # ── Conversation ──
        conv_label = Gtk.Label(label="Conversation")
        conv_label.set_halign(Gtk.Align.START)
        conv_label.get_style_context().add_class('section-label')
        vbox.pack_start(conv_label, False, False, 0)

        conv_scroll = Gtk.ScrolledWindow()
        conv_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        conv_scroll.set_min_content_height(120)

        self._conversation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._conversation_box.get_style_context().add_class('conversation-view')
        conv_scroll.add(self._conversation_box)
        self._conv_scroll = conv_scroll
        vbox.pack_start(conv_scroll, True, True, 0)
        vbox.pack_start(self._make_separator(), False, False, 0)

        # ── Event log ──
        event_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        event_label = Gtk.Label(label="Events")
        event_label.set_halign(Gtk.Align.START)
        event_label.get_style_context().add_class('section-label')
        event_header.pack_start(event_label, True, True, 0)

        self._filter_combo = Gtk.ComboBoxText()
        for f in ['All', 'Pipeline', 'Turns', 'Errors']:
            self._filter_combo.append_text(f)
        self._filter_combo.set_active(0)
        self._filter_combo.connect('changed', self._on_filter_changed)
        event_header.pack_start(self._filter_combo, False, False, 8)

        vbox.pack_start(event_header, False, False, 0)

        event_scroll = Gtk.ScrolledWindow()
        event_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        event_scroll.set_min_content_height(140)

        self._event_buffer = Gtk.TextBuffer()
        self._event_view = Gtk.TextView(buffer=self._event_buffer)
        self._event_view.set_editable(False)
        self._event_view.set_cursor_visible(False)
        self._event_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._event_view.get_style_context().add_class('event-log')
        event_scroll.add(self._event_view)
        self._event_scroll = event_scroll
        vbox.pack_start(event_scroll, True, True, 0)

    def _make_separator(self):
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep.get_style_context().add_class('separator')
        return sep

    def _make_button(self, label, callback):
        btn = Gtk.Button(label=label)
        btn.get_style_context().add_class('ctrl-btn')
        btn.connect('clicked', lambda _: callback())
        return btn

    def _make_metric(self, name, value):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        name_label = Gtk.Label(label=name)
        name_label.get_style_context().add_class('metric-label')
        box.pack_start(name_label, False, False, 0)

        val_label = Gtk.Label(label=value)
        val_label.get_style_context().add_class('metric-value')
        box.pack_start(val_label, False, False, 0)

        # Store reference to value label for updates
        box._value_label = val_label
        return box

    # ── Drawing ──

    def _draw_status_dot(self, widget, cr):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        r, g, b = DOT_COLORS.get(self._status, DOT_COLORS['idle'])

        # Glow effect
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 1
        if self._status not in ('idle', 'disconnected'):
            cr.set_source_rgba(r, g, b, 0.3)
            cr.arc(cx, cy, radius + 2, 0, 2 * math.pi)
            cr.fill()

        cr.set_source_rgb(r, g, b)
        cr.arc(cx, cy, radius, 0, 2 * math.pi)
        cr.fill()

    def _draw_pipeline_dot(self, widget, cr, stage_idx):
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        active = self._pipeline_active[stage_idx]

        if active:
            r, g, b = 0.29, 0.85, 0.50  # green
            # Glow
            cr.set_source_rgba(r, g, b, 0.3)
            cr.arc(w / 2, h / 2, 6, 0, 2 * math.pi)
            cr.fill()
        else:
            r, g, b = 0.3, 0.3, 0.35  # dim gray

        cr.set_source_rgb(r, g, b)
        cr.arc(w / 2, h / 2, 4, 0, 2 * math.pi)
        cr.fill()

    # ── Session discovery ──

    def _check_session(self):
        """Discover/switch to current session's bus."""
        try:
            if self._active_session_path.exists():
                session_dir = Path(self._active_session_path.read_text().strip())
                bus_path = session_dir / "events.jsonl"
                if bus_path != self._bus_path and bus_path.exists():
                    self._switch_bus(bus_path, session_dir)
        except Exception:
            pass
        return True

    def _switch_bus(self, bus_path, session_dir):
        """Switch to a new session's bus."""
        if self._bus_file:
            try:
                self._bus_file.close()
            except Exception:
                pass

        self._bus_path = bus_path
        self._bus_file = open(bus_path, "r")
        self._bus_file.seek(0, 2)  # Seek to end (only tail new events)

        # Set up writer for commands
        if self._bus_writer:
            self._bus_writer.close()
        sid = session_dir.name
        self._bus_writer = EventBusWriter(bus_path, "dashboard", sid)
        self._bus_writer.open()

        self._update_status('listening', 0, '')

    # ── Event polling ──

    def _poll_events(self):
        """Read new lines from events.jsonl."""
        if not self._bus_file:
            return True
        try:
            while True:
                line = self._bus_file.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    try:
                        evt = BusEvent.from_json_line(line)
                        self._handle_event(evt)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except Exception:
            pass
        return True

    # ── Event handling ──

    def _handle_event(self, evt):
        """Process a single bus event."""
        etype = evt.type

        # Status
        if etype == 'status':
            status = evt.payload.get('status', 'idle')
            model = evt.payload.get('model', self._model)
            self._update_status(status, evt.gen, model)

            # Pipeline: mic is active when listening/hearing
            if status in ('listening', 'hearing'):
                self._activate_stage(0, timeout_ms=3000)
            elif status == 'speaking':
                self._activate_stage(4, timeout_ms=5000)

        # Metrics
        elif etype == 'stt_complete':
            latency = evt.payload.get('latency_ms')
            if latency is not None:
                self._stt_metric._value_label.set_text(f"{int(latency)}ms")
            self._activate_stage(1, timeout_ms=2000)

        elif etype == 'llm_first_token':
            latency = evt.payload.get('latency_ms')
            if latency is not None:
                self._ttft_metric._value_label.set_text(f"{int(latency)}ms")
            self._activate_stage(2, timeout_ms=5000)

        elif etype == 'tts_complete':
            latency = evt.payload.get('latency_ms')
            if latency is not None:
                self._tts_metric._value_label.set_text(f"{int(latency)}ms")

        # Pipeline activation
        elif etype in STAGE_ACTIVATE:
            idx = STAGE_ACTIVATE[etype]
            if idx is not None:
                timeout = 5000 if idx == 2 else 3000  # LLM gets longer timeout
                self._activate_stage(idx, timeout_ms=timeout)

        # Pipeline deactivation
        if etype == 'stt_complete':
            self._deactivate_stage(1, delay_ms=500)
        elif etype == 'llm_complete':
            self._deactivate_stage(2, delay_ms=500)
        elif etype == 'tts_complete':
            self._deactivate_stage(3, delay_ms=500)

        # Conversation turns
        if etype == 'user':
            text = evt.payload.get('text', '')
            if text:
                self._add_conversation_turn('You', text)
        elif etype == 'assistant':
            text = evt.payload.get('text', '')
            if text:
                self._add_conversation_turn('AI', text)

        # Queue depths (ephemeral but might appear in file)
        if etype == 'queue_depths':
            depths = evt.payload
            # Map known queue names to stage indices
            for key, idx in [('stt', 1), ('llm', 2), ('composer', 3), ('playback', 4)]:
                val = depths.get(key, 0)
                if idx < len(self._queue_labels):
                    self._queue_labels[idx].set_text(f"Q:{val}" if val else "")
                    self._queue_depths[idx] = val

        # Event log
        self._append_event_log(evt)

    def _update_status(self, status, gen, model):
        """Update status bar UI."""
        self._status = status
        self._gen_id = gen
        if model:
            self._model = model

        label_text = STATUS_LABELS.get(status, status.title())
        self._status_label.set_text(label_text)
        self._status_dot.queue_draw()

        if gen:
            self._gen_label.set_text(f"Gen: {gen}")
        if self._model:
            # Short model name
            short = self._model
            for prefix in ('claude-', 'gpt-'):
                if short.startswith(prefix):
                    short = short[len(prefix):]
            # Remove date suffix
            parts = short.rsplit('-', 1)
            if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
                short = parts[0]
            self._model_label.set_text(short)

        # Update mute button label
        if status == 'muted':
            self._mute_btn.set_label("Unmute")
            self._mute_btn.get_style_context().add_class('muted')
        else:
            self._mute_btn.set_label("Mute")
            self._mute_btn.get_style_context().remove_class('muted')

    # ── Pipeline stage activation ──

    def _activate_stage(self, idx, timeout_ms=3000):
        """Activate a pipeline stage dot with auto-deactivation timeout."""
        if idx >= len(PIPELINE_STAGES):
            return
        self._pipeline_active[idx] = True
        self._pipeline_dots[idx].queue_draw()
        self._pipeline_labels[idx].get_style_context().remove_class('pipeline-label')
        self._pipeline_labels[idx].get_style_context().add_class('pipeline-label-active')

        # Cancel existing timeout
        if self._pipeline_timeouts[idx]:
            GLib.source_remove(self._pipeline_timeouts[idx])

        # Set new auto-deactivation timeout
        self._pipeline_timeouts[idx] = GLib.timeout_add(
            timeout_ms, self._deactivate_stage_cb, idx)

    def _deactivate_stage(self, idx, delay_ms=0):
        """Deactivate a pipeline stage after optional delay."""
        if delay_ms:
            # Cancel existing timeout and set a short one
            if self._pipeline_timeouts[idx]:
                GLib.source_remove(self._pipeline_timeouts[idx])
            self._pipeline_timeouts[idx] = GLib.timeout_add(
                delay_ms, self._deactivate_stage_cb, idx)
        else:
            self._deactivate_stage_cb(idx)

    def _deactivate_stage_cb(self, idx):
        """Callback to deactivate a pipeline stage."""
        if idx >= len(PIPELINE_STAGES):
            return False
        self._pipeline_active[idx] = False
        self._pipeline_dots[idx].queue_draw()
        self._pipeline_labels[idx].get_style_context().remove_class('pipeline-label-active')
        self._pipeline_labels[idx].get_style_context().add_class('pipeline-label')
        self._pipeline_timeouts[idx] = None
        return False  # Don't repeat

    # ── Conversation panel ──

    def _add_conversation_turn(self, role, text):
        """Add a conversation turn to the panel."""
        # Truncate long text
        display_text = text[:200] + "..." if len(text) > 200 else text

        label = Gtk.Label(label=f"{role}: {display_text}")
        label.set_halign(Gtk.Align.START)
        label.set_line_wrap(True)
        label.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
        label.set_max_width_chars(60)

        css_class = 'turn-user' if role == 'You' else 'turn-assistant'
        label.get_style_context().add_class(css_class)
        label.show()

        self._conversation_box.pack_start(label, False, False, 0)
        self._conversation_count += 1

        # Cap entries
        if self._conversation_count > self.MAX_CONVERSATION_TURNS:
            children = self._conversation_box.get_children()
            if children:
                self._conversation_box.remove(children[0])
                self._conversation_count -= 1

        # Auto-scroll to bottom
        GLib.idle_add(self._scroll_to_bottom, self._conv_scroll)

    def _scroll_to_bottom(self, scrolled_window):
        adj = scrolled_window.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())
        return False

    # ── Event log ──

    def _append_event_log(self, evt):
        """Append an event to the log, respecting the current filter."""
        etype = evt.type

        # Apply filter
        if self._event_filter == 'All' and etype in _SUPPRESS_ALL:
            return
        elif self._event_filter == 'Pipeline' and etype not in _FILTER_PIPELINE:
            return
        elif self._event_filter == 'Turns' and etype not in _FILTER_TURNS:
            return
        elif self._event_filter == 'Errors' and etype not in _FILTER_ERRORS:
            return

        # Format line
        ts_str = time.strftime("%H:%M:%S", time.localtime(evt.ts))
        parts = [f"{ts_str} {etype}"]

        # Add relevant payload info
        payload = evt.payload
        if 'text' in payload:
            text = payload['text']
            if len(text) > 60:
                text = text[:60] + "..."
            parts.append(f'"{text}"')
        if 'latency_ms' in payload:
            parts.append(f"({int(payload['latency_ms'])}ms)")
        if 'status' in payload:
            parts.append(payload['status'])
        if 'action' in payload:
            parts.append(payload['action'])
        if 'error' in payload:
            err = payload['error']
            if len(err) > 80:
                err = err[:80] + "..."
            parts.append(err)

        line = " ".join(parts) + "\n"

        # Append to buffer
        end_iter = self._event_buffer.get_end_iter()
        self._event_buffer.insert(end_iter, line)
        self._event_log_lines += 1

        # Cap lines
        if self._event_log_lines > self.MAX_EVENT_LOG_LINES:
            start = self._event_buffer.get_start_iter()
            first_newline = start.copy()
            first_newline.forward_to_line_end()
            first_newline.forward_char()
            self._event_buffer.delete(start, first_newline)
            self._event_log_lines -= 1

        # Auto-scroll
        GLib.idle_add(self._scroll_to_bottom, self._event_scroll)

    def _on_filter_changed(self, combo):
        self._event_filter = combo.get_active_text() or 'All'

    # ── Commands ──

    def _send_command(self, action):
        """Write command to bus JSONL."""
        if self._bus_writer:
            self._bus_writer.emit("command", action=action)

    def _on_mute(self):
        if self._status == 'muted':
            self._send_command('unmute')
        else:
            self._send_command('mute')

    def _on_interrupt(self):
        self._send_command('interrupt')

    def _on_restart(self):
        self._send_command('stop')
        # Write restart signal to status file
        try:
            self._status_file.write_text('restart_live')
        except Exception:
            pass

    def _on_stop(self):
        self._send_command('stop')
        try:
            subprocess.Popen(['systemctl', '--user', 'stop', 'push-to-talk.service'])
        except Exception:
            pass

    # ── Cleanup ──

    def do_destroy(self):
        """Clean up on window close."""
        if self._bus_file:
            try:
                self._bus_file.close()
            except Exception:
                pass
        if self._bus_writer:
            try:
                self._bus_writer.close()
            except Exception:
                pass
        Gtk.Window.do_destroy(self)
