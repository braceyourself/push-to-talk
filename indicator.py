#!/usr/bin/env python3
"""
Push-to-Talk Status Indicator

A small floating colored dot that shows recording status.
Hover/click to see status and manage the service.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Pango
import os
import subprocess
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "status"
VOCAB_FILE = Path(__file__).parent / "vocabulary.txt"
LOG_CMD = ['journalctl', '--user', '-u', 'push-to-talk', '-n', '5', '--no-pager', '-o', 'cat']
DOT_SIZE = 20
CORNER_MARGIN = 10

# Colors for each status
COLORS = {
    'idle': (0.5, 0.5, 0.5, 0.3),
    'recording': (1.0, 0.2, 0.2, 0.9),
    'processing': (1.0, 0.8, 0.0, 0.9),
    'success': (0.2, 0.9, 0.2, 0.9),
    'error': (1.0, 0.0, 0.0, 0.9),
}

STATUS_TEXT = {
    'idle': 'Idle - Ready',
    'recording': 'Recording...',
    'processing': 'Transcribing...',
    'success': 'Success',
    'error': 'Error',
}


class StatusPopup(Gtk.Window):
    """Popup window showing status details and controls."""

    def __init__(self, parent_indicator):
        super().__init__(type=Gtk.WindowType.POPUP)
        self.parent_indicator = parent_indicator

        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_border_width(10)

        # Main container
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add(vbox)

        # Title
        title = Gtk.Label()
        title.set_markup("<b>Push-to-Talk</b>")
        vbox.pack_start(title, False, False, 0)

        # Separator
        vbox.pack_start(Gtk.Separator(), False, False, 0)

        # Status
        self.status_label = Gtk.Label()
        self.status_label.set_xalign(0)
        vbox.pack_start(self.status_label, False, False, 0)

        # Vocabulary count
        self.vocab_label = Gtk.Label()
        self.vocab_label.set_xalign(0)
        vbox.pack_start(self.vocab_label, False, False, 0)

        # Recent log
        log_label = Gtk.Label()
        log_label.set_markup("<b>Recent:</b>")
        log_label.set_xalign(0)
        vbox.pack_start(log_label, False, False, 0)

        self.log_text = Gtk.Label()
        self.log_text.set_xalign(0)
        self.log_text.set_line_wrap(True)
        self.log_text.set_max_width_chars(40)
        self.log_text.set_ellipsize(Pango.EllipsizeMode.END)
        self.log_text.set_lines(3)
        vbox.pack_start(self.log_text, False, False, 0)

        # Separator
        vbox.pack_start(Gtk.Separator(), False, False, 5)

        # Buttons
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        vbox.pack_start(btn_box, False, False, 0)

        restart_btn = Gtk.Button(label="Restart")
        restart_btn.connect("clicked", self.on_restart)
        btn_box.pack_start(restart_btn, True, True, 0)

        vocab_btn = Gtk.Button(label="Vocabulary")
        vocab_btn.connect("clicked", self.on_edit_vocab)
        btn_box.pack_start(vocab_btn, True, True, 0)

        logs_btn = Gtk.Button(label="Logs")
        logs_btn.connect("clicked", self.on_view_logs)
        btn_box.pack_start(logs_btn, True, True, 0)

        # Style
        self.get_style_context().add_class('popup')
        css = Gtk.CssProvider()
        css.load_from_data(b'''
            .popup {
                background-color: #2d2d2d;
                color: #ffffff;
                border-radius: 8px;
                border: 1px solid #555;
            }
            button {
                background: #444;
                color: white;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px 8px;
            }
            button:hover {
                background: #555;
            }
        ''')
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def update_content(self):
        """Update the popup content."""
        status = self.parent_indicator.status
        self.status_label.set_text(f"Status: {STATUS_TEXT.get(status, status)}")

        # Vocabulary count
        try:
            vocab_count = len([l for l in VOCAB_FILE.read_text().splitlines()
                             if l.strip() and not l.startswith('#')])
            self.vocab_label.set_text(f"Vocabulary: {vocab_count} words")
        except:
            self.vocab_label.set_text("Vocabulary: N/A")

        # Recent logs
        try:
            result = subprocess.run(LOG_CMD, capture_output=True, text=True, timeout=2)
            lines = result.stdout.strip().split('\n')[-3:]
            # Extract just the transcribed text if present
            recent = []
            for line in lines:
                if 'Transcribed:' in line:
                    recent.append(line.split('Transcribed:')[1].strip()[:50])
            self.log_text.set_text('\n'.join(recent) if recent else 'No recent transcriptions')
        except:
            self.log_text.set_text('Unable to fetch logs')

    def on_restart(self, button):
        subprocess.Popen(['systemctl', '--user', 'restart', 'push-to-talk.service'])
        self.hide()

    def on_edit_vocab(self, button):
        subprocess.Popen(['xdg-open', str(VOCAB_FILE)])
        self.hide()

    def on_view_logs(self, button):
        subprocess.Popen(['gnome-terminal', '--', 'journalctl', '--user', '-u',
                         'push-to-talk', '-f', '--no-pager'])
        self.hide()

    def show_at(self, x, y):
        self.update_content()
        self.move(x, y + DOT_SIZE + 5)
        self.show_all()


class StatusIndicator(Gtk.Window):
    def __init__(self):
        super().__init__()

        self.status = 'idle'
        self.success_timeout = None
        self.popup = None
        self.hover_timeout = None

        # Window setup - use regular window but undecorated
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_accept_focus(False)
        self.set_app_paintable(True)
        self.set_default_size(DOT_SIZE, DOT_SIZE)
        self.set_resizable(False)
        self.set_type_hint(Gdk.WindowTypeHint.DOCK)

        # Enable events
        self.set_events(Gdk.EventMask.ENTER_NOTIFY_MASK |
                       Gdk.EventMask.LEAVE_NOTIFY_MASK |
                       Gdk.EventMask.BUTTON_PRESS_MASK)

        self.connect('enter-notify-event', self.on_enter)
        self.connect('leave-notify-event', self.on_leave)
        self.connect('button-press-event', self.on_click)

        # Enable transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)

        # Drawing
        self.connect('draw', self.on_draw)

        # Position at top center of primary monitor (ultrawide at 1920,1440)
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        if monitor:
            geometry = monitor.get_geometry()
            self.pos_x = geometry.x + (geometry.width - DOT_SIZE) // 2
            self.pos_y = geometry.y + 35  # Below panel
        else:
            # Fallback: hardcoded for ultrawide at 1920,1440 with 3840 width
            self.pos_x = 1920 + (3840 - DOT_SIZE) // 2
            self.pos_y = 1440 + 35  # Below panel

        print(f"Indicator position: {self.pos_x}, {self.pos_y}", flush=True)
        self.move(self.pos_x, self.pos_y)

        # Watch status file
        GLib.timeout_add(100, self.check_status)

        # Initialize status file
        STATUS_FILE.write_text('idle')

        self.show_all()

    def on_draw(self, widget, cr):
        cr.set_operator(1)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()

        color = COLORS.get(self.status, COLORS['idle'])
        cr.set_source_rgba(*color)
        cr.arc(DOT_SIZE / 2, DOT_SIZE / 2, DOT_SIZE / 2 - 2, 0, 2 * 3.14159)
        cr.fill()

        cr.set_source_rgba(0.2, 0.2, 0.2, 0.5)
        cr.set_line_width(1)
        cr.arc(DOT_SIZE / 2, DOT_SIZE / 2, DOT_SIZE / 2 - 2, 0, 2 * 3.14159)
        cr.stroke()

        return False

    def on_enter(self, widget, event):
        # Show popup after brief hover
        if self.hover_timeout:
            GLib.source_remove(self.hover_timeout)
        self.hover_timeout = GLib.timeout_add(300, self.show_popup)
        return True

    def on_leave(self, widget, event):
        if self.hover_timeout:
            GLib.source_remove(self.hover_timeout)
            self.hover_timeout = None
        # Hide popup after delay (allows moving to popup)
        GLib.timeout_add(500, self.maybe_hide_popup)
        return True

    def on_click(self, widget, event):
        self.show_popup()
        return True

    def show_popup(self):
        if not self.popup:
            self.popup = StatusPopup(self)
        self.popup.show_at(self.pos_x - 100, self.pos_y)
        self.hover_timeout = None
        return False

    def maybe_hide_popup(self):
        if self.popup and self.popup.get_visible():
            # Check if mouse is over popup
            display = Gdk.Display.get_default()
            seat = display.get_default_seat()
            pointer = seat.get_pointer()
            _, x, y = pointer.get_position()

            # Get popup geometry
            px, py = self.popup.get_position()
            pw, ph = self.popup.get_size()

            # Also check indicator
            if not (px <= x <= px + pw and py <= y <= py + ph):
                if not (self.pos_x <= x <= self.pos_x + DOT_SIZE and
                       self.pos_y <= y <= self.pos_y + DOT_SIZE):
                    self.popup.hide()
        return False

    def check_status(self):
        try:
            if STATUS_FILE.exists():
                new_status = STATUS_FILE.read_text().strip()
                if new_status != self.status:
                    self.set_status(new_status)
        except:
            pass
        return True

    def set_status(self, status):
        if self.success_timeout:
            GLib.source_remove(self.success_timeout)
            self.success_timeout = None

        self.status = status
        self.queue_draw()

        if status == 'success':
            self.success_timeout = GLib.timeout_add(1500, self.return_to_idle)

    def return_to_idle(self):
        self.status = 'idle'
        self.queue_draw()
        self.success_timeout = None
        STATUS_FILE.write_text('idle')
        return False


def main():
    indicator = StatusIndicator()
    Gtk.main()


if __name__ == '__main__':
    main()
