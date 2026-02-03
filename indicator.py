#!/usr/bin/env python3
"""
Push-to-Talk Status Indicator

A small floating colored dot that shows recording status.
Hover/click to see status and manage the service.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Pango
import cairo
import os
import json
import subprocess
import tempfile
from pathlib import Path

# Try to load AppIndicator3 for tray mode
try:
    gi.require_version('AppIndicator3', '0.1')
    from gi.repository import AppIndicator3
    APPINDICATOR_AVAILABLE = True
except (ValueError, ImportError):
    APPINDICATOR_AVAILABLE = False

STATUS_FILE = Path(__file__).parent / "status"
VOCAB_FILE = Path(__file__).parent / "vocabulary.txt"
CONFIG_FILE = Path(__file__).parent / "config.json"
OPENAI_KEY_FILE = Path.home() / ".config" / "openai" / "api_key"

# OpenAI voice options
OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Key mapping options (display names only — pynput is in push-to-talk.py)
MODIFIER_KEY_OPTIONS = {
    "ctrl_r": "Right Ctrl",
    "ctrl_l": "Left Ctrl",
    "shift_r": "Right Shift",
    "shift_l": "Left Shift",
    "alt_r": "Right Alt",
    "alt_l": "Left Alt",
}
INTERRUPT_KEY_OPTIONS = {
    "escape": "Escape",
    "space": "Spacebar",
    "pause": "Pause",
    "scroll_lock": "Scroll Lock",
}


def load_config():
    """Load configuration."""
    default = {
        "tts_backend": "piper",
        "openai_voice": "nova",
        "ai_mode": "claude",
        "debug_mode": False,
        "ptt_key": "ctrl_r",
        "ai_key": "shift_r",
        "interrupt_key": "escape",
        "indicator_style": "floating",
        "indicator_x": None,
        "indicator_y": None,
    }
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = json.load(f)
                return {**default, **config}
    except:
        pass
    return default


def save_config(config):
    """Save configuration."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except:
        pass


def get_openai_api_key():
    """Get OpenAI API key."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if OPENAI_KEY_FILE.exists():
        return OPENAI_KEY_FILE.read_text().strip()
    return None


def save_openai_api_key(key):
    """Save OpenAI API key."""
    OPENAI_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPENAI_KEY_FILE.write_text(key)
    OPENAI_KEY_FILE.chmod(0o600)


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
    'listening': (0.2, 0.6, 1.0, 0.9),  # Blue - AI listening
    'speaking': (0.8, 0.4, 1.0, 0.9),   # Purple - AI speaking (mic muted)
}

STATUS_TEXT = {
    'idle': 'Idle - Ready',
    'recording': 'Recording...',
    'processing': 'Transcribing...',
    'success': 'Success',
    'error': 'Error',
    'listening': 'AI Listening...',
    'speaking': 'AI Speaking...',
}


class SettingsWindow(Gtk.Window):
    """Settings window with tabbed interface."""

    def __init__(self, parent_indicator):
        super().__init__(title="Push-to-Talk Settings")
        self.parent_indicator = parent_indicator

        self.set_default_size(450, 400)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_resizable(False)

        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(main_box)

        # Create notebook (tabs)
        notebook = Gtk.Notebook()
        main_box.pack_start(notebook, True, True, 0)

        # Add tabs
        notebook.append_page(self.create_general_tab(), Gtk.Label(label="General"))
        notebook.append_page(self.create_api_keys_tab(), Gtk.Label(label="API Keys"))
        notebook.append_page(self.create_hotkeys_tab(), Gtk.Label(label="Hotkeys"))
        notebook.append_page(self.create_advanced_tab(), Gtk.Label(label="Advanced"))

        # Bottom buttons
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        btn_box.set_margin_top(10)
        btn_box.set_margin_bottom(10)
        btn_box.set_margin_start(10)
        btn_box.set_margin_end(10)
        main_box.pack_start(btn_box, False, False, 0)

        close_btn = Gtk.Button(label="Close")
        close_btn.connect("clicked", lambda w: self.destroy())
        btn_box.pack_end(close_btn, False, False, 0)

        # Apply CSS
        self.apply_styles()

    def apply_styles(self):
        """Apply dark theme styles."""
        css = Gtk.CssProvider()
        css.load_from_data(b'''
            window {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            notebook {
                background-color: #2d2d2d;
            }
            notebook tab {
                background-color: #3d3d3d;
                padding: 8px 16px;
                color: #ffffff;
            }
            notebook tab:checked {
                background-color: #4d4d4d;
            }
            notebook > stack {
                background-color: #2d2d2d;
            }
            label {
                color: #ffffff;
            }
            entry {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
            button {
                background: #444;
                color: white;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 6px 12px;
            }
            button:hover {
                background: #555;
            }
            combobox button {
                background: #3d3d3d;
            }
            combobox window {
                background-color: #3d3d3d;
            }
            checkbutton {
                color: #ffffff;
            }
            .status-valid {
                color: #4ade80;
            }
            .status-invalid {
                color: #f87171;
            }
            .status-missing {
                color: #fbbf24;
            }
            .section-title {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 8px;
            }
            .info-text {
                color: #9ca3af;
                font-size: 12px;
            }
        ''')
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def create_general_tab(self):
        """Create the General settings tab."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)

        config = load_config()

        # Indicator Style Section
        ind_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        ind_label = Gtk.Label(label="Status Indicator")
        ind_label.set_xalign(0)
        ind_label.get_style_context().add_class('section-title')
        ind_section.pack_start(ind_label, False, False, 0)

        ind_mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        ind_mode_label = Gtk.Label(label="Indicator Style:")
        ind_mode_label.set_xalign(0)
        ind_mode_box.pack_start(ind_mode_label, False, False, 0)

        self.ind_style_combo = Gtk.ComboBoxText()
        self.ind_style_combo.append("floating", "Floating Dot")
        self.ind_style_combo.append("tray", "System Tray")
        self.ind_style_combo.set_active_id(config.get('indicator_style', 'floating'))
        self.ind_style_combo.connect("changed", self.on_indicator_style_changed)
        ind_mode_box.pack_end(self.ind_style_combo, False, False, 0)

        ind_section.pack_start(ind_mode_box, False, False, 0)

        self.ind_restart_label = Gtk.Label()
        self.ind_restart_label.set_xalign(0)
        ind_section.pack_start(self.ind_restart_label, False, False, 0)

        if not APPINDICATOR_AVAILABLE:
            na_label = Gtk.Label(label="System tray requires gir1.2-appindicator3-0.1")
            na_label.set_xalign(0)
            na_label.get_style_context().add_class('info-text')
            ind_section.pack_start(na_label, False, False, 0)

        ind_info = Gtk.Label(label="Floating dot is draggable. System tray integrates\nwith your desktop panel.")
        ind_info.set_xalign(0)
        ind_info.get_style_context().add_class('info-text')
        ind_section.pack_start(ind_info, False, False, 0)

        box.pack_start(ind_section, False, False, 0)

        # AI Mode Section
        ai_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        ai_label = Gtk.Label(label="AI Mode")
        ai_label.set_xalign(0)
        ai_label.get_style_context().add_class('section-title')
        ai_section.pack_start(ai_label, False, False, 0)

        ai_mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        ai_mode_label = Gtk.Label(label="Assistant Backend:")
        ai_mode_label.set_xalign(0)
        ai_mode_box.pack_start(ai_mode_label, False, False, 0)

        self.ai_mode_combo = Gtk.ComboBoxText()
        self.ai_mode_combo.append("claude", "Claude + Whisper")
        self.ai_mode_combo.append("realtime", "OpenAI Realtime (GPT-4o)")
        self.ai_mode_combo.set_active_id(config.get('ai_mode', 'claude'))
        self.ai_mode_combo.connect("changed", self.on_ai_mode_changed)
        ai_mode_box.pack_end(self.ai_mode_combo, False, False, 0)

        ai_section.pack_start(ai_mode_box, False, False, 0)

        ai_info = Gtk.Label(label="Claude mode uses local Whisper + Claude CLI.\nRealtime mode uses OpenAI's voice-to-voice API.")
        ai_info.set_xalign(0)
        ai_info.get_style_context().add_class('info-text')
        ai_section.pack_start(ai_info, False, False, 0)

        box.pack_start(ai_section, False, False, 0)

        # TTS Section
        tts_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        tts_label = Gtk.Label(label="Text-to-Speech")
        tts_label.set_xalign(0)
        tts_label.get_style_context().add_class('section-title')
        tts_section.pack_start(tts_label, False, False, 0)

        # TTS Backend
        tts_backend_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        tts_backend_label = Gtk.Label(label="TTS Backend:")
        tts_backend_label.set_xalign(0)
        tts_backend_box.pack_start(tts_backend_label, False, False, 0)

        self.tts_backend_combo = Gtk.ComboBoxText()
        self.tts_backend_combo.append("piper", "Piper (Local)")
        self.tts_backend_combo.append("openai", "OpenAI (Cloud)")
        self.tts_backend_combo.set_active_id(config.get('tts_backend', 'piper'))
        self.tts_backend_combo.connect("changed", self.on_tts_backend_changed)
        tts_backend_box.pack_end(self.tts_backend_combo, False, False, 0)

        tts_section.pack_start(tts_backend_box, False, False, 0)

        # OpenAI Voice
        voice_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        voice_label = Gtk.Label(label="OpenAI Voice:")
        voice_label.set_xalign(0)
        voice_box.pack_start(voice_label, False, False, 0)

        self.voice_combo = Gtk.ComboBoxText()
        for voice in OPENAI_VOICES:
            self.voice_combo.append(voice, voice.capitalize())
        self.voice_combo.set_active_id(config.get('openai_voice', 'nova'))
        self.voice_combo.connect("changed", self.on_voice_changed)
        voice_box.pack_end(self.voice_combo, False, False, 0)

        tts_section.pack_start(voice_box, False, False, 0)

        tts_info = Gtk.Label(label="Piper is free and runs locally.\nOpenAI voices require an API key and have usage costs.")
        tts_info.set_xalign(0)
        tts_info.get_style_context().add_class('info-text')
        tts_section.pack_start(tts_info, False, False, 0)

        box.pack_start(tts_section, False, False, 0)

        return box

    def create_api_keys_tab(self):
        """Create the API Keys settings tab."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)

        # OpenAI API Key Section
        openai_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        openai_label = Gtk.Label(label="OpenAI API Key")
        openai_label.set_xalign(0)
        openai_label.get_style_context().add_class('section-title')
        openai_section.pack_start(openai_label, False, False, 0)

        # Status
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status_label = Gtk.Label(label="Status:")
        status_box.pack_start(status_label, False, False, 0)

        self.api_status_label = Gtk.Label()
        self.update_api_status()
        status_box.pack_start(self.api_status_label, False, False, 0)

        openai_section.pack_start(status_box, False, False, 0)

        # API Key Entry
        key_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_placeholder_text("sk-...")
        self.api_key_entry.set_visibility(False)
        self.api_key_entry.set_width_chars(40)
        current_key = get_openai_api_key()
        if current_key:
            self.api_key_entry.set_text(current_key)
        key_box.pack_start(self.api_key_entry, True, True, 0)

        # Show/Hide toggle
        self.show_key_btn = Gtk.Button(label="Show")
        self.show_key_btn.connect("clicked", self.toggle_key_visibility)
        key_box.pack_start(self.show_key_btn, False, False, 0)

        openai_section.pack_start(key_box, False, False, 0)

        # Save button
        save_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        save_btn = Gtk.Button(label="Save API Key")
        save_btn.connect("clicked", self.on_save_api_key)
        save_box.pack_start(save_btn, False, False, 0)

        self.save_status_label = Gtk.Label()
        save_box.pack_start(self.save_status_label, False, False, 0)

        openai_section.pack_start(save_box, False, False, 0)

        # Info
        info = Gtk.Label(label="Get your API key from https://platform.openai.com/api-keys\nThe key is stored in ~/.config/openai/api_key")
        info.set_xalign(0)
        info.get_style_context().add_class('info-text')
        openai_section.pack_start(info, False, False, 0)

        box.pack_start(openai_section, False, False, 0)

        return box

    def create_hotkeys_tab(self):
        """Create the Hotkeys settings tab with configurable dropdowns."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)

        config = load_config()

        title = Gtk.Label(label="Keyboard Shortcuts")
        title.set_xalign(0)
        title.get_style_context().add_class('section-title')
        box.pack_start(title, False, False, 0)

        # Push-to-Talk key
        ptt_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        ptt_label = Gtk.Label(label="Push-to-Talk (Dictate):")
        ptt_label.set_xalign(0)
        ptt_row.pack_start(ptt_label, True, True, 0)

        self.ptt_combo = Gtk.ComboBoxText()
        for key_id, name in MODIFIER_KEY_OPTIONS.items():
            self.ptt_combo.append(key_id, name)
        self.ptt_combo.set_active_id(config.get('ptt_key', 'ctrl_r'))
        self.ptt_combo.connect("changed", self.on_hotkey_changed)
        ptt_row.pack_end(self.ptt_combo, False, False, 0)
        box.pack_start(ptt_row, False, False, 0)

        ptt_desc = Gtk.Label(label="Hold to record, release to transcribe and type")
        ptt_desc.set_xalign(0)
        ptt_desc.get_style_context().add_class('info-text')
        box.pack_start(ptt_desc, False, False, 0)

        # AI Assistant key
        ai_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        ai_label = Gtk.Label(label="AI Assistant modifier:")
        ai_label.set_xalign(0)
        ai_row.pack_start(ai_label, True, True, 0)

        self.ai_combo = Gtk.ComboBoxText()
        for key_id, name in MODIFIER_KEY_OPTIONS.items():
            self.ai_combo.append(key_id, name)
        self.ai_combo.set_active_id(config.get('ai_key', 'shift_r'))
        self.ai_combo.connect("changed", self.on_hotkey_changed)
        ai_row.pack_end(self.ai_combo, False, False, 0)
        box.pack_start(ai_row, False, False, 0)

        ai_desc = Gtk.Label(label="Hold PTT + this key for AI voice assistant")
        ai_desc.set_xalign(0)
        ai_desc.get_style_context().add_class('info-text')
        box.pack_start(ai_desc, False, False, 0)

        # Interrupt key
        int_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        int_label = Gtk.Label(label="Interrupt AI:")
        int_label.set_xalign(0)
        int_row.pack_start(int_label, True, True, 0)

        self.interrupt_combo = Gtk.ComboBoxText()
        for key_id, name in INTERRUPT_KEY_OPTIONS.items():
            self.interrupt_combo.append(key_id, name)
        self.interrupt_combo.set_active_id(config.get('interrupt_key', 'escape'))
        self.interrupt_combo.connect("changed", self.on_hotkey_changed)
        int_row.pack_end(self.interrupt_combo, False, False, 0)
        box.pack_start(int_row, False, False, 0)

        int_desc = Gtk.Label(label="Press to stop AI speech during response")
        int_desc.set_xalign(0)
        int_desc.get_style_context().add_class('info-text')
        box.pack_start(int_desc, False, False, 0)

        # Validation error label
        self.hotkey_error_label = Gtk.Label()
        self.hotkey_error_label.set_xalign(0)
        self.hotkey_error_label.set_margin_top(10)
        box.pack_start(self.hotkey_error_label, False, False, 0)

        # Restart notice area
        self.hotkey_restart_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.hotkey_restart_box.set_margin_top(10)

        self.hotkey_restart_label = Gtk.Label()
        self.hotkey_restart_box.pack_start(self.hotkey_restart_label, False, False, 0)

        restart_btn = Gtk.Button(label="Restart Now")
        restart_btn.connect("clicked", lambda w: subprocess.Popen([
            'systemctl', '--user', 'restart', 'push-to-talk.service'
        ]))
        self.hotkey_restart_box.pack_start(restart_btn, False, False, 0)

        self.hotkey_restart_box.set_no_show_all(True)
        self.hotkey_restart_box.hide()
        box.pack_start(self.hotkey_restart_box, False, False, 0)

        return box

    def on_hotkey_changed(self, combo):
        """Handle hotkey dropdown change with validation."""
        ptt_id = self.ptt_combo.get_active_id()
        ai_id = self.ai_combo.get_active_id()
        interrupt_id = self.interrupt_combo.get_active_id()

        # Validate PTT and AI keys are different
        if ptt_id == ai_id:
            self.hotkey_error_label.set_markup(
                "<span foreground='#f87171'>PTT and AI keys must be different!</span>")
            self.hotkey_restart_box.hide()
            return

        self.hotkey_error_label.set_text("")

        # Save config
        config = load_config()
        config['ptt_key'] = ptt_id
        config['ai_key'] = ai_id
        config['interrupt_key'] = interrupt_id
        save_config(config)

        # Show restart notice
        self.hotkey_restart_label.set_markup(
            "<span foreground='#fbbf24'>Restart required for changes to take effect</span>")
        self.hotkey_restart_box.show_all()

    def create_advanced_tab(self):
        """Create the Advanced settings tab."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)

        config = load_config()

        # Vocabulary Section
        vocab_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        vocab_label = Gtk.Label(label="Vocabulary")
        vocab_label.set_xalign(0)
        vocab_label.get_style_context().add_class('section-title')
        vocab_section.pack_start(vocab_label, False, False, 0)

        vocab_count = 0
        try:
            vocab_count = len([l for l in VOCAB_FILE.read_text().splitlines()
                             if l.strip() and not l.startswith('#')])
        except:
            pass

        vocab_info = Gtk.Label(label=f"Custom words: {vocab_count}")
        vocab_info.set_xalign(0)
        vocab_section.pack_start(vocab_info, False, False, 0)

        vocab_btn = Gtk.Button(label="Edit Vocabulary File")
        vocab_btn.connect("clicked", lambda w: subprocess.Popen(['xdg-open', str(VOCAB_FILE)]))
        vocab_section.pack_start(vocab_btn, False, False, 0)

        vocab_tip = Gtk.Label(label="Add custom words to improve transcription accuracy.\nSay \"correction: [word]\" to add words via voice.")
        vocab_tip.set_xalign(0)
        vocab_tip.get_style_context().add_class('info-text')
        vocab_section.pack_start(vocab_tip, False, False, 0)

        box.pack_start(vocab_section, False, False, 0)

        # Debug Section
        debug_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        debug_label = Gtk.Label(label="Debugging")
        debug_label.set_xalign(0)
        debug_label.get_style_context().add_class('section-title')
        debug_section.pack_start(debug_label, False, False, 0)

        self.debug_check = Gtk.CheckButton(label="Enable debug mode")
        self.debug_check.set_active(config.get('debug_mode', False))
        self.debug_check.connect("toggled", self.on_debug_toggled)
        debug_section.pack_start(self.debug_check, False, False, 0)

        logs_btn = Gtk.Button(label="View Logs")
        logs_btn.connect("clicked", lambda w: subprocess.Popen([
            'gnome-terminal', '--', 'journalctl', '--user', '-u',
            'push-to-talk', '-f', '--no-pager'
        ]))
        debug_section.pack_start(logs_btn, False, False, 0)

        box.pack_start(debug_section, False, False, 0)

        # Service Section
        service_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        service_label = Gtk.Label(label="Service")
        service_label.set_xalign(0)
        service_label.get_style_context().add_class('section-title')
        service_section.pack_start(service_label, False, False, 0)

        restart_btn = Gtk.Button(label="Restart Service")
        restart_btn.connect("clicked", lambda w: subprocess.Popen([
            'systemctl', '--user', 'restart', 'push-to-talk.service'
        ]))
        service_section.pack_start(restart_btn, False, False, 0)

        box.pack_start(service_section, False, False, 0)

        return box

    def update_api_status(self):
        """Update API key status indicator."""
        key = get_openai_api_key()
        if key and key.startswith('sk-'):
            self.api_status_label.set_markup("<span foreground='#4ade80'>Valid key found</span>")
        elif key:
            self.api_status_label.set_markup("<span foreground='#fbbf24'>Key found (may be invalid)</span>")
        else:
            self.api_status_label.set_markup("<span foreground='#f87171'>No key configured</span>")

    def toggle_key_visibility(self, button):
        """Toggle API key visibility."""
        visible = self.api_key_entry.get_visibility()
        self.api_key_entry.set_visibility(not visible)
        self.show_key_btn.set_label("Hide" if not visible else "Show")

    def on_save_api_key(self, button):
        """Save the API key."""
        key = self.api_key_entry.get_text().strip()
        if key:
            save_openai_api_key(key)
            self.update_api_status()
            self.save_status_label.set_markup("<span foreground='#4ade80'>Saved!</span>")
            GLib.timeout_add(2000, lambda: self.save_status_label.set_text(""))
        else:
            self.save_status_label.set_markup("<span foreground='#f87171'>No key entered</span>")

    def on_ai_mode_changed(self, combo):
        """Handle AI mode change."""
        config = load_config()
        config['ai_mode'] = combo.get_active_id()
        save_config(config)

    def on_tts_backend_changed(self, combo):
        """Handle TTS backend change."""
        config = load_config()
        config['tts_backend'] = combo.get_active_id()
        save_config(config)

    def on_voice_changed(self, combo):
        """Handle voice selection change."""
        config = load_config()
        config['openai_voice'] = combo.get_active_id()
        save_config(config)

    def on_debug_toggled(self, button):
        """Handle debug mode toggle."""
        config = load_config()
        config['debug_mode'] = button.get_active()
        save_config(config)

    def on_indicator_style_changed(self, combo):
        """Handle indicator style change."""
        config = load_config()
        config['indicator_style'] = combo.get_active_id()
        save_config(config)
        self.ind_restart_label.set_markup(
            "<span foreground='#fbbf24'>Restart service for this to take effect</span>")


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

        # AI Mode
        self.ai_mode_label = Gtk.Label()
        self.ai_mode_label.set_xalign(0)
        vbox.pack_start(self.ai_mode_label, False, False, 0)

        # TTS Backend
        self.backend_label = Gtk.Label()
        self.backend_label.set_xalign(0)
        vbox.pack_start(self.backend_label, False, False, 0)

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

        settings_btn = Gtk.Button(label="Settings")
        settings_btn.connect("clicked", self.on_settings)
        btn_box.pack_start(settings_btn, True, True, 0)

        restart_btn = Gtk.Button(label="Restart")
        restart_btn.connect("clicked", self.on_restart)
        btn_box.pack_start(restart_btn, True, True, 0)

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

        # AI Mode & TTS Backend
        config = load_config()
        ai_mode = config.get('ai_mode', 'claude')
        tts = config.get('tts_backend', 'piper')
        voice = config.get('openai_voice', 'nova')

        ai_display = "Realtime (GPT-4o)" if ai_mode == 'realtime' else "Claude + Whisper"
        self.ai_mode_label.set_text(f"AI: {ai_display}")

        tts_display = f"OpenAI ({voice})" if tts == 'openai' else "Piper (local)"
        self.backend_label.set_text(f"Voice: {tts_display}")

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

    def on_settings(self, button):
        """Open settings window."""
        settings = SettingsWindow(self.parent_indicator)
        settings.show_all()
        self.hide()

    def on_restart(self, button):
        subprocess.Popen(['systemctl', '--user', 'restart', 'push-to-talk.service'])
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

        # Drag state
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_threshold = 5

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
                       Gdk.EventMask.BUTTON_PRESS_MASK |
                       Gdk.EventMask.BUTTON_RELEASE_MASK |
                       Gdk.EventMask.POINTER_MOTION_MASK)

        self.connect('enter-notify-event', self.on_enter)
        self.connect('leave-notify-event', self.on_leave)
        self.connect('button-press-event', self.on_button_press)
        self.connect('button-release-event', self.on_button_release)
        self.connect('motion-notify-event', self.on_motion)

        # Enable transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)

        # Drawing
        self.connect('draw', self.on_draw)

        # Position: load from config or default to top center of primary monitor
        config = load_config()
        saved_x = config.get('indicator_x')
        saved_y = config.get('indicator_y')

        if saved_x is not None and saved_y is not None:
            self.pos_x = int(saved_x)
            self.pos_y = int(saved_y)
        else:
            display = Gdk.Display.get_default()
            monitor = display.get_primary_monitor()
            if monitor:
                geometry = monitor.get_geometry()
                self.pos_x = geometry.x + (geometry.width - DOT_SIZE) // 2
                self.pos_y = geometry.y + 35  # Below panel
            else:
                self.pos_x = 1920 + (3840 - DOT_SIZE) // 2
                self.pos_y = 1440 + 35

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
        # Show popup after brief hover (suppress during drag)
        if self.dragging:
            return True
        if self.hover_timeout:
            GLib.source_remove(self.hover_timeout)
        self.hover_timeout = GLib.timeout_add(300, self.show_popup)
        return True

    def on_leave(self, widget, event):
        if self.hover_timeout:
            GLib.source_remove(self.hover_timeout)
            self.hover_timeout = None
        if not self.dragging:
            # Hide popup after delay (allows moving to popup)
            GLib.timeout_add(500, self.maybe_hide_popup)
        return True

    def on_button_press(self, widget, event):
        if event.button == 1:
            self.drag_start_x = event.x_root
            self.drag_start_y = event.y_root
            self.dragging = False  # Not yet — wait for motion threshold
        return True

    def on_button_release(self, widget, event):
        if event.button == 1:
            if not self.dragging:
                # Click without drag — show popup
                self.show_popup()
            else:
                # Drag ended — save position
                self.dragging = False
                config = load_config()
                config['indicator_x'] = self.pos_x
                config['indicator_y'] = self.pos_y
                save_config(config)
        return True

    def on_motion(self, widget, event):
        if event.state & Gdk.ModifierType.BUTTON1_MASK:
            dx = event.x_root - self.drag_start_x
            dy = event.y_root - self.drag_start_y

            if not self.dragging:
                # Check threshold before starting drag
                if abs(dx) > self.drag_threshold or abs(dy) > self.drag_threshold:
                    self.dragging = True
                    # Cancel hover popup
                    if self.hover_timeout:
                        GLib.source_remove(self.hover_timeout)
                        self.hover_timeout = None
                    # Hide popup if visible
                    if self.popup and self.popup.get_visible():
                        self.popup.hide()

            if self.dragging:
                self.pos_x = int(self.pos_x + dx)
                self.pos_y = int(self.pos_y + dy)
                self.move(self.pos_x, self.pos_y)
                self.drag_start_x = event.x_root
                self.drag_start_y = event.y_root
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


class TrayIndicator:
    """System tray indicator using AppIndicator3."""

    ICON_SIZE = 22

    def __init__(self):
        self.status = 'idle'
        self.success_timeout = None
        self.icon_dir = tempfile.mkdtemp(prefix='ptt-icons-')
        self.icon_paths = {}

        # Pre-generate icons for each status
        for status_name, rgba in COLORS.items():
            self._generate_icon(status_name, rgba)

        # Create indicator
        self.indicator = AppIndicator3.Indicator.new(
            'push-to-talk',
            self.icon_paths['idle'],
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_title('Push-to-Talk')

        # Build menu
        self.menu = Gtk.Menu()

        self.status_item = Gtk.MenuItem(label=STATUS_TEXT.get('idle', 'Idle'))
        self.status_item.set_sensitive(False)
        self.menu.append(self.status_item)

        self.menu.append(Gtk.SeparatorMenuItem())

        settings_item = Gtk.MenuItem(label='Settings')
        settings_item.connect('activate', self.on_settings)
        self.menu.append(settings_item)

        restart_item = Gtk.MenuItem(label='Restart Service')
        restart_item.connect('activate', lambda w: subprocess.Popen([
            'systemctl', '--user', 'restart', 'push-to-talk.service'
        ]))
        self.menu.append(restart_item)

        logs_item = Gtk.MenuItem(label='View Logs')
        logs_item.connect('activate', lambda w: subprocess.Popen([
            'gnome-terminal', '--', 'journalctl', '--user', '-u',
            'push-to-talk', '-f', '--no-pager'
        ]))
        self.menu.append(logs_item)

        self.menu.show_all()
        self.indicator.set_menu(self.menu)

        # Watch status file
        GLib.timeout_add(100, self.check_status)

        # Initialize status file
        STATUS_FILE.write_text('idle')

    def _generate_icon(self, name, rgba):
        """Generate a colored circle icon as PNG."""
        size = self.ICON_SIZE
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
        cr = cairo.Context(surface)

        # Transparent background
        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        # Colored circle
        r, g, b, a = rgba
        cr.set_source_rgba(r, g, b, a)
        cr.arc(size / 2, size / 2, size / 2 - 1, 0, 2 * 3.14159)
        cr.fill()

        # Border
        cr.set_source_rgba(0.2, 0.2, 0.2, 0.5)
        cr.set_line_width(1)
        cr.arc(size / 2, size / 2, size / 2 - 1, 0, 2 * 3.14159)
        cr.stroke()

        path = os.path.join(self.icon_dir, f'{name}.png')
        surface.write_to_png(path)
        # AppIndicator needs path without extension
        self.icon_paths[name] = path[:-4]  # strip .png

    def on_settings(self, widget):
        """Open settings window."""
        settings = SettingsWindow(self)
        settings.show_all()

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
        icon_path = self.icon_paths.get(status, self.icon_paths['idle'])
        self.indicator.set_icon_full(icon_path, STATUS_TEXT.get(status, status))
        self.status_item.set_label(STATUS_TEXT.get(status, status))

        if status == 'success':
            self.success_timeout = GLib.timeout_add(1500, self.return_to_idle)

    def return_to_idle(self):
        self.status = 'idle'
        icon_path = self.icon_paths.get('idle')
        self.indicator.set_icon_full(icon_path, 'Idle - Ready')
        self.status_item.set_label('Idle - Ready')
        self.success_timeout = None
        STATUS_FILE.write_text('idle')
        return False


def main():
    config = load_config()
    style = config.get('indicator_style', 'floating')

    if style == 'tray' and APPINDICATOR_AVAILABLE:
        print("Starting tray indicator", flush=True)
        indicator = TrayIndicator()
    else:
        if style == 'tray' and not APPINDICATOR_AVAILABLE:
            print("AppIndicator3 not available, falling back to floating dot", flush=True)
        print("Starting floating indicator", flush=True)
        indicator = StatusIndicator()

    Gtk.main()


if __name__ == '__main__':
    main()
