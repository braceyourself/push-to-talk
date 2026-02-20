#!/usr/bin/env python3
"""
Push-to-Talk Dictation Service

Hold Right Ctrl to record, release to transcribe and type into focused input.
Learns from corrections and maintains a custom vocabulary.
"""

import os
import sys
import re
import time
import math
import json
import wave
import shutil
import tempfile
import subprocess
import threading
import signal
import atexit
from collections import deque
from pathlib import Path

# --- Auto-detect DISPLAY for systemd service startup ---
def _detect_display():
    """Detect the active X display if DISPLAY is not set or unreachable."""
    if os.environ.get('DISPLAY'):
        # Test if the current DISPLAY actually works
        try:
            subprocess.run(['xdpyinfo'], capture_output=True, timeout=2,
                           env={**os.environ})
            return  # Current DISPLAY works fine
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except subprocess.CalledProcessError:
            pass  # DISPLAY set but not working, try to detect

    # Try common display values
    for display in [':1', ':0', ':2']:
        try:
            env = {**os.environ, 'DISPLAY': display}
            result = subprocess.run(['xdpyinfo'], capture_output=True, timeout=2, env=env)
            if result.returncode == 0:
                os.environ['DISPLAY'] = display
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # Fallback: check who owns the X server
    try:
        result = subprocess.run(['w', '-hs'], capture_output=True, text=True, timeout=2)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].startswith(':'):
                os.environ['DISPLAY'] = parts[1]
                return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

def _wait_for_display(max_wait=30):
    """Wait for X display to become available (for login race condition)."""
    start = time.time()
    while time.time() - start < max_wait:
        _detect_display()
        if os.environ.get('DISPLAY'):
            try:
                env = {**os.environ}
                result = subprocess.run(['xdpyinfo'], capture_output=True, timeout=2, env=env)
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        time.sleep(1)
    return False

if not _wait_for_display():
    print("ERROR: Could not connect to X display after 30s", file=sys.stderr)
    sys.exit(1)

# Also ensure XAUTHORITY is set
if not os.environ.get('XAUTHORITY'):
    xauth = f"/run/user/{os.getuid()}/gdm/Xauthority"
    if os.path.exists(xauth):
        os.environ['XAUTHORITY'] = xauth

import whisper
from pynput import keyboard

# Import OpenAI for TTS (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import OpenAI Realtime (optional)
try:
    from openai_realtime import RealtimeSession, get_api_key as get_realtime_api_key, is_available as realtime_available
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    realtime_available = lambda: False

# Import Live Session (optional)
try:
    from live_session import LiveSession
    LIVE_SESSION_AVAILABLE = True
except ImportError:
    LIVE_SESSION_AVAILABLE = False

def get_openai_api_key():
    """Get OpenAI API key from environment or file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for path in [
        Path.home() / ".config" / "openai" / "api_key",
        Path.home() / ".openai" / "api_key",
    ]:
        if path.exists():
            return path.read_text().strip()
    return None


def get_deepgram_api_key():
    """Get Deepgram API key from environment or file."""
    key = os.environ.get("DEEPGRAM_API_KEY")
    if key:
        return key
    for path in [
        Path.home() / ".config" / "deepgram" / "api_key",
        Path.home() / ".deepgram" / "api_key",
    ]:
        if path.exists():
            return path.read_text().strip()
    return None


def prompt_api_key():
    """Open a terminal to prompt the user for their OpenAI API key.
    Returns True if a key was saved, False otherwise."""
    script = r'''#!/bin/bash
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

echo ""
echo -e "${YELLOW}=== OpenAI API Key Required ===${NC}"
echo ""
echo "The AI assistant needs an OpenAI API key to work."
echo ""
echo "Push-to-Talk uses OpenAI for:"
echo "  - Realtime voice conversations (GPT-4o)"
echo "  - Higher quality TTS voices"
echo ""
echo -e "${BLUE}Your key stays on your machine.${NC}"
echo "  - Stored locally at ~/.config/openai/api_key (read-only by you)"
echo "  - Used only for direct API calls from this device to OpenAI"
echo "  - Never sent to any third party"
echo ""
read -p "Paste your OpenAI API key (or press Enter to cancel): " api_key
if [ -n "$api_key" ]; then
    mkdir -p "$HOME/.config/openai"
    echo "$api_key" > "$HOME/.config/openai/api_key"
    chmod 600 "$HOME/.config/openai/api_key"
    echo ""
    echo -e "${GREEN}[OK]${NC} API key saved. You can now use the AI assistant."
    echo "    Press Right Ctrl + Right Shift to start."
    sleep 2
else
    echo ""
    echo "No key entered. AI assistant will not be available until a key is configured."
    sleep 2
fi
'''
    try:
        # Write a temp script so the terminal can run it
        script_path = Path("/tmp/ptt-api-key-prompt.sh")
        script_path.write_text(script)
        script_path.chmod(0o755)

        # Try to open a terminal
        terminals = [
            ['gnome-terminal', '--', 'bash', str(script_path)],
            ['xfce4-terminal', '-e', f'bash {script_path}'],
            ['konsole', '-e', 'bash', str(script_path)],
            ['xterm', '-e', 'bash', str(script_path)],
        ]
        for cmd in terminals:
            if shutil.which(cmd[0]):
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        return False
    except Exception as e:
        print(f"Failed to open API key prompt: {e}", flush=True)
        return False

# Key mapping options
MODIFIER_KEY_OPTIONS = {
    "ctrl_r": ("Right Ctrl", keyboard.Key.ctrl_r),
    "ctrl_l": ("Left Ctrl", keyboard.Key.ctrl_l),
    "shift_r": ("Right Shift", keyboard.Key.shift_r),
    "shift_l": ("Left Shift", keyboard.Key.shift_l),
    "alt_r": ("Right Alt", keyboard.Key.alt_r),
    "alt_l": ("Left Alt", keyboard.Key.alt_l),
}
INTERRUPT_KEY_OPTIONS = {
    "escape": ("Escape", keyboard.Key.esc),
    "space": ("Spacebar", keyboard.Key.space),
    "pause": ("Pause", keyboard.Key.pause),
    "scroll_lock": ("Scroll Lock", keyboard.Key.scroll_lock),
}

# Configuration
WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large
BASE_DIR = Path(__file__).parent
VOCAB_FILE = BASE_DIR / "vocabulary.txt"
STATUS_FILE = BASE_DIR / "status"
INDICATOR_SCRIPT = BASE_DIR / "indicator.py"
HISTORY_SIZE = 10  # Number of recent transcriptions to remember
CLAUDE_CLI = Path.home() / ".local" / "bin" / "claude"
CLAUDE_SESSION_DIR = BASE_DIR / "claude-session"  # Directory for Claude session persistence

# Piper TTS configuration
PIPER_CMD = BASE_DIR / "venv" / "bin" / "piper"
PIPER_MODEL = BASE_DIR / "piper-voices" / "en_US-lessac-medium.onnx"

# Auto-listen configuration
AUTO_LISTEN_SECONDS = 4  # Seconds to auto-listen after TTS completes
SILENCE_THRESHOLD = -35  # dB threshold - below this is considered silence

# Config file
CONFIG_FILE = BASE_DIR / "config.json"


def load_config():
    """Load configuration from file."""
    default = {
        "tts_backend": "piper",
        "openai_voice": "nova",
        "ai_mode": "claude",  # "claude", "realtime", or "interview"
        "ptt_key": "ctrl_r",
        "ai_key": "shift_r",
        "interrupt_key": "escape",
        "indicator_style": "floating",
        "indicator_x": None,
        "indicator_y": None,
        "smart_transcription": False,  # Use AI to fix transcription errors
        "dictation_mode": "dictate",  # "dictate", "prompt", or "stream"
        "save_audio": False,  # Save recordings to disk
        "audio_dir": "~/Audio/push-to-talk",  # Where to save recordings
        "interview_topic": "",  # Topic for interview mode
        "interview_context_dirs": [],  # Repo paths for interview context
        "conversation_project_dir": "",  # Project directory for conversation mode
        "live_model": "claude-sonnet-4-5-20250929",  # Claude model for live voice
        "live_stt": "deepgram",          # "deepgram" or "whisper"
        "live_tts": "openai",            # "openai" or "piper"
        "live_fillers": True,            # Play filler sounds during processing
        "live_barge_in": True,           # Allow speaking over AI response
        "auto_start_listening": True,    # Auto-start live session on service startup
        "live_auto_mute": True,          # Allow key-based muting (tap/hold to mute)
        "verbal_hooks": [
            # Example hooks - customize these:
            # {"trigger": "open browser", "command": "xdg-open https://google.com"},
            # {"trigger": "lock screen", "command": "gnome-screensaver-command -l"},
            # {"trigger": "search for *", "command": "xdg-open 'https://google.com/search?q={}'"},
        ],
    }
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = json.load(f)
                merged = {**default, **config}
                # Migrate old "live" dictation mode to "dictate"
                if merged.get('dictation_mode') == 'live':
                    merged['dictation_mode'] = 'dictate'
                    save_config(merged)
                return merged
    except Exception as e:
        print(f"Error loading config: {e}", flush=True)
    return default


def save_config(config):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}", flush=True)


def speak_openai(text, voice="nova", save_path=None):
    """Speak text using OpenAI TTS API. Optionally save audio to save_path."""
    api_key = get_openai_api_key()
    if not api_key or not OPENAI_AVAILABLE:
        print("OpenAI TTS not available", flush=True)
        return None

    try:
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="pcm"
        )

        # Play the audio directly
        process = subprocess.Popen(
            ['aplay', '-r', '24000', '-f', 'S16_LE', '-t', 'raw', '-q'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Stream the audio to aplay, optionally collecting for save
        audio_buffer = [] if save_path else None
        for chunk in response.iter_bytes(chunk_size=4096):
            process.stdin.write(chunk)
            if audio_buffer is not None:
                audio_buffer.append(chunk)
        process.stdin.close()

        # Save to wav if requested
        if save_path and audio_buffer:
            pcm_data = b''.join(audio_buffer)
            with wave.open(str(save_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)
                wf.writeframes(pcm_data)
            print(f"TTS audio saved: {save_path}", flush=True)

        return process

    except Exception as e:
        print(f"OpenAI TTS error: {e}", flush=True)
        return None

def show_prompt_dialog(text):
    """Show a dialog to preview/edit transcription before typing. Returns edited text or None if cancelled."""
    try:
        result = subprocess.run(
            ['zenity', '--entry',
             '--title=Push-to-Talk',
             '--text=Edit transcription:',
             '--entry-text', text,
             '--width=500'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None  # Cancelled
    except Exception as e:
        print(f"Prompt dialog error: {e}", flush=True)
        return text  # Fallback to original


def check_verbal_hooks(text, hooks):
    """Check if text matches any verbal hook triggers and execute the command."""
    if not hooks:
        return False

    text_lower = text.lower().strip().rstrip('.,!?')

    for hook in hooks:
        trigger = hook.get('trigger', '').lower()
        command = hook.get('command', '')

        if not trigger or not command:
            continue

        # Handle wildcard triggers (e.g., "search for *")
        if '*' in trigger:
            prefix = trigger.replace('*', '').strip()
            if text_lower.startswith(prefix):
                # Extract the wildcard portion
                captured = text[len(prefix):].strip().rstrip('.,!?')
                if captured:
                    # Replace {} in command with captured text
                    final_command = command.replace('{}', captured)
                    print(f"Verbal hook matched: '{trigger}' → {final_command}", flush=True)
                    subprocess.Popen(final_command, shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Notification
                    subprocess.Popen(
                        ['notify-send', '-t', '2000', 'Voice Command', f'{trigger}: {captured}'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    return True
        else:
            # Exact match (case-insensitive)
            if text_lower == trigger or text_lower.startswith(trigger + ' ') or text_lower.startswith(trigger + ','):
                print(f"Verbal hook matched: '{trigger}' → {command}", flush=True)
                subprocess.Popen(command, shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Notification
                subprocess.Popen(
                    ['notify-send', '-t', '2000', 'Voice Command', trigger],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True

    return False


def smart_transcribe(text):
    """Use AI to fix transcription errors and extract intended meaning."""
    api_key = get_openai_api_key()
    if not api_key or not OPENAI_AVAILABLE:
        return text

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Fix any transcription errors in the user's dictated text. "
                               "Correct misspellings, wrong words that sound similar, and grammar issues. "
                               "Preserve the original meaning and tone. Output ONLY the corrected text, nothing else. "
                               "If the text looks correct, output it unchanged."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=500,
            temperature=0.1
        )
        corrected = response.choices[0].message.content.strip()
        if corrected and corrected != text:
            print(f"Smart correction: '{text}' → '{corrected}'", flush=True)
            return corrected
        return text
    except Exception as e:
        print(f"Smart transcription error: {e}", flush=True)
        return text


# Correction patterns - when detected, extract and learn the word
CORRECTION_PATTERNS = [
    r"^(?:add word|learn word|remember)[:\s]+(.+)",  # explicit commands (start of string)
    r"^(?:correction|correct)[:\s]+(.+)",  # "correction: word"
]

# Indicator process
indicator_process = None


def set_status(status):
    """Update the status indicator."""
    try:
        STATUS_FILE.write_text(status)
    except:
        pass


def get_audio_level(audio_file):
    """Get the maximum audio level in dB using sox."""
    try:
        result = subprocess.run(
            ['sox', audio_file, '-n', 'stat'],
            capture_output=True,
            text=True
        )
        # sox outputs stats to stderr
        for line in result.stderr.splitlines():
            if 'Maximum amplitude' in line:
                amp = float(line.split(':')[1].strip())
                if amp > 0:
                    return 20 * math.log10(amp)
                return -100  # Effectively silent
        return -100
    except Exception as e:
        print(f"Error checking audio level: {e}", flush=True)
        return 0  # Assume not silent on error


def start_indicator():
    """Start the status indicator process."""
    global indicator_process
    if INDICATOR_SCRIPT.exists():
        indicator_process = subprocess.Popen(
            ['python3', str(INDICATOR_SCRIPT)],
            stdout=None,  # Inherit parent stdout for debugging
            stderr=None   # Inherit parent stderr for debugging
        )
        print("Status indicator started", flush=True)


def stop_indicator():
    """Stop the status indicator process."""
    global indicator_process
    if indicator_process:
        indicator_process.terminate()
        try:
            indicator_process.wait(timeout=2)
        except:
            indicator_process.kill()
        indicator_process = None
    # Clean up status file
    try:
        STATUS_FILE.unlink()
    except:
        pass


class VocabularyManager:
    """Manages custom vocabulary for Whisper."""

    def __init__(self, vocab_file):
        self.vocab_file = Path(vocab_file)
        self.words = set()
        self.load()

    def load(self):
        """Load vocabulary from file."""
        self.words = set()
        if self.vocab_file.exists():
            for line in self.vocab_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    self.words.add(line)
        print(f"Loaded {len(self.words)} vocabulary words", flush=True)

    def add(self, word):
        """Add a word to vocabulary."""
        word = word.strip().strip('.,!?:;')
        if word and word not in self.words and len(word) > 1:
            self.words.add(word)
            self.save()
            print(f"Added to vocabulary: {word}", flush=True)
            # Show notification
            subprocess.Popen(
                ['notify-send', '-t', '2000', 'Vocabulary Updated', f'Added: {word}'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        return False

    def save(self):
        """Save vocabulary to file."""
        # Read existing file to preserve comments
        lines = []
        if self.vocab_file.exists():
            for line in self.vocab_file.read_text().splitlines():
                if line.startswith('#') or not line.strip():
                    lines.append(line)

        # Add all words
        for word in sorted(self.words):
            if word not in lines:
                lines.append(word)

        self.vocab_file.write_text('\n'.join(lines) + '\n')

    def get_prompt(self):
        """Generate initial prompt for Whisper."""
        if not self.words:
            return None
        return "Transcript may include: " + ", ".join(sorted(self.words)) + "."


class InterviewSession:
    """Manages state for an AI interview session."""

    WRAP_SIGNALS = [
        "that's a wrap", "thats a wrap", "end interview", "end the interview",
        "stop interview", "stop the interview", "we're done", "were done",
        "that's all", "thats all", "wrap it up", "wrap up",
    ]

    def __init__(self, topic, context_dirs=None, audio_dir=None):
        self.session_id = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.topic = topic
        self.context_dirs = context_dirs or []
        self.sequence = 0
        self.transcript = []
        self.active = True
        self.status = 'starting'  # starting, idle, recording, processing

        # Directories
        base_dir = Path(audio_dir or '~/Audio/push-to-talk').expanduser()
        self.session_dir = base_dir / 'sessions' / self.session_id
        self.claude_session_dir = self.session_dir / '.claude-session'
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.claude_session_dir.mkdir(parents=True, exist_ok=True)

    def next_sequence(self):
        """Increment and return sequence number."""
        self.sequence += 1
        return self.sequence

    def add_entry(self, role, text, audio_path=None):
        """Append an entry to the transcript."""
        entry = {
            'role': role,
            'text': text,
            'audio_path': str(audio_path) if audio_path else None,
            'time': time.strftime('%H:%M:%S'),
            'sequence': self.sequence,
        }
        self.transcript.append(entry)
        return entry

    def is_wrap_signal(self, text):
        """Check if the text signals the user wants to end the interview."""
        text_lower = text.lower().strip().rstrip('.,!?')
        return any(signal in text_lower for signal in self.WRAP_SIGNALS)

    def build_system_prompt(self):
        """Build the interviewer system prompt, optionally reading context from dirs."""
        context_parts = []
        for dir_path in self.context_dirs:
            claude_md = Path(dir_path).expanduser() / 'CLAUDE.md'
            if claude_md.exists():
                try:
                    context_parts.append(f"--- Context from {dir_path} ---\n{claude_md.read_text()}")
                except Exception:
                    pass

        context_block = ""
        if context_parts:
            context_block = "\n\nYou have the following background context about the guest's work:\n" + "\n\n".join(context_parts)

        prompt = f"""You are an engaging, natural podcast interviewer. You are interviewing a guest about: {self.topic}

Your style:
- Be warm, curious, and conversational — like a great podcast host
- Ask ONE question at a time
- Follow up on interesting answers before changing topics
- React naturally to what the guest says (brief acknowledgment before next question)
- Keep your responses concise — 2-3 sentences max
- No markdown formatting — speak naturally as if this is audio
- Structure the conversation: warm intro → deep exploration → wrap-up
- When the guest signals they want to wrap up, give a brief, warm closing statement summarizing key takeaways{context_block}"""

        return prompt

    def save_metadata(self):
        """Write session metadata to metadata.json."""
        metadata = {
            'session_id': self.session_id,
            'topic': self.topic,
            'context_dirs': self.context_dirs,
            'total_segments': self.sequence,
            'transcript': self.transcript,
            'created': self.session_id,
            'completed': time.strftime('%Y-%m-%d_%H-%M-%S'),
        }
        meta_path = self.session_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {meta_path}", flush=True)


class ConversationSession:
    """Manages state for a conversation with Claude + full tool access."""

    END_SIGNALS = [
        "end conversation", "stop conversation", "goodbye", "bye",
        "that's all", "thats all", "we're done", "were done",
    ]

    def __init__(self, project_dir=None):
        self.session_id = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.active = True
        self.turn_count = 0
        self.project_dir = Path(project_dir or Path.home()).expanduser().resolve()

    def build_system_prompt(self):
        """Build a voice-friendly system prompt for conversation mode."""
        return (
            f"You are a chill, sharp dev buddy talking through a voice interface. "
            f"You're helpful but not wordy — like a coworker who just answers the question. "
            f"Working directory: {self.project_dir}\n\n"
            f"CRITICAL — your responses are spoken aloud via TTS:\n"
            f"- Max 1-2 short sentences. Be extremely brief.\n"
            f"- Plain spoken language only — zero markdown, code blocks, lists, or formatting\n"
            f"- Never read out file contents, paths, or command output verbatim — summarize in plain English\n"
            f"- When you do something, just confirm briefly: 'Done' or 'Fixed it' or 'Yeah, there are 12 files in there'\n"
            f"- Only elaborate if explicitly asked for detail"
        )

    def is_end_signal(self, text):
        """Check if the text signals the user wants to end the conversation."""
        text_lower = text.lower().strip().rstrip('.,!?')
        return any(signal in text_lower for signal in self.END_SIGNALS)


class PushToTalk:
    def __init__(self):
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_lock = threading.Lock()
        self.other_keys_pressed = set()

        # AI assistant mode tracking
        self.ai_mode = False  # True when recording for AI assistant
        self.ctrl_r_pressed = False
        self.shift_r_pressed = False
        self._ptt_muted_live = False  # True when live session muted for PTT dictation
        self.tts_process = None  # Track TTS process for interruption

        # Interview mode
        self.interview_session = None

        # Conversation mode
        self.conversation_session = None

        # Live voice session
        self.live_session = None
        self.live_thread = None
        self.config_watcher_running = True
        self._last_config_mtime = 0

        # Stream mode state
        self.stream_active = False
        self.stream_stop_event = threading.Event()
        self.last_transcribed_words = []  # For deduplication

        # Load config
        self.config = load_config()
        tts = self.config.get('tts_backend', 'piper')
        ai_mode = self.config.get('ai_mode', 'claude')
        print(f"TTS backend: {tts}, AI mode: {ai_mode}", flush=True)

        # Resolve hotkeys from config
        ptt_key_id = self.config.get('ptt_key', 'ctrl_r')
        ai_key_id = self.config.get('ai_key', 'shift_r')
        interrupt_key_id = self.config.get('interrupt_key', 'escape')

        # Validate ptt and ai keys are different
        if ptt_key_id == ai_key_id:
            print("WARNING: PTT and AI keys are the same, falling back to defaults", flush=True)
            ptt_key_id = 'ctrl_r'
            ai_key_id = 'shift_r'

        self.ptt_key = MODIFIER_KEY_OPTIONS.get(ptt_key_id, MODIFIER_KEY_OPTIONS['ctrl_r'])[1]
        self.ai_key = MODIFIER_KEY_OPTIONS.get(ai_key_id, MODIFIER_KEY_OPTIONS['shift_r'])[1]
        self.interrupt_key = INTERRUPT_KEY_OPTIONS.get(interrupt_key_id, INTERRUPT_KEY_OPTIONS['escape'])[1]

        ptt_name = MODIFIER_KEY_OPTIONS.get(ptt_key_id, MODIFIER_KEY_OPTIONS['ctrl_r'])[0]
        ai_name = MODIFIER_KEY_OPTIONS.get(ai_key_id, MODIFIER_KEY_OPTIONS['shift_r'])[0]
        interrupt_name = INTERRUPT_KEY_OPTIONS.get(interrupt_key_id, INTERRUPT_KEY_OPTIONS['escape'])[0]
        print(f"Hotkeys: PTT={ptt_name}, AI={ai_name}, Interrupt={interrupt_name}", flush=True)

        # Diagnostic output for Realtime mode
        print(f"  REALTIME_AVAILABLE (module imported): {REALTIME_AVAILABLE}", flush=True)
        if REALTIME_AVAILABLE:
            print(f"  realtime_available() (API key + deps): {realtime_available()}", flush=True)
        print(f"  OpenAI API key found: {get_openai_api_key() is not None}", flush=True)
        if ai_mode == 'realtime' and not (REALTIME_AVAILABLE and realtime_available()):
            print("  WARNING: ai_mode is 'realtime' but Realtime API is not available!", flush=True)

        # Realtime session
        self.realtime_session = None
        self.realtime_thread = None

        # Vocabulary and history
        self.vocab = VocabularyManager(VOCAB_FILE)
        self.history = deque(maxlen=HISTORY_SIZE)

        print("Loading Whisper model...", flush=True)
        self.model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model '{WHISPER_MODEL}' loaded.", flush=True)
        print(f"Push-to-Talk ready. Hold {ptt_name} to dictate.", flush=True)

        # Ensure mic is unmuted on startup (cleanup from interrupted sessions)
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                       capture_output=True)

        # Set idle status
        set_status('idle')

        # Start config watcher for live mode auto-start
        self.config_watcher_thread = threading.Thread(target=self._watch_config, daemon=True)
        self.config_watcher_thread.start()

        # Auto-start live session if configured
        if self.config.get('auto_start_listening', True) and self.config.get('ai_mode') == 'live':
            print("Auto-starting live session (auto_start_listening=true)", flush=True)
            self.start_live_session()

    def _watch_config(self):
        """Watch config.json for ai_mode changes and live restart signals."""
        config_path = str(CONFIG_FILE)
        restart_signal = BASE_DIR / "status"
        while self.config_watcher_running:
            try:
                # Check for restart_live signal from overlay
                if restart_signal.exists():
                    try:
                        content = restart_signal.read_text().strip()
                        if content == "restart_live":
                            restart_signal.write_text("idle")  # Clear signal
                            if not self.live_session:
                                print("Config watcher: Restart live session requested", flush=True)
                                self.start_live_session()
                    except Exception:
                        pass

                mtime = os.path.getmtime(config_path)
                if mtime != self._last_config_mtime:
                    self._last_config_mtime = mtime
                    new_config = load_config()
                    old_ai_mode = self.config.get('ai_mode', 'claude')
                    new_ai_mode = new_config.get('ai_mode', 'claude')
                    self.config = new_config
                    if new_ai_mode == 'live' and old_ai_mode != 'live':
                        if not self.live_session:
                            print("Config watcher: Live mode selected, auto-starting session", flush=True)
                            self.start_live_session()
                    elif old_ai_mode == 'live' and new_ai_mode != 'live':
                        if self.live_session:
                            print("Config watcher: Left live mode, stopping session", flush=True)
                            self.stop_live_session()
            except Exception:
                pass  # Don't crash the watcher
            time.sleep(0.5)

    def toggle_tts_backend(self):
        """Toggle between Piper and OpenAI TTS."""
        current = self.config.get('tts_backend', 'piper')
        if current == 'piper':
            if OPENAI_AVAILABLE and get_openai_api_key():
                self.config['tts_backend'] = 'openai'
                print("Switched to OpenAI TTS", flush=True)
            else:
                print("OpenAI TTS not available (missing API key)", flush=True)
                return
        else:
            self.config['tts_backend'] = 'piper'
            print("Switched to Piper TTS", flush=True)
        save_config(self.config)

    def toggle_ai_mode(self):
        """Toggle between Claude and Realtime AI modes."""
        current = self.config.get('ai_mode', 'claude')
        if current == 'claude':
            if REALTIME_AVAILABLE and realtime_available():
                self.config['ai_mode'] = 'realtime'
                print("Switched to Realtime AI mode", flush=True)
            else:
                print("Realtime mode not available (missing API key or dependencies)", flush=True)
                return
        else:
            self.config['ai_mode'] = 'claude'
            print("Switched to Claude AI mode", flush=True)
        save_config(self.config)

    def start_realtime_session(self):
        """Start an OpenAI Realtime session in a background thread."""
        if not REALTIME_AVAILABLE:
            print("ERROR: openai_realtime module not available", flush=True)
            set_status('error')
            return False

        if not realtime_available():
            print("ERROR: Realtime API check failed (websockets or API key issue)", flush=True)
            set_status('error')
            return False

        api_key = get_openai_api_key()
        if not api_key:
            print("ERROR: OpenAI API key not found for Realtime session", flush=True)
            set_status('error')
            return False

        # Create session object immediately so toggle check works
        self.realtime_session = RealtimeSession(api_key, on_status=set_status)

        def run_session():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.realtime_session.run())
            finally:
                loop.close()
                self.realtime_session = None

        self.realtime_thread = threading.Thread(target=run_session, daemon=True)
        self.realtime_thread.start()
        return True

    def stop_realtime_session(self):
        """Stop the current Realtime session."""
        if self.realtime_session:
            self.realtime_session.stop()
            self.realtime_session = None
        if self.realtime_thread:
            self.realtime_thread.join(timeout=2)
            self.realtime_thread = None
        # Always unmute mic when stopping realtime
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                       capture_output=True)
        set_status('idle')

    def start_live_session(self):
        """Start a live voice conversation session."""
        if not LIVE_SESSION_AVAILABLE:
            print("ERROR: live_session module not available", flush=True)
            set_status('error')
            return False

        openai_key = get_openai_api_key()
        if not openai_key:
            print("No OpenAI API key found for live session TTS", flush=True)
            prompt_api_key()
            return False

        deepgram_key = get_deepgram_api_key()
        if not deepgram_key:
            print("No Deepgram API key found for live session STT", flush=True)
            self.show_error("Deepgram API key required. Set DEEPGRAM_API_KEY env var.")
            return False

        voice = self.config.get('openai_voice', 'ash')
        model = self.config.get('live_model', 'claude-sonnet-4-5-20250929')
        fillers = self.config.get('live_fillers', True)
        barge_in = self.config.get('live_barge_in', True)
        self.live_session = LiveSession(
            openai_api_key=openai_key,
            deepgram_api_key=deepgram_key,
            voice=voice,
            model=model,
            on_status=set_status,
            fillers_enabled=fillers,
            barge_in_enabled=barge_in,
            whisper_model=self.model,
        )

        # Ensure TaskManager singleton is initialized for this session
        from task_manager import TaskManager
        TaskManager()  # Initialize singleton if not already

        def run_session():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.live_session.run())
            finally:
                loop.close()
                self.live_session = None
                set_status('idle')

        self.live_thread = threading.Thread(target=run_session, daemon=True)
        self.live_thread.start()
        return True

    def stop_live_session(self):
        """Stop the live voice session cleanly."""
        if self.live_session:
            self.live_session.stop()
            self.live_session = None
        if self.live_thread:
            self.live_thread.join(timeout=2)
            self.live_thread = None
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                       capture_output=True)
        set_status('idle')

    # No timers. All logic runs synchronously in the pynput thread.
    # Press: record time + state, unmute if muted (optimistic for hold-to-talk).
    # Release: check elapsed. Tap (<500ms) = cycle. Hold (>=500ms) = mute+flush.

    def show_error(self, message):
        """Show error notification and briefly flash error status."""
        set_status('error')
        # Desktop notification (normal urgency so it auto-dismisses)
        subprocess.Popen(
            ['notify-send', '-t', '3000', 'Push-to-Talk Error', message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Return to idle after 2 seconds
        threading.Timer(2.0, lambda: set_status('idle')).start()

    def speak(self, text, save_path=None):
        """Speak text using configured TTS backend. Optionally save audio to save_path."""
        tts_backend = self.config.get('tts_backend', 'piper')

        if tts_backend == 'openai' and OPENAI_AVAILABLE and get_openai_api_key():
            voice = self.config.get('openai_voice', 'nova')
            return speak_openai(text, voice, save_path=save_path)
        else:
            # Use Piper
            if save_path:
                # Write to file first, then play from file
                piper_proc = subprocess.run(
                    f'echo {subprocess.list2cmdline([text])} | "{PIPER_CMD}" --model "{PIPER_MODEL}" --output_file "{save_path}" 2>/dev/null',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"TTS audio saved: {save_path}", flush=True)
                return subprocess.Popen(
                    ['aplay', '-q', str(save_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                return subprocess.Popen(
                    f'echo {subprocess.list2cmdline([text])} | "{PIPER_CMD}" --model "{PIPER_MODEL}" --output-raw 2>/dev/null | aplay -r 22050 -f S16_LE -t raw -q',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

    def detect_correction(self, text):
        """Check if text contains a correction command and extract the word."""
        text_lower = text.lower().strip()

        for pattern in CORRECTION_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                corrected = match.group(1).strip().strip('.,!?:;')
                # Get the original case from the text
                orig_match = re.search(pattern, text, re.IGNORECASE)
                if orig_match:
                    corrected = orig_match.group(1).strip().strip('.,!?:;')
                return corrected
        return None

    def save_audio(self, audio_file, mode, text=None):
        """Save audio file and optional transcription text to configured directory."""
        if not self.config.get('save_audio', False):
            return
        try:
            audio_dir = Path(self.config.get('audio_dir', '~/Audio/push-to-talk')).expanduser()
            audio_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            base_name = f"ptt_{timestamp}_{mode}"
            dest = audio_dir / f"{base_name}.wav"
            shutil.copy2(audio_file, dest)
            print(f"Audio saved: {dest}", flush=True)
            if text:
                txt_dest = audio_dir / f"{base_name}.txt"
                txt_dest.write_text(text)
        except Exception as e:
            print(f"Error saving audio: {e}", flush=True)

    def start_recording(self, force=False):
        if self.recording or self.stream_active:
            print("Already recording, skipping", flush=True)
            return

        # Reload config to pick up any changes from indicator
        self.config = load_config()

        # Check for stream mode
        if self.config.get('dictation_mode') == 'stream' and not force:
            self.start_stream_mode()
            return

        self.recording = True
        self.record_start_time = time.time()
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.temp_file.close()

        self.record_process = subprocess.Popen([
            'pw-record',
            '--format', 's16',
            '--rate', '44100',
            '--channels', '2',
            self.temp_file.name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        set_status('recording')
        print("Recording...", flush=True)


    def transcribe_and_type(self, temp_file):
        if not temp_file or not os.path.exists(temp_file.name):
            set_status('idle')
            return

        try:
            file_size = os.path.getsize(temp_file.name)
            print(f"Recording file size: {file_size} bytes", flush=True)
            if file_size < 5000:
                set_status('idle')
                print("Recording too short, skipping.", flush=True)
                return

            # Convert to 16kHz mono for Whisper
            converted_file = temp_file.name.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file.name,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Build prompt from vocabulary and recent history
            prompt = self.vocab.get_prompt()
            if self.history:
                context = " ".join(self.history)
                if prompt:
                    prompt = f"{prompt} Recent context: {context}"
                else:
                    prompt = context

            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False,
                    initial_prompt=prompt
                )

            text = result['text'].strip()

            if text:
                print(f"Transcribed: {text}", flush=True)

                # Built-in voice commands
                text_lower = text.lower().strip().rstrip('.,!?')
                if text_lower in ['clear', 'clear input', 'clear that', 'clear all']:
                    print(f"Voice command: clear input", flush=True)
                    # Ctrl+U clears line in terminals/readline, or delete last transcription
                    if self.history:
                        # Delete exactly what we last typed
                        last_text = self.history[-1] + ' '  # Include trailing space
                        delete_count = len(last_text)
                        subprocess.run(['xdotool', 'key', '--repeat', str(delete_count), '--delay', '5', 'BackSpace'], check=True)
                        self.history.pop()
                    else:
                        # Fallback: Ctrl+U
                        subprocess.run(['xdotool', 'key', 'ctrl+u'], check=True)
                    set_status('success')
                    return
                if text_lower in ['undo', 'undo that']:
                    print(f"Voice command: undo", flush=True)
                    subprocess.run(['xdotool', 'key', 'ctrl+z'], check=True)
                    set_status('success')
                    return
                if text_lower in ['select all', 'select everything']:
                    print(f"Voice command: select all", flush=True)
                    subprocess.run(['xdotool', 'key', 'ctrl+a'], check=True)
                    set_status('success')
                    return
                if text_lower in ['copy', 'copy that']:
                    print(f"Voice command: copy", flush=True)
                    subprocess.run(['xdotool', 'key', 'ctrl+c'], check=True)
                    set_status('success')
                    return
                if text_lower in ['paste', 'paste that']:
                    print(f"Voice command: paste", flush=True)
                    subprocess.run(['xdotool', 'key', 'ctrl+v'], check=True)
                    set_status('success')
                    return
                if text_lower in ['new line', 'newline', 'enter']:
                    print(f"Voice command: new line", flush=True)
                    subprocess.run(['xdotool', 'key', 'Return'], check=True)
                    set_status('success')
                    return
                if text_lower in ['dictate mode', 'go dictate', 'dictation mode']:
                    print(f"Voice command: switching to dictate mode", flush=True)
                    self.config['dictation_mode'] = 'dictate'
                    save_config(self.config)
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Dictate mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    set_status('success')
                    return
                if text_lower in ['prompt mode', 'preview mode', 'go prompt']:
                    print(f"Voice command: switching to prompt mode", flush=True)
                    self.config['dictation_mode'] = 'prompt'
                    save_config(self.config)
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Prompt mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    set_status('success')
                    return
                if text_lower in ['stream mode', 'go stream', 'streaming mode']:
                    print(f"Voice command: switching to stream mode", flush=True)
                    self.config['dictation_mode'] = 'stream'
                    save_config(self.config)
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Stream mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    set_status('success')
                    return
                if text_lower in ['live mode', 'go live', 'going live']:
                    print(f"Voice command: switching to live AI mode", flush=True)
                    self.config['ai_mode'] = 'live'
                    save_config(self.config)
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Live mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    set_status('success')
                    return
                if text_lower in ['save audio', 'start saving']:
                    print(f"Voice command: enabling audio save", flush=True)
                    self.config['save_audio'] = True
                    save_config(self.config)
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Audio saving enabled'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    set_status('success')
                    return
                if text_lower in ['stop saving', 'stop recording']:
                    print(f"Voice command: disabling audio save", flush=True)
                    self.config['save_audio'] = False
                    save_config(self.config)
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Audio saving disabled'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    set_status('success')
                    return

                # Check for verbal hooks
                hooks = self.config.get('verbal_hooks', [])
                if check_verbal_hooks(text, hooks):
                    set_status('success')
                    return

                # Apply smart transcription if enabled
                if self.config.get('smart_transcription', False):
                    text = smart_transcribe(text)

                # Check for correction commands
                correction = self.detect_correction(text)
                if correction:
                    # Replace the last word from previous transcription
                    if self.history:
                        last_transcription = self.history[-1]
                        last_words = last_transcription.split()
                        if last_words:
                            old_word = last_words[-1].rstrip('.,!?:;')
                            # Delete the old word (including trailing space) and type correction
                            delete_len = len(old_word) + 1  # +1 for the space after
                            subprocess.run(['xdotool', 'key', '--repeat', str(delete_len), 'BackSpace'], check=True)
                            subprocess.run(['xdotool', 'type', '--delay', '12', '--', correction + ' '], check=True)
                            # Update history with corrected version
                            last_words[-1] = correction
                            self.history[-1] = ' '.join(last_words)
                            print(f"Replaced '{old_word}' with '{correction}'", flush=True)

                    self.vocab.add(correction)
                    set_status('success')
                    return

                # Prompt mode: show dialog to edit before typing
                if self.config.get('dictation_mode', 'dictate') == 'prompt':
                    text = show_prompt_dialog(text)
                    if text is None:
                        print("Prompt cancelled", flush=True)
                        set_status('idle')
                        return

                # Add to history
                self.history.append(text)

                # Save audio before typing (so file still exists)
                self.save_audio(temp_file.name, 'dictation', text)

                # Type the text
                subprocess.run(['xdotool', 'type', '--delay', '12', '--', text + ' '],
                             check=True)
                self.other_keys_pressed.clear()
                set_status('success')
            else:
                print("No speech detected.", flush=True)
                set_status('idle')

        except Exception as e:
            print(f"Error during transcription: {e}", flush=True)
            set_status('error')
        finally:
            for f in [temp_file.name, temp_file.name.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass

    def stop_tts(self):
        """Stop any ongoing TTS playback."""
        if self.tts_process and self.tts_process.poll() is None:
            self.tts_process.terminate()
            try:
                self.tts_process.wait(timeout=1)
            except:
                self.tts_process.kill()
            self.tts_process = None
            print("TTS interrupted", flush=True)
            return True  # Was interrupted
        return False

    def start_stream_mode(self):
        """Start streaming transcription - records and transcribes in chunks."""
        self.stream_active = True
        self.stream_stop_event.clear()
        self.last_transcribed_words = []
        set_status('recording')
        print("Stream mode started", flush=True)

        CHUNK_DURATION = 3.0  # seconds per chunk
        OVERLAP = 0.5  # overlap between chunks

        def stream_worker():
            chunk_num = 0
            while not self.stream_stop_event.is_set():
                # Record a chunk
                chunk_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                chunk_file.close()

                record_proc = subprocess.Popen([
                    'pw-record',
                    '--format', 's16',
                    '--rate', '44100',
                    '--channels', '2',
                    chunk_file.name
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Wait for chunk duration (or stop signal)
                start_time = time.time()
                while time.time() - start_time < CHUNK_DURATION:
                    if self.stream_stop_event.is_set():
                        break
                    time.sleep(0.1)

                record_proc.terminate()
                record_proc.wait()

                # Check if we got any audio
                if os.path.getsize(chunk_file.name) < 5000:
                    os.unlink(chunk_file.name)
                    continue

                # Transcribe in background
                chunk_num += 1
                threading.Thread(
                    target=self.transcribe_stream_chunk,
                    args=(chunk_file.name, chunk_num),
                    daemon=True
                ).start()

                # Sleep for overlap (so next chunk overlaps)
                if not self.stream_stop_event.is_set():
                    time.sleep(CHUNK_DURATION - OVERLAP)

            self.stream_active = False
            set_status('idle')
            print("Stream mode stopped", flush=True)

        threading.Thread(target=stream_worker, daemon=True).start()

    def stop_stream_mode(self):
        """Stop streaming transcription."""
        self.stream_stop_event.set()

    def transcribe_stream_chunk(self, chunk_file, chunk_num):
        """Transcribe a single chunk and type new words."""
        try:
            # Check audio level - skip silent chunks
            audio_level = get_audio_level(chunk_file)
            if audio_level < SILENCE_THRESHOLD:
                print(f"Stream chunk {chunk_num}: silent ({audio_level:.1f}dB), skipping", flush=True)
                return

            # Convert to 16kHz mono
            converted_file = chunk_file.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', chunk_file,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Transcribe
            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False
                )

            text = result['text'].strip()
            if text:
                # Check for mode-switching voice commands in stream mode
                text_lower = text.lower().strip().rstrip('.,!?')
                if text_lower in ['dictate mode', 'go dictate', 'dictation mode']:
                    print(f"Stream: switching to dictate mode", flush=True)
                    self.config['dictation_mode'] = 'dictate'
                    save_config(self.config)
                    self.stop_stream_mode()
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Dictate mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                if text_lower in ['prompt mode', 'preview mode', 'go prompt']:
                    print(f"Stream: switching to prompt mode", flush=True)
                    self.config['dictation_mode'] = 'prompt'
                    save_config(self.config)
                    self.stop_stream_mode()
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Prompt mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                if text_lower in ['live mode', 'go live', 'going live']:
                    print(f"Stream: switching to live AI mode", flush=True)
                    self.config['ai_mode'] = 'live'
                    save_config(self.config)
                    self.stop_stream_mode()
                    subprocess.Popen(['notify-send', '-t', '2000', 'Push-to-Talk', 'Live mode activated'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return

                words = text.split()

                # Deduplicate: find new words not in last transcription
                # Use simple overlap detection
                new_words = []
                overlap_found = False

                for i, word in enumerate(words):
                    # Check if this word starts a sequence matching end of last transcription
                    if not overlap_found and self.last_transcribed_words:
                        # Look for overlap point
                        for j in range(min(5, len(self.last_transcribed_words))):
                            check_idx = len(self.last_transcribed_words) - 1 - j
                            if check_idx >= 0 and word.lower().rstrip('.,!?') == self.last_transcribed_words[check_idx].lower().rstrip('.,!?'):
                                # Found overlap, take words after this point
                                new_words = words[i+1:]
                                overlap_found = True
                                break
                        if overlap_found:
                            break

                if not overlap_found:
                    # No overlap found, use all words (first chunk or gap)
                    new_words = words

                if new_words:
                    new_text = ' '.join(new_words)
                    print(f"Stream chunk {chunk_num}: '{new_text}'", flush=True)
                    self.save_audio(chunk_file, 'stream', new_text)
                    subprocess.run(['xdotool', 'type', '--delay', '10', '--', new_text + ' '], check=True)

                # Update last transcribed words for next dedup
                self.last_transcribed_words = words[-10:]  # Keep last 10 words

        except Exception as e:
            print(f"Stream chunk error: {e}", flush=True)
        finally:
            for f in [chunk_file, chunk_file.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass

    def auto_listen_for_followup(self):
        """Auto-listen for a follow-up after TTS completes."""
        print(f"Auto-listening for {AUTO_LISTEN_SECONDS} seconds...", flush=True)
        set_status('recording')

        # Record for fixed duration
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()

        record_process = subprocess.Popen([
            'pw-record',
            '--format', 's16',
            '--rate', '44100',
            '--channels', '2',
            temp_file.name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for auto-listen duration
        time.sleep(AUTO_LISTEN_SECONDS)

        record_process.terminate()
        record_process.wait()

        # Check if we got any meaningful audio
        file_size = os.path.getsize(temp_file.name)
        audio_level = get_audio_level(temp_file.name)
        print(f"Auto-listen: size={file_size} bytes, level={audio_level:.1f} dB", flush=True)

        if file_size < 10000 or audio_level < SILENCE_THRESHOLD:
            print("No follow-up detected (silence), going idle.", flush=True)
            set_status('idle')
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return

        set_status('processing')

        # Convert and transcribe
        converted_file = temp_file.name.replace('.wav', '_16k.wav')
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file.name,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False
                )

            followup = result['text'].strip()

            if followup and len(followup) > 3:
                self.save_audio(temp_file.name, 'followup', followup)

            # Clean up temp files
            for f in [temp_file.name, converted_file]:
                try:
                    os.unlink(f)
                except:
                    pass

            if followup and len(followup) > 3:
                print(f"Follow-up detected: {followup}", flush=True)
                # Process as new AI question (recursive call via temp file trick)
                self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                self.temp_file.close()
                # Write a dummy file, we'll use the transcribed text directly
                self.process_ai_question(followup)
            else:
                print("No meaningful follow-up, going idle.", flush=True)
                set_status('idle')

        except Exception as e:
            print(f"Error in auto-listen: {e}", flush=True)
            set_status('idle')
            for f in [temp_file.name, converted_file]:
                try:
                    os.unlink(f)
                except:
                    pass

    def transcribe_and_ask_ai(self, temp_file):
        """Transcribe audio and send to Claude, then speak the response."""
        if not temp_file or not os.path.exists(temp_file.name):
            set_status('idle')
            return

        try:
            file_size = os.path.getsize(temp_file.name)
            print(f"AI mode - Recording file size: {file_size} bytes", flush=True)
            if file_size < 5000:
                set_status('idle')
                print("Recording too short, skipping.", flush=True)
                return

            # Convert to 16kHz mono for Whisper
            converted_file = temp_file.name.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file.name,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Transcribe
            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False
                )

            question = result['text'].strip()

            if question:
                print(f"AI Question: {question}", flush=True)
                self.save_audio(temp_file.name, 'ai', question)
                set_status('processing')

                # Send to Claude CLI (continue same session)
                try:
                    # Ensure session directory exists
                    CLAUDE_SESSION_DIR.mkdir(exist_ok=True)

                    result = subprocess.run(
                        [
                            str(CLAUDE_CLI),
                            '-c', '-p', question,
                            '--permission-mode', 'acceptEdits',
                            '--add-dir', str(Path.home() / '.claude'),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=str(CLAUDE_SESSION_DIR)
                    )
                    response = result.stdout.strip()

                    if response:
                        self.handle_ai_response(response)
                    else:
                        print("No response from Claude", flush=True)
                        set_status('error')

                except subprocess.TimeoutExpired:
                    print("Claude request timed out", flush=True)
                    set_status('error')
                except Exception as e:
                    print(f"Error calling Claude: {e}", flush=True)
                    set_status('error')
            else:
                print("No speech detected.", flush=True)
                set_status('idle')

        except Exception as e:
            print(f"Error during AI transcription: {e}", flush=True)
            set_status('error')
        finally:
            for f in [temp_file.name, temp_file.name.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass

    def process_ai_question(self, question):
        """Process an AI question and speak the response."""
        print(f"AI Question: {question}", flush=True)
        set_status('processing')

        try:
            CLAUDE_SESSION_DIR.mkdir(exist_ok=True)

            result = subprocess.run(
                [
                    str(CLAUDE_CLI),
                    '-c', '-p', question,
                    '--permission-mode', 'acceptEdits',
                    '--add-dir', str(Path.home() / '.claude'),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(CLAUDE_SESSION_DIR)
            )
            response = result.stdout.strip()

            if response:
                self.handle_ai_response(response)
            else:
                print("No response from Claude", flush=True)
                set_status('error')

        except subprocess.TimeoutExpired:
            print("Claude request timed out", flush=True)
            set_status('error')
        except Exception as e:
            print(f"Error calling Claude: {e}", flush=True)
            set_status('error')

    def handle_ai_response(self, response):
        """Speak an AI response and auto-listen for follow-up."""
        print(f"AI Response: {response[:100]}...", flush=True)
        set_status('success')

        # Speak the response and wait for completion
        self.tts_process = self.speak(response)
        if self.tts_process:
            print(f"TTS started (pid={self.tts_process.pid}), waiting...", flush=True)
            self.tts_process.wait()
            print(f"TTS finished (returncode={self.tts_process.returncode})", flush=True)
            self.tts_process = None
        else:
            print("TTS returned None — no audio played", flush=True)

        # Auto-listen for follow-up
        self.auto_listen_for_followup()

    # --- Interview Mode Methods ---

    def start_interview(self):
        """Start a new interview session."""
        topic = self.config.get('interview_topic', '')
        if not topic:
            # Ask for topic via zenity
            try:
                result = subprocess.run(
                    ['zenity', '--entry',
                     '--title=Interview Mode',
                     '--text=What topic should we discuss?',
                     '--width=400'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0 and result.stdout.strip():
                    topic = result.stdout.strip()
                else:
                    print("Interview cancelled - no topic", flush=True)
                    set_status('idle')
                    return
            except Exception as e:
                print(f"Zenity error: {e}", flush=True)
                set_status('idle')
                return

        context_dirs = self.config.get('interview_context_dirs', [])
        audio_dir = self.config.get('audio_dir', '~/Audio/push-to-talk')

        self.interview_session = InterviewSession(topic, context_dirs, audio_dir)

        subprocess.Popen(
            ['notify-send', '-t', '3000', 'Interview Started', topic],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Interview started: {topic} (session: {self.interview_session.session_id})", flush=True)

        # Write system prompt and start first question in background
        threading.Thread(
            target=self._interview_generate_question,
            args=("Read system_prompt.md and begin the interview.",),
            daemon=True
        ).start()

    def _interview_generate_question(self, prompt):
        """Generate an interviewer question via Claude CLI and speak it."""
        session = self.interview_session
        if not session or not session.active:
            return

        try:
            set_status('processing')

            # Write system prompt on first call
            system_prompt_path = session.claude_session_dir / 'system_prompt.md'
            if not system_prompt_path.exists():
                system_prompt_path.write_text(session.build_system_prompt())

            # Call Claude CLI in the session directory
            result = subprocess.run(
                [
                    str(CLAUDE_CLI),
                    '-c', '-p', prompt,
                    '--permission-mode', 'acceptEdits',
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(session.claude_session_dir)
            )
            response = result.stdout.strip()

            if not response:
                print("No response from Claude for interview", flush=True)
                set_status('error')
                return

            print(f"Interviewer: {response[:100]}...", flush=True)

            # Save TTS audio
            seq = session.next_sequence()
            audio_path = session.session_dir / f"{seq:03d}_interviewer.wav"

            set_status('success')
            self.tts_process = self.speak(response, save_path=str(audio_path))
            if self.tts_process:
                self.tts_process.wait()
                self.tts_process = None

            session.add_entry('interviewer', response, audio_path)

            # Ready for user to press PTT
            session.status = 'idle'
            set_status('idle')
            print("Waiting for user response (press PTT)...", flush=True)

        except subprocess.TimeoutExpired:
            print("Claude interview request timed out", flush=True)
            set_status('error')
        except Exception as e:
            print(f"Interview question error: {e}", flush=True)
            set_status('error')

    def _interview_process_answer(self, text, audio_file):
        """Process the user's answer and continue the interview."""
        session = self.interview_session
        if not session or not session.active:
            return

        try:
            # Copy user audio to session dir
            seq = session.next_sequence()
            user_audio_path = session.session_dir / f"{seq:03d}_user.wav"
            if audio_file and os.path.exists(audio_file):
                shutil.copy2(audio_file, user_audio_path)

            session.add_entry('user', text, user_audio_path)
            print(f"User: {text}", flush=True)

            # Check for wrap signal
            if session.is_wrap_signal(text):
                print("Wrap signal detected", flush=True)
                # Ask Claude for closing statement
                self._interview_generate_question(
                    "The guest wants to wrap up. Give a brief, warm closing statement summarizing key takeaways from the conversation."
                )
                self._end_interview()
                return

            # Continue interview - user's answer becomes the prompt
            self._interview_generate_question(text)

        except Exception as e:
            print(f"Interview answer processing error: {e}", flush=True)
            set_status('error')

    def _end_interview(self):
        """End the current interview session."""
        session = self.interview_session
        if not session:
            return

        session.active = False
        session.save_metadata()

        subprocess.Popen(
            ['notify-send', '-t', '5000', 'Interview Complete',
             f'Session saved to {session.session_dir}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Interview ended: {session.session_dir}", flush=True)

        # Post-process in background
        threading.Thread(
            target=self._post_process_interview,
            args=(session,),
            daemon=True
        ).start()

        self.interview_session = None
        set_status('idle')

    def _post_process_interview(self, session):
        """Post-process a completed interview: stitch audio, transcript, show notes."""
        print(f"Post-processing interview: {session.session_dir}", flush=True)

        try:
            # Collect audio files in sequence order
            audio_files = sorted(session.session_dir.glob('[0-9][0-9][0-9]_*.wav'))
            if not audio_files:
                print("No audio files to post-process", flush=True)
                return

            # Step 1: Normalize all audio to 44100 Hz stereo
            normalized_dir = session.session_dir / '.normalized'
            normalized_dir.mkdir(exist_ok=True)
            normalized_files = []

            for af in audio_files:
                norm_path = normalized_dir / af.name
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(af),
                    '-ar', '44100', '-ac', '2',
                    str(norm_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                normalized_files.append(norm_path)

            # Step 2: Stitch using ffmpeg concat demuxer
            concat_list = session.session_dir / '.concat_list.txt'
            with open(concat_list, 'w') as f:
                for nf in normalized_files:
                    f.write(f"file '{nf}'\n")

            stitched_wav = session.session_dir / f"interview_{session.session_id}.wav"
            subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                str(stitched_wav)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Step 3: Convert to MP3
            stitched_mp3 = session.session_dir / f"interview_{session.session_id}.mp3"
            subprocess.run([
                'ffmpeg', '-y', '-i', str(stitched_wav),
                '-codec:a', 'libmp3lame', '-qscale:a', '2',
                str(stitched_mp3)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Step 4: Write transcript
            transcript_path = session.session_dir / 'transcript.md'
            with open(transcript_path, 'w') as f:
                f.write(f"# Interview: {session.topic}\n")
                f.write(f"Date: {session.session_id}\n\n")
                f.write("---\n\n")
                for entry in session.transcript:
                    role = "Interviewer" if entry['role'] == 'interviewer' else "Guest"
                    f.write(f"**{role}** [{entry['time']}]:\n{entry['text']}\n\n")

            # Step 5: Generate show notes via Claude CLI
            try:
                transcript_text = transcript_path.read_text()
                show_notes_prompt = (
                    f"Based on this interview transcript, generate show notes with: "
                    f"1) A compelling title, 2) A brief summary (2-3 sentences), "
                    f"3) Key topics discussed (bulleted list), "
                    f"4) Notable quotes from the guest. "
                    f"Format as clean markdown.\n\n{transcript_text}"
                )
                result = subprocess.run(
                    [str(CLAUDE_CLI), '-p', show_notes_prompt],
                    capture_output=True, text=True, timeout=120
                )
                if result.stdout.strip():
                    show_notes_path = session.session_dir / 'show-notes.md'
                    show_notes_path.write_text(result.stdout.strip())
                    print(f"Show notes generated: {show_notes_path}", flush=True)
            except Exception as e:
                print(f"Show notes generation failed: {e}", flush=True)

            # Cleanup temp files
            shutil.rmtree(normalized_dir, ignore_errors=True)
            try:
                concat_list.unlink()
            except:
                pass

            # Step 6: Notify
            subprocess.Popen(
                ['notify-send', '-t', '10000', 'Post-Processing Complete',
                 f'Interview ready: {session.session_dir}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"Post-processing complete: {session.session_dir}", flush=True)

        except Exception as e:
            print(f"Post-processing error: {e}", flush=True)
            subprocess.Popen(
                ['notify-send', '-t', '5000', 'Post-Processing Error', str(e)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _interview_transcribe_and_process(self, temp_file):
        """Transcribe user audio and route to interview processing."""
        if not temp_file or not os.path.exists(temp_file.name):
            set_status('idle')
            return

        try:
            file_size = os.path.getsize(temp_file.name)
            if file_size < 5000:
                set_status('idle')
                print("Recording too short, skipping.", flush=True)
                return

            set_status('processing')

            # Convert to 16kHz mono for Whisper
            converted_file = temp_file.name.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file.name,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False
                )

            text = result['text'].strip()

            if text:
                print(f"Interview answer transcribed: {text}", flush=True)
                self._interview_process_answer(text, temp_file.name)
            else:
                print("No speech detected in interview answer.", flush=True)
                set_status('idle')

        except Exception as e:
            print(f"Interview transcription error: {e}", flush=True)
            set_status('error')
        finally:
            for f in [temp_file.name, temp_file.name.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass

    # --- Conversation Mode Methods ---

    def start_conversation(self):
        """Start a new conversation session with Claude + tools."""
        project_dir = self.config.get('conversation_project_dir', '')

        if not project_dir:
            # Ask for project dir via zenity
            try:
                result = subprocess.run(
                    ['zenity', '--file-selection',
                     '--directory',
                     '--title=Choose Project Directory for Conversation'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0 and result.stdout.strip():
                    project_dir = result.stdout.strip()
                    # Save for next time
                    self.config['conversation_project_dir'] = project_dir
                    save_config(self.config)
                else:
                    print("Conversation cancelled - no directory", flush=True)
                    set_status('idle')
                    return
            except Exception as e:
                print(f"Zenity error: {e}", flush=True)
                set_status('idle')
                return

        self.conversation_session = ConversationSession(project_dir)

        subprocess.Popen(
            ['notify-send', '-t', '3000', 'Conversation Started',
             f'Project: {self.conversation_session.project_dir}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Conversation started in {self.conversation_session.project_dir}", flush=True)

        # Speak a ready prompt, then auto-listen for first question
        threading.Thread(target=self._conversation_ready, daemon=True).start()

    def _conversation_ready(self):
        """Speak ready message and auto-listen for the first question."""
        self.tts_process = self.speak("Ready. What would you like to know?")
        if self.tts_process:
            self.tts_process.wait()
            self.tts_process = None

        if self.conversation_session and self.conversation_session.active:
            self._conversation_auto_listen()

    def _conversation_process_question(self, question):
        """Send a question to Claude with full tool access and speak the response."""
        session = self.conversation_session
        if not session or not session.active:
            return

        # Check for end signal
        if session.is_end_signal(question):
            print("End signal detected in conversation", flush=True)
            self._end_conversation()
            return

        try:
            set_status('processing')
            session.turn_count += 1
            print(f"Conversation Q{session.turn_count}: {question}", flush=True)

            result = subprocess.run(
                [
                    str(CLAUDE_CLI),
                    '-c', '-p', question,
                    '--permission-mode', 'bypassPermissions',
                    '--append-system-prompt', session.build_system_prompt(),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(session.project_dir)
            )
            response = result.stdout.strip()

            # Strip thinking blocks from response
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL).strip()

            if not response:
                print("No response from Claude in conversation", flush=True)
                set_status('error')
                return

            print(f"Conversation A{session.turn_count}: {response[:100]}...", flush=True)

            # Speak the response
            set_status('success')
            self.tts_process = self.speak(response)
            if self.tts_process:
                self.tts_process.wait()
                self.tts_process = None

            # Auto-listen for follow-up if session still active
            if session.active:
                self._conversation_auto_listen()

        except subprocess.TimeoutExpired:
            print("Claude conversation request timed out", flush=True)
            set_status('error')
        except Exception as e:
            print(f"Conversation error: {e}", flush=True)
            set_status('error')

    def _conversation_auto_listen(self):
        """Auto-listen for a follow-up in conversation mode."""
        session = self.conversation_session
        if not session or not session.active:
            return

        print(f"Conversation auto-listening for {AUTO_LISTEN_SECONDS} seconds...", flush=True)
        set_status('recording')

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()

        record_process = subprocess.Popen([
            'pw-record',
            '--format', 's16',
            '--rate', '44100',
            '--channels', '2',
            temp_file.name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(AUTO_LISTEN_SECONDS)

        record_process.terminate()
        record_process.wait()

        # Check audio level
        file_size = os.path.getsize(temp_file.name)
        audio_level = get_audio_level(temp_file.name)
        print(f"Conversation auto-listen: size={file_size}, level={audio_level:.1f} dB", flush=True)

        if file_size < 10000 or audio_level < SILENCE_THRESHOLD:
            print("No follow-up detected (silence), going idle.", flush=True)
            set_status('idle')
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return

        set_status('processing')

        # Transcribe
        converted_file = temp_file.name.replace('.wav', '_16k.wav')
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file.name,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False
                )

            followup = result['text'].strip()

            for f in [temp_file.name, converted_file]:
                try:
                    os.unlink(f)
                except:
                    pass

            if followup and len(followup) > 3:
                print(f"Conversation follow-up: {followup}", flush=True)
                self._conversation_process_question(followup)
            else:
                print("No meaningful follow-up, going idle.", flush=True)
                set_status('idle')

        except Exception as e:
            print(f"Conversation auto-listen error: {e}", flush=True)
            set_status('idle')
            for f in [temp_file.name, converted_file]:
                try:
                    os.unlink(f)
                except:
                    pass

    def _conversation_transcribe_and_process(self, temp_file):
        """Transcribe PTT recording and route to conversation processing."""
        if not temp_file or not os.path.exists(temp_file.name):
            set_status('idle')
            return

        try:
            file_size = os.path.getsize(temp_file.name)
            if file_size < 5000:
                set_status('idle')
                print("Recording too short, skipping.", flush=True)
                return

            set_status('processing')

            converted_file = temp_file.name.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file.name,
                '-ar', '16000', '-ac', '1',
                converted_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with self.model_lock:
                result = self.model.transcribe(
                    converted_file,
                    language='en',
                    fp16=False
                )

            text = result['text'].strip()

            if text:
                print(f"Conversation transcribed: {text}", flush=True)
                self._conversation_process_question(text)
            else:
                print("No speech detected in conversation.", flush=True)
                set_status('idle')

        except Exception as e:
            print(f"Conversation transcription error: {e}", flush=True)
            set_status('error')
        finally:
            for f in [temp_file.name, temp_file.name.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass

    def _end_conversation(self):
        """End the current conversation session."""
        session = self.conversation_session
        if not session:
            return

        session.active = False

        # Speak goodbye
        self.tts_process = self.speak("Ending conversation. Talk to you later!")
        if self.tts_process:
            self.tts_process.wait()
            self.tts_process = None

        subprocess.Popen(
            ['notify-send', '-t', '3000', 'Conversation Ended',
             f'{session.turn_count} exchanges'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Conversation ended after {session.turn_count} turns", flush=True)

        self.conversation_session = None
        set_status('idle')

    def on_press(self, key):
        try:
            self._on_press_inner(key)
        except Exception as e:
            print(f"ERROR in on_press: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _on_press_inner(self, key):
        # Interrupt key stops realtime AI or live session
        if key == self.interrupt_key and self.realtime_session:
            print("Interrupt key pressed - interrupting AI", flush=True)
            self.realtime_session.request_interrupt()
            return
        if key == self.interrupt_key and self.live_session:
            print("Interrupt key pressed - interrupting live session", flush=True)
            self.live_session.request_interrupt()
            return

        # Track modifier keys
        if key == self.ptt_key:
            self.ctrl_r_pressed = True
        elif key == self.ai_key:
            self.shift_r_pressed = True
            # Stop any ongoing TTS when starting new recording
            self.stop_tts()

        # Live mode: Right Shift = tap to cycle, hold for push-to-talk
        # No timers. Press records state; release decides tap vs hold.
        if key == self.ai_key and not self.ctrl_r_pressed:
            ai_mode = self.config.get('ai_mode', 'claude')
            if ai_mode == 'live':
                # Guard against key repeat events (Linux sends repeats while held)
                # Time-based fallback: if held flag stuck >2s, force reset (missed release)
                if getattr(self, '_live_key_held', False):
                    if time.time() - getattr(self, '_live_key_press_time', 0) > 2.0:
                        print("Live: force-resetting stuck key held flag", flush=True)
                        self._live_key_held = False
                    else:
                        return
                self._live_key_held = True
                self._live_press_processed = True  # Release handler checks this
                self._live_key_press_time = time.time()
                self._live_starting_session = False
                if not self.live_session or not self.live_session.running:
                    # Idle or dead session (idle timeout) → start new session
                    if self.live_session and not self.live_session.running:
                        print("Live: dead session detected, starting fresh", flush=True)
                        self.live_session = None
                    self._live_starting_session = True
                    self.start_live_session()
                elif self.live_session.playing_audio:
                    # Interrupt playback immediately
                    print("Live: interrupting playback", flush=True)
                    self.live_session.request_interrupt()
                    # Record state so release handler knows context
                    self._live_press_state = 'interrupted'
                    self.live_session.set_muted(False)
                else:
                    # Record state at press time for release handler
                    self._live_press_state = 'muted' if self.live_session.muted else 'listening'
                    print(f"Live press: state={self._live_press_state}, muted={self.live_session.muted}, running={self.live_session.running}", flush=True)
                    # Optimistic unmute: if muted, unmute now for hold-to-talk.
                    # If this turns out to be a tap, release handler reverts it.
                    if self.live_session.muted:
                        self.live_session.set_muted(False)
                        print(f"Live press: unmuted (optimistic), muted now={self.live_session.muted}", flush=True)
                return
        elif key == self.ai_key and self.ctrl_r_pressed:
            print(f"Live: ai_key blocked by ctrl_r_pressed", flush=True)

        # Check for AI assistant mode (both modifiers held)
        if self.ctrl_r_pressed and self.shift_r_pressed and not self.recording:
            # If realtime session is running, stop it (toggle behavior)
            if self.realtime_session:
                print("Stopping Realtime session (toggle)", flush=True)
                self.stop_realtime_session()
                return

            # Clear any stale keys for AI mode
            self.other_keys_pressed.clear()
            self.ai_mode = True

            ai_mode = self.config.get('ai_mode', 'claude')
            print(f"AI assistant mode activated (mode: {ai_mode})", flush=True)

            if ai_mode == 'interview':
                if not self.interview_session:
                    # First trigger: start the interview
                    self.start_interview()
                elif self.interview_session.status == 'idle':
                    # Subsequent triggers: record an answer
                    self.start_recording(force=True)
                else:
                    print(f"Interview busy (status: {self.interview_session.status})", flush=True)
            elif ai_mode == 'conversation':
                if not self.conversation_session:
                    # First trigger: start the conversation
                    self.start_conversation()
                else:
                    # Subsequent triggers: record a follow-up
                    self.start_recording(force=True)
            elif ai_mode == 'live':
                if self.live_session:
                    # Session running — Ctrl+Shift stops it (toggle)
                    print("Stopping live session (Ctrl+Shift toggle)", flush=True)
                    self.stop_live_session()
                else:
                    # Session not running — start it
                    self.start_live_session()
            elif ai_mode == 'realtime':
                # Check why Realtime might not work and report errors
                if not REALTIME_AVAILABLE:
                    msg = "Realtime mode failed: websockets not installed"
                    print(f"ERROR: {msg}", flush=True)
                    self.show_error(msg)
                    return
                if not get_openai_api_key():
                    print("No OpenAI API key found, prompting user...", flush=True)
                    prompt_api_key()
                    return
                # All checks passed, start Realtime session (stays active until toggled off)
                self.start_realtime_session()
            else:
                # Use Claude + Whisper + TTS
                self.start_recording(force=True)
            return

        # Regular dictation mode (PTT key only, no AI key)
        if key == self.ptt_key:
            # Reset stale shift_r flag if user is clearly just using PTT
            # (shift_r not physically pressed but flag stuck from previous AI mode)
            if self.shift_r_pressed:
                print(f"Clearing stale shift_r_pressed flag", flush=True)
                self.shift_r_pressed = False
            if self.other_keys_pressed:
                print(f"Clearing stale keys: {self.other_keys_pressed}", flush=True)
                self.other_keys_pressed.clear()
            # Mute live session during PTT dictation so it doesn't pick up the audio
            if self.live_session and not self.live_session.muted:
                self._ptt_muted_live = True
                self.live_session.set_muted(True)
                print("PTT key pressed, muted live session, starting recording", flush=True)
            else:
                self._ptt_muted_live = False
                print("PTT key pressed, starting recording", flush=True)
            self.ai_mode = False
            self.start_recording()
        elif key not in (self.ptt_key, self.ai_key):
            self.other_keys_pressed.add(key)

    def on_release(self, key):
        try:
            self._on_release_inner(key)
        except Exception as e:
            print(f"ERROR in on_release: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _on_release_inner(self, key):
        # Track modifier releases
        if key == self.ptt_key:
            self.ctrl_r_pressed = False
        elif key == self.ai_key:
            self.shift_r_pressed = False

        # Live mode: tap = cycle, hold = mute+flush on release
        if key == self.ai_key:
            ai_mode = self.config.get('ai_mode', 'claude')
            if ai_mode == 'live':
                self._live_key_held = False  # Clear repeat guard

                # Only process release if the press handler actually ran.
                # Without this, a blocked press (e.g. ctrl_r_pressed was True)
                # causes the release handler to read stale state and mute incorrectly.
                if not getattr(self, '_live_press_processed', False):
                    return
                self._live_press_processed = False

                press_time = getattr(self, '_live_key_press_time', 0)
                elapsed = time.time() - press_time
                elapsed_ms = int(elapsed * 1000)
                press_state = getattr(self, '_live_press_state', None)

                if getattr(self, '_live_starting_session', False):
                    # First tap started the session — don't do anything
                    self._live_starting_session = False
                    print("Live: session started (ignoring release)", flush=True)

                elif elapsed < 0.5 and self.live_session:
                    auto_mute = self.config.get('live_auto_mute', True)
                    # TAP (<500ms): cycle based on state at press time
                    if press_state == 'listening' and auto_mute:
                        self.live_session.set_muted(True)
                        print(f"Live tap ({elapsed_ms}ms): listening → muted", flush=True)
                    elif press_state == 'muted':
                        # Ensure unmute (press handler does optimistic unmute, but reinforce here)
                        self.live_session.set_muted(False)
                        print(f"Live tap ({elapsed_ms}ms): muted → listening", flush=True)
                    elif press_state == 'interrupted':
                        self.live_session.set_muted(False)
                        print(f"Live tap ({elapsed_ms}ms): interrupted → listening", flush=True)

                elif self.live_session:
                    auto_mute = self.config.get('live_auto_mute', True)
                    if elapsed >= 2.0 and press_state == 'muted':
                        # LONG HOLD (>=2s) from muted: stop session
                        print(f"Live long hold ({elapsed_ms}ms): stopping session", flush=True)
                        self.stop_live_session()
                    elif press_state == 'muted':
                        # HOLD from muted: keep listening (ensure unmuted)
                        self.live_session.set_muted(False)
                        print(f"Live hold ({elapsed_ms}ms): muted → listening (held)", flush=True)
                    elif auto_mute:
                        # HOLD (>=500ms) from listening: mute + flush transcript
                        self.live_session.set_muted(True)
                        print(f"Live hold ({elapsed_ms}ms): muted (released)", flush=True)

                return

        # Stop recording when keys are released (but NOT realtime session - that's toggle-based)
        if key in (self.ptt_key, self.ai_key):
            if self.recording or self.stream_active:
                # For AI mode: ignore release if either key still held,
                # or if recording just started (spurious release events)
                if self.ai_mode:
                    elapsed = time.time() - getattr(self, 'record_start_time', 0)
                    if self.ctrl_r_pressed or self.shift_r_pressed:
                        return
                    if elapsed < 0.5:
                        print(f"Ignoring early release ({elapsed:.2f}s), re-arming keys", flush=True)
                        self.ctrl_r_pressed = True
                        self.shift_r_pressed = True
                        return
                print(f"Key released, stopping recording (AI mode: {self.ai_mode})", flush=True)
                self.stop_recording_with_mode()
        elif key not in (self.ptt_key, self.ai_key):
            self.other_keys_pressed.discard(key)

    def stop_recording_with_mode(self):
        """Stop recording and process based on mode."""
        # Handle stream mode
        if self.stream_active:
            self.stop_stream_mode()
            return

        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        # Restore live session mute state if we muted it for PTT dictation
        if getattr(self, '_ptt_muted_live', False):
            self._ptt_muted_live = False
            if self.live_session:
                self.live_session.set_muted(False)
                print("PTT done, unmuted live session", flush=True)

        # Dismiss indicator immediately - transcription happens in background
        set_status('idle')
        print("Processing...", flush=True)

        # Capture temp_file locally to avoid race conditions with new recordings
        temp_file = self.temp_file
        self.temp_file = None

        if self.interview_session and self.interview_session.active:
            self.interview_session.status = 'recording'
            threading.Thread(target=self._interview_transcribe_and_process, args=(temp_file,), daemon=True).start()
        elif self.conversation_session and self.conversation_session.active:
            threading.Thread(target=self._conversation_transcribe_and_process, args=(temp_file,), daemon=True).start()
        elif self.ai_mode:
            threading.Thread(target=self.transcribe_and_ask_ai, args=(temp_file,), daemon=True).start()
        else:
            threading.Thread(target=self.transcribe_and_type, args=(temp_file,), daemon=True).start()

    def run(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


def main():
    def signal_handler(sig, frame):
        print("\nShutting down...", flush=True)
        stop_indicator()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(stop_indicator)

    # Start the indicator
    start_indicator()

    ptt = PushToTalk()
    ptt.run()


if __name__ == '__main__':
    main()
