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
import shutil
import tempfile
import subprocess
import threading
import signal
import atexit
from collections import deque
from pathlib import Path
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
        "ai_mode": "claude",  # "claude" or "realtime"
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


def speak_openai(text, voice="nova"):
    """Speak text using OpenAI TTS API."""
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

        # Stream the audio to aplay
        for chunk in response.iter_bytes(chunk_size=4096):
            process.stdin.write(chunk)
        process.stdin.close()
        return process

    except Exception as e:
        print(f"OpenAI TTS error: {e}", flush=True)
        return None

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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
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
        self.tts_process = None  # Track TTS process for interruption

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

        # Set idle status
        set_status('idle')

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

    def speak(self, text):
        """Speak text using configured TTS backend."""
        tts_backend = self.config.get('tts_backend', 'piper')

        if tts_backend == 'openai' and OPENAI_AVAILABLE and get_openai_api_key():
            voice = self.config.get('openai_voice', 'nova')
            return speak_openai(text, voice)
        else:
            # Use Piper
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

    def start_recording(self, force=False):
        if self.recording:
            print("Already recording, skipping", flush=True)
            return

        if self.other_keys_pressed and not force:
            print(f"Blocked by other keys: {self.other_keys_pressed}", flush=True)
            return

        self.recording = True
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


    def transcribe_and_type(self):
        if not self.temp_file or not os.path.exists(self.temp_file.name):
            set_status('idle')
            return

        try:
            file_size = os.path.getsize(self.temp_file.name)
            print(f"Recording file size: {file_size} bytes", flush=True)
            if file_size < 5000:
                set_status('idle')
                print("Recording too short, skipping.", flush=True)
                return

            # Convert to 16kHz mono for Whisper
            converted_file = self.temp_file.name.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', self.temp_file.name,
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

                # Check for correction commands
                correction = self.detect_correction(text)
                if correction:
                    self.vocab.add(correction)
                    # Don't type correction commands
                    if any(text.lower().startswith(cmd) for cmd in
                           ['add word', 'learn word', 'remember ', 'correction']):
                        print("Correction command detected, not typing.", flush=True)
                        set_status('success')
                        return

                # Add to history
                self.history.append(text)

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
            for f in [self.temp_file.name, self.temp_file.name.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass
            self.temp_file = None

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

        # Wait for auto-listen duration (can be interrupted by key press)
        start_time = time.time()
        while time.time() - start_time < AUTO_LISTEN_SECONDS:
            if self.ctrl_r_pressed or self.shift_r_pressed:
                # User pressed a key, they want to keep talking
                print("Key press detected, extending listen...", flush=True)
                # Let the normal key handling take over
                record_process.terminate()
                record_process.wait()
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                return
            time.sleep(0.1)

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

    def transcribe_and_ask_ai(self):
        """Transcribe audio and send to Claude, then speak the response."""
        if not self.temp_file or not os.path.exists(self.temp_file.name):
            set_status('idle')
            return

        try:
            file_size = os.path.getsize(self.temp_file.name)
            print(f"AI mode - Recording file size: {file_size} bytes", flush=True)
            if file_size < 5000:
                set_status('idle')
                print("Recording too short, skipping.", flush=True)
                return

            # Convert to 16kHz mono for Whisper
            converted_file = self.temp_file.name.replace('.wav', '_16k.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', self.temp_file.name,
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
                        print(f"AI Response: {response[:100]}...", flush=True)
                        set_status('success')

                        # Speak the response and wait for completion
                        self.tts_process = self.speak(response)
                        if self.tts_process:
                            self.tts_process.wait()
                            self.tts_process = None

                        # Auto-listen for follow-up
                        self.auto_listen_for_followup()
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
            for f in [self.temp_file.name, self.temp_file.name.replace('.wav', '_16k.wav')]:
                try:
                    os.unlink(f)
                except:
                    pass
            self.temp_file = None

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
                print(f"AI Response: {response[:100]}...", flush=True)
                set_status('success')

                # Speak the response and wait for completion
                self.tts_process = self.speak(response)
                if self.tts_process:
                    self.tts_process.wait()
                    self.tts_process = None

                # Auto-listen for follow-up
                self.auto_listen_for_followup()
            else:
                print("No response from Claude", flush=True)
                set_status('error')

        except subprocess.TimeoutExpired:
            print("Claude request timed out", flush=True)
            set_status('error')
        except Exception as e:
            print(f"Error calling Claude: {e}", flush=True)
            set_status('error')

    def on_press(self, key):
        # Interrupt key stops realtime AI
        if key == self.interrupt_key and self.realtime_session:
            print("Interrupt key pressed - interrupting AI", flush=True)
            self.realtime_session.request_interrupt()
            return

        # Track modifier keys
        if key == self.ptt_key:
            self.ctrl_r_pressed = True
        elif key == self.ai_key:
            self.shift_r_pressed = True
            # Stop any ongoing TTS when starting new recording
            self.stop_tts()

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

            if ai_mode == 'realtime':
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
        if key == self.ptt_key and not self.shift_r_pressed:
            if self.other_keys_pressed:
                print(f"PTT key pressed (blocked by: {self.other_keys_pressed})", flush=True)
                self.other_keys_pressed.clear()
                print("Cleared stale keys, try again", flush=True)
            else:
                print("PTT key pressed, starting recording", flush=True)
                self.ai_mode = False
                self.start_recording()
        elif key not in (self.ptt_key, self.ai_key):
            self.other_keys_pressed.add(key)

    def on_release(self, key):
        # Track modifier releases
        if key == self.ptt_key:
            self.ctrl_r_pressed = False
        elif key == self.ai_key:
            self.shift_r_pressed = False

        # Stop recording when keys are released (but NOT realtime session - that's toggle-based)
        if key in (self.ptt_key, self.ai_key):
            if self.recording:
                print(f"Key released, stopping recording (AI mode: {self.ai_mode})", flush=True)
                self.stop_recording_with_mode()
        elif key not in (self.ptt_key, self.ai_key):
            self.other_keys_pressed.discard(key)

    def stop_recording_with_mode(self):
        """Stop recording and process based on mode."""
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        set_status('processing')
        print("Processing...", flush=True)

        if self.ai_mode:
            threading.Thread(target=self.transcribe_and_ask_ai, daemon=True).start()
        else:
            threading.Thread(target=self.transcribe_and_type, daemon=True).start()

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
