#!/usr/bin/env python3
"""
Push-to-Talk Dictation Service

Hold Right Ctrl to record, release to transcribe and type into focused input.
Learns from corrections and maintains a custom vocabulary.
"""

import os
import sys
import re
import tempfile
import subprocess
import threading
import signal
import atexit
from collections import deque
from pathlib import Path
import whisper
from pynput import keyboard

# Configuration
PTT_KEY = keyboard.Key.ctrl_r  # Right Control
WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large
BASE_DIR = Path(__file__).parent
VOCAB_FILE = BASE_DIR / "vocabulary.txt"
STATUS_FILE = BASE_DIR / "status"
INDICATOR_SCRIPT = BASE_DIR / "indicator.py"
HISTORY_SIZE = 10  # Number of recent transcriptions to remember

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

        # Vocabulary and history
        self.vocab = VocabularyManager(VOCAB_FILE)
        self.history = deque(maxlen=HISTORY_SIZE)

        print("Loading Whisper model...", flush=True)
        self.model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model '{WHISPER_MODEL}' loaded.", flush=True)
        print(f"Push-to-Talk ready. Hold {PTT_KEY} to dictate.", flush=True)

        # Set idle status
        set_status('idle')

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

    def start_recording(self):
        if self.recording:
            return

        if self.other_keys_pressed:
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

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        set_status('processing')
        print("Processing...", flush=True)

        threading.Thread(target=self.transcribe_and_type, daemon=True).start()

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

    def on_press(self, key):
        if key == PTT_KEY:
            print(f"PTT key pressed (other_keys: {len(self.other_keys_pressed)})", flush=True)
            self.start_recording()
        else:
            self.other_keys_pressed.add(key)

    def on_release(self, key):
        if key == PTT_KEY:
            print("PTT key released", flush=True)
            self.stop_recording()
        else:
            self.other_keys_pressed.discard(key)

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
