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
PTT_KEY = keyboard.Key.ctrl_r  # Right Control for dictation
AI_KEY = keyboard.Key.shift_r  # Right Shift (+ Right Ctrl) for AI assistant
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

        # AI assistant mode tracking
        self.ai_mode = False  # True when recording for AI assistant
        self.ctrl_r_pressed = False
        self.shift_r_pressed = False
        self.tts_process = None  # Track TTS process for interruption

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

        # Check if we got any audio
        file_size = os.path.getsize(temp_file.name)
        print(f"Auto-listen recording size: {file_size} bytes", flush=True)

        if file_size < 10000:  # Threshold for meaningful audio
            print("No follow-up detected, going idle.", flush=True)
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

                        # Speak the response using Piper and wait for completion
                        self.tts_process = subprocess.Popen(
                            f'echo {subprocess.list2cmdline([response])} | "{PIPER_CMD}" --model "{PIPER_MODEL}" --output-raw 2>/dev/null | aplay -r 22050 -f S16_LE -t raw -q',
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        self.tts_process.wait()  # Wait for TTS to finish
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

                # Speak the response using Piper and wait for completion
                self.tts_process = subprocess.Popen(
                    f'echo {subprocess.list2cmdline([response])} | "{PIPER_CMD}" --model "{PIPER_MODEL}" --output-raw 2>/dev/null | aplay -r 22050 -f S16_LE -t raw -q',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
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
        # Track modifier keys
        if key == PTT_KEY:
            self.ctrl_r_pressed = True
        elif key == AI_KEY:
            self.shift_r_pressed = True
            # Stop any ongoing TTS when starting new recording
            self.stop_tts()

        # Check for AI assistant mode (Right Shift + Right Ctrl)
        if self.ctrl_r_pressed and self.shift_r_pressed and not self.recording:
            # Clear any stale keys for AI mode
            self.other_keys_pressed.clear()
            print("AI assistant mode activated", flush=True)
            self.ai_mode = True
            self.start_recording(force=True)
            return

        # Regular dictation mode (Right Ctrl only, no Right Shift)
        if key == PTT_KEY and not self.shift_r_pressed:
            if self.other_keys_pressed:
                print(f"PTT key pressed (blocked by: {self.other_keys_pressed})", flush=True)
                self.other_keys_pressed.clear()
                print("Cleared stale keys, try again", flush=True)
            else:
                print("PTT key pressed, starting recording", flush=True)
                self.ai_mode = False
                self.start_recording()
        elif key not in (PTT_KEY, AI_KEY):
            self.other_keys_pressed.add(key)

    def on_release(self, key):
        # Track modifier releases
        if key == PTT_KEY:
            self.ctrl_r_pressed = False
        elif key == AI_KEY:
            self.shift_r_pressed = False

        # Stop recording when either key of the combo is released
        if key in (PTT_KEY, AI_KEY) and self.recording:
            print(f"Key released, stopping recording (AI mode: {self.ai_mode})", flush=True)
            self.stop_recording_with_mode()
        elif key not in (PTT_KEY, AI_KEY):
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
