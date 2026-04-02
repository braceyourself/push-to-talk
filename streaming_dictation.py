"""Real-time streaming dictation using Deepgram WebSocket STT.

Audio flows: mic → Deepgram Nova-3 → xdotool type
Text appears in the focused window as you speak, while Right Ctrl is held.

Only final transcripts are typed (not interim) to avoid backspace churn.
Deepgram's endpointing (~300ms) means words appear with minimal delay.
"""

import subprocess
import threading
import time

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets.listen_v1_control_message import (
    ListenV1ControlMessage,
)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096  # ~128ms at 16kHz 16-bit mono


class StreamingDictation:
    """Streams mic audio to Deepgram and types transcripts in real time."""

    def __init__(self, api_key, on_final=None, on_status=None):
        self._api_key = api_key
        self._on_final = on_final or (lambda t: None)
        self._on_status = on_status or (lambda s: None)
        self._running = False
        self._stop_event = threading.Event()
        self._dg_connection = None
        self._dg_context = None
        self._record_proc = None
        self._has_typed = False  # Whether we've typed anything yet

    @property
    def running(self):
        return self._running

    def start(self):
        """Start streaming dictation in a background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._has_typed = False
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        """Stop streaming dictation gracefully."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        if self._record_proc:
            try:
                self._record_proc.terminate()
                self._record_proc.wait(timeout=2)
            except Exception:
                pass
            self._record_proc = None

        if self._dg_connection:
            try:
                self._dg_connection.send_control(
                    ListenV1ControlMessage(type="Finalize")
                )
            except Exception:
                pass

        if self._dg_context:
            try:
                self._dg_context.__exit__(None, None, None)
            except Exception:
                pass
            self._dg_context = None
        self._dg_connection = None

    def _run(self):
        """Connect to Deepgram, capture audio, stream it, type results."""
        try:
            self._on_status('recording')

            client = DeepgramClient(api_key=self._api_key)
            self._dg_context = client.listen.v1.connect(
                model="nova-3",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
                channels="1",
                interim_results="true",
                endpointing="300",
                utterance_end_ms="1000",
                smart_format="true",
                punctuate="true",
                language="en",
            )
            dg_connection = self._dg_context.__enter__()

            dg_connection.on(EventType.OPEN, self._on_open)
            dg_connection.on(EventType.MESSAGE, self._on_message)
            dg_connection.on(EventType.ERROR, self._on_error)
            dg_connection.on(EventType.CLOSE, self._on_close)

            self._dg_connection = dg_connection

            # start_listening() is BLOCKING — run in daemon thread
            listener_thread = threading.Thread(
                target=dg_connection.start_listening, daemon=True
            )
            listener_thread.start()

            # Capture audio from default mic via PipeWire
            self._record_proc = subprocess.Popen(
                [
                    'pw-record', '--format', 's16',
                    '--rate', str(SAMPLE_RATE),
                    '--channels', str(CHANNELS),
                    '-',
                ],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )

            # Forward audio chunks to Deepgram
            while not self._stop_event.is_set():
                data = self._record_proc.stdout.read(CHUNK_SIZE)
                if not data:
                    break
                try:
                    dg_connection.send_media(data)
                except Exception:
                    break

        except Exception as e:
            print(f"StreamingDictation: error: {e}", flush=True)
            self._on_status('error')
        finally:
            self.stop()
            self._on_status('idle')
            print("StreamingDictation: stopped", flush=True)

    # ── Deepgram callbacks ────────────────────────────────────────

    def _on_open(self, *args):
        print("StreamingDictation: connected to Deepgram", flush=True)

    def _on_close(self, *args):
        pass

    def _on_error(self, *args):
        print(f"StreamingDictation: websocket error", flush=True)

    def _on_message(self, *args):
        """Handle transcript results — type finals into focused window."""
        result = args[-1] if args else None
        if not result or getattr(result, 'type', None) != 'Results':
            return

        try:
            alternatives = result.channel.alternatives
            if not alternatives:
                return

            transcript = alternatives[0].transcript.strip()
            if not transcript:
                return

            if not result.is_final:
                return

            # Prepend space if not first phrase
            text = (" " + transcript) if self._has_typed else transcript
            self._has_typed = True

            self._type_text(text)
            self._on_final(transcript)
            print(f"StreamingDictation: '{transcript}'", flush=True)

        except Exception as e:
            print(f"StreamingDictation: message error: {e}", flush=True)

    # ── Output ────────────────────────────────────────────────────

    @staticmethod
    def _type_text(text):
        """Type text into the focused window using xdotool."""
        subprocess.run(
            ['xdotool', 'type', '--clearmodifiers', '--delay', '8', text],
            timeout=10, capture_output=True,
        )
