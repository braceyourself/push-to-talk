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
FINALIZE_WAIT = 1.5  # seconds to wait for final transcript after key release


class StreamingDictation:
    """Streams mic audio to Deepgram and types transcripts in real time."""

    def __init__(self, api_key, on_final=None, on_status=None):
        self._api_key = api_key
        self._on_final = on_final or (lambda t: None)
        self._on_status = on_status or (lambda s: None)
        self._running = False
        self._stop_event = threading.Event()
        self._finalized = threading.Event()
        self._dg_connection = None
        self._dg_context = None
        self._record_proc = None
        self._listener_thread = None
        self._has_typed = False

    @property
    def running(self):
        return self._running

    def start(self):
        """Start streaming dictation in a background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._finalized.clear()
        self._has_typed = False
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        """Signal stop — the worker thread handles graceful shutdown."""
        self._stop_event.set()

    def _run(self):
        """Connect to Deepgram, capture audio, stream it, type results."""
        dg_connection = None
        dg_context = None
        try:
            self._on_status('recording')

            client = DeepgramClient(api_key=self._api_key)
            dg_context = client.listen.v1.connect(
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
            dg_connection = dg_context.__enter__()
            self._dg_connection = dg_connection
            self._dg_context = dg_context

            dg_connection.on(EventType.OPEN, self._on_open)
            dg_connection.on(EventType.MESSAGE, self._on_message)
            dg_connection.on(EventType.ERROR, self._on_error)
            dg_connection.on(EventType.CLOSE, self._on_close)

            # start_listening() is BLOCKING — run in daemon thread
            self._listener_thread = threading.Thread(
                target=dg_connection.start_listening, daemon=True
            )
            self._listener_thread.start()

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

            # Forward audio chunks to Deepgram until stop signal
            while not self._stop_event.is_set():
                data = self._record_proc.stdout.read(CHUNK_SIZE)
                if not data:
                    break
                try:
                    dg_connection.send_media(data)
                except Exception:
                    break

            # ── Key released: flush remaining audio ──────────────────
            # Stop recording
            if self._record_proc:
                self._record_proc.terminate()
                self._record_proc.wait(timeout=2)
                self._record_proc = None

            # Tell Deepgram to flush its buffer and send final transcript
            self._on_status('processing')
            try:
                dg_connection.send_control(
                    ListenV1ControlMessage(type="Finalize")
                )
            except Exception:
                pass

            # Wait for the final transcript to arrive
            self._finalized.wait(timeout=FINALIZE_WAIT)

        except Exception as e:
            print(f"StreamingDictation: error: {e}", flush=True)
            self._on_status('error')
        finally:
            # Clean up connection
            if dg_context:
                try:
                    dg_context.__exit__(None, None, None)
                except Exception:
                    pass
            if self._listener_thread:
                self._listener_thread.join(timeout=2)
            self._dg_connection = None
            self._dg_context = None
            self._running = False
            self._on_status('idle')
            print("StreamingDictation: stopped", flush=True)

    # ── Deepgram callbacks ────────────────────────────────────────

    def _on_open(self, *args):
        print("StreamingDictation: connected to Deepgram", flush=True)

    def _on_close(self, *args):
        self._finalized.set()  # Unblock wait on close

    def _on_error(self, *args):
        print(f"StreamingDictation: websocket error", flush=True)
        self._finalized.set()

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
                # Empty final after Finalize = Deepgram is done
                if result.is_final and self._stop_event.is_set():
                    self._finalized.set()
                return

            if not result.is_final:
                return

            # Prepend space if not first phrase
            text = (" " + transcript) if self._has_typed else transcript
            self._has_typed = True

            self._type_text(text)
            self._on_final(transcript)
            print(f"StreamingDictation: '{transcript}'", flush=True)

            # If we're in shutdown, this final transcript might be the last
            if self._stop_event.is_set():
                self._finalized.set()

        except Exception as e:
            print(f"StreamingDictation: message error: {e}", flush=True)

    # ── Output ────────────────────────────────────────────────────

    @staticmethod
    def _type_text(text):
        """Type text into the focused window using xdotool.

        Transcripts arrive after key release so modifiers are clear.
        xdotool type is simpler and doesn't interfere with clipboard.
        """
        subprocess.run(
            ['xdotool', 'type', '--clearmodifiers', '--delay', '8', text],
            timeout=10, capture_output=True,
        )
