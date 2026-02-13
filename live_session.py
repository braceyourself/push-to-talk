#!/usr/bin/env python3
"""
Live voice conversation session using OpenAI Realtime API.
Connects via WebSocket for real-time voice-to-voice conversation
with personality loading, idle timeout, and context summarization.
"""

import os
import json
import base64
import asyncio
import subprocess
import time
from pathlib import Path

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Audio settings for OpenAI Realtime API
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4096

# OpenAI Realtime API endpoint
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

# Context management thresholds
SUMMARY_TRIGGER = 20000  # tokens before triggering summarization
KEEP_LAST_TURNS = 3      # turns to preserve when summarizing


class ConversationState:
    """Tracks conversation turns for context management."""

    def __init__(self):
        self.history = []        # list of {role, item_id, text}
        self.summary_count = 0
        self.latest_tokens = 0
        self.summarizing = False


class LiveSession:
    """OpenAI Realtime voice session with personality and memory."""

    def __init__(self, api_key, voice="ash", on_status=None):
        self.api_key = api_key
        self.voice = voice
        self.on_status = on_status or (lambda s: None)
        self.ws = None
        self.running = False
        self.audio_player = None
        self.playing_audio = False
        self.audio_done_time = 0
        self._interrupt_requested = False
        self.conversation = ConversationState()
        self.personality_prompt = self._build_personality()
        self._idle_timer = None
        self._idle_timeout = 120  # 2 minutes

    def _build_personality(self):
        """Load personality from multiple .md files in personality/ directory."""
        parts = []
        personality_dir = Path(__file__).parent / "personality"
        if personality_dir.exists():
            for md_file in sorted(personality_dir.glob("*.md")):
                content = md_file.read_text().strip()
                if content:
                    parts.append(content)
        return "\n\n".join(parts)

    def _set_status(self, status):
        """Update status via callback."""
        self.on_status(status)

    async def connect(self):
        """Connect to OpenAI Realtime API and configure session."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        self.ws = await websockets.connect(
            REALTIME_URL,
            additional_headers=headers,
            ping_interval=20,
            max_size=None
        )

        # Configure session with personality and voice settings
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
                "tools": [],
                "tool_choice": "none"
            }
        }))

        print("Live session: Connected", flush=True)

        # Seed context from previous conversation if available
        await self.seed_context()

        return True

    async def disconnect(self):
        """Disconnect from the API and clean up resources."""
        self.running = False
        self._cancel_idle_timer()

        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        if self.audio_player:
            try:
                self.audio_player.terminate()
                self.audio_player.wait(timeout=2)
            except Exception:
                pass
            self.audio_player = None

        # Always unmute mic on disconnect
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                       capture_output=True)
        print("Live session: Disconnected", flush=True)

    def start_audio_player(self):
        """Start the audio player subprocess for PCM16 playback."""
        self.audio_player = subprocess.Popen(
            ['aplay', '-r', '24000', '-f', 'S16_LE', '-t', 'raw', '-q'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    async def send_audio(self, audio_data):
        """Send audio data to the API as base64-encoded PCM16."""
        if self.ws:
            encoded = base64.b64encode(audio_data).decode('utf-8')
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": encoded
            }))

    async def handle_events(self):
        """Process incoming events from the Realtime API."""
        try:
            async for message in self.ws:
                if not self.running:
                    break

                # Check for interrupt request
                if self._interrupt_requested:
                    self._interrupt_requested = False
                    await self._interrupt()

                data = json.loads(message)
                event_type = data.get("type", "")

                # Log non-audio events for debugging
                if event_type not in ("response.audio.delta",):
                    print(f"Live session event: {event_type}", flush=True)

                if event_type == "response.audio.delta":
                    # Play audio chunk and mute mic to prevent echo
                    if not self.playing_audio:
                        self.playing_audio = True
                        self._set_status("speaking")
                        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '1'],
                                       capture_output=True)
                    self._reset_idle_timer()
                    audio_data = base64.b64decode(data.get("delta", ""))
                    if audio_data and self.audio_player and self.audio_player.stdin:
                        try:
                            self.audio_player.stdin.write(audio_data)
                            self.audio_player.stdin.flush()
                        except Exception:
                            pass

                elif event_type == "response.audio_transcript.delta":
                    text = data.get("delta", "")
                    if text:
                        print(text, end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    print("", flush=True)

                elif event_type == "response.audio.done":
                    # Audio finished playing - unmute mic after brief delay
                    self.audio_done_time = time.time()

                    async def delayed_unmute():
                        await asyncio.sleep(0.5)
                        self.playing_audio = False
                        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                                       capture_output=True)
                        self._set_status("listening")
                        print("Live session: Mic unmuted", flush=True)

                    asyncio.create_task(delayed_unmute())

                elif event_type == "response.done":
                    # Track conversation turn and token usage
                    response = data.get("response", {})
                    usage = response.get("usage", {})
                    total_tokens = usage.get("total_tokens", 0)
                    self.conversation.latest_tokens = total_tokens

                    # Extract assistant transcript from output items
                    output_items = response.get("output", [])
                    for item in output_items:
                        if item.get("type") == "message" and item.get("role") == "assistant":
                            item_id = item.get("id", "")
                            # Extract text from content
                            content = item.get("content", [])
                            text = ""
                            for part in content:
                                if part.get("type") == "audio":
                                    text = part.get("transcript", "")
                                    break
                            self.conversation.history.append({
                                "role": "assistant",
                                "item_id": item_id,
                                "text": text
                            })

                    self._reset_idle_timer()

                    # Trigger summarization if needed
                    await self.maybe_summarize()

                    # If audio.done hasn't fired yet, do delayed unmute
                    if self.playing_audio:
                        async def fallback_unmute():
                            await asyncio.sleep(1.0)
                            if self.playing_audio:
                                self.playing_audio = False
                                subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                                               capture_output=True)
                                self._set_status("listening")
                                print("Live session: Mic unmuted (fallback)", flush=True)

                        asyncio.create_task(fallback_unmute())

                elif event_type == "input_audio_buffer.speech_started":
                    self.playing_audio = False
                    subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                                   capture_output=True)
                    self._set_status("listening")
                    self._reset_idle_timer()
                    print("\n[Listening...]", flush=True)

                elif event_type == "input_audio_buffer.speech_stopped":
                    self.playing_audio = True
                    subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '1'],
                                   capture_output=True)
                    self._set_status("speaking")
                    self._reset_idle_timer()
                    print("[Processing...]", flush=True)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # Track user turn in conversation history
                    item_id = data.get("item_id", "")
                    transcript = data.get("transcript", "").strip()
                    if transcript:
                        self.conversation.history.append({
                            "role": "user",
                            "item_id": item_id,
                            "text": transcript
                        })
                        print(f"User: {transcript}", flush=True)

                elif event_type == "error":
                    error = data.get("error", {})
                    print(f"Live session error: {error}", flush=True)
                    self._set_status("error")

                elif event_type == "session.created":
                    session = data.get("session", {})
                    session_id = session.get("id", "unknown")
                    print(f"Live session: Session ID {session_id}", flush=True)

        except websockets.exceptions.ConnectionClosed:
            print("Live session: Connection closed", flush=True)
            self._set_status("disconnected")
        except Exception as e:
            print(f"Live session: Event handler error: {e}", flush=True)

    async def record_and_send(self):
        """Record audio from microphone and send to API continuously."""
        process = await asyncio.create_subprocess_exec(
            'pw-record', '--format', 's16', '--rate', '24000', '--channels', '1', '-',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print("Live session: Audio recording started", flush=True)
        chunks_sent = 0

        try:
            while self.running:
                audio_data = await process.stdout.read(CHUNK_SIZE)
                if audio_data:
                    # Skip sending while AI is speaking to avoid echo
                    time_since_audio = time.time() - self.audio_done_time
                    if not self.playing_audio and time_since_audio > 1.0:
                        await self.send_audio(audio_data)
                        chunks_sent += 1
                        if chunks_sent % 200 == 0:
                            print(f"Live session: Sent {chunks_sent} audio chunks", flush=True)
                else:
                    await asyncio.sleep(0.01)
        finally:
            process.terminate()
            await process.wait()
            print(f"Live session: Audio recording stopped ({chunks_sent} chunks sent)", flush=True)

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

            text = "\n".join(
                f"{t['role']}: {t['text']}" for t in to_summarize if t.get('text')
            )
            if not text.strip():
                return

            print("Live session: Summarizing conversation context...", flush=True)

            # Use gpt-4o-mini for cheap summarization (run in executor to not block)
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, self._summarize_text, text)

            if not summary:
                return

            # Inject summary as system message at root position
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
                if turn.get("item_id"):
                    try:
                        await self.ws.send(json.dumps({
                            "type": "conversation.item.delete",
                            "item_id": turn["item_id"]
                        }))
                    except Exception as e:
                        print(f"Live session: Failed to delete item {turn['item_id']}: {e}", flush=True)

            # Update local state
            self.conversation.history = self.conversation.history[-KEEP_LAST_TURNS:]
            print(f"Live session: Summarized {len(to_summarize)} turns", flush=True)

        except Exception as e:
            print(f"Live session: Summarization error: {e}", flush=True)
        finally:
            self.conversation.summarizing = False

    def _summarize_text(self, text):
        """Synchronous helper for summarization via gpt-4o-mini."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "Summarize this conversation concisely, preserving key facts, "
                               "decisions, and context the assistant needs to continue naturally."
                }, {
                    "role": "user",
                    "content": text
                }],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Live session: Summarization API error: {e}", flush=True)
            return None

    async def seed_context(self):
        """Inject conversation summary as first message on reconnect."""
        if not self.conversation.history:
            return

        # Build summary from existing history
        text = "\n".join(
            f"{t['role']}: {t['text']}" for t in self.conversation.history if t.get('text')
        )
        if not text.strip():
            return

        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, self._summarize_text, text)

        if summary:
            self.conversation.summary_count += 1
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "previous_item_id": "root",
                "item": {
                    "id": f"seed_{self.conversation.summary_count}",
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": f"[Previous conversation context]: {summary}"}]
                }
            }))
            print("Live session: Seeded context from previous conversation", flush=True)

    def _reset_idle_timer(self):
        """Reset the idle timeout timer."""
        self._cancel_idle_timer()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._idle_timer = loop.call_later(
                    self._idle_timeout, self._on_idle_timeout
                )
        except RuntimeError:
            pass  # No event loop available

    def _cancel_idle_timer(self):
        """Cancel the current idle timer if active."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _on_idle_timeout(self):
        """Handle idle timeout - gracefully stop the session."""
        print(f"Live session: Idle timeout ({self._idle_timeout}s), disconnecting", flush=True)
        self.running = False

    async def _interrupt(self):
        """Interrupt the current response."""
        if self.ws and self.playing_audio:
            print("Live session: Interrupting response", flush=True)
            self.playing_audio = False
            await self.ws.send(json.dumps({"type": "response.cancel"}))
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    def request_interrupt(self):
        """Thread-safe way to request an interrupt."""
        self._interrupt_requested = True

    async def run(self):
        """Run the live session main loop."""
        self._set_status("processing")
        self.running = True

        try:
            await self.connect()
            self.start_audio_player()
            self._set_status("listening")
            self._reset_idle_timer()

            # Run event handler and audio recording concurrently
            await asyncio.gather(
                self.handle_events(),
                self.record_and_send()
            )

        except Exception as e:
            print(f"Live session error: {e}", flush=True)
            self._set_status("error")
        finally:
            await self.disconnect()
            self._set_status("idle")

    def stop(self):
        """Stop the session gracefully."""
        self.running = False
        self._cancel_idle_timer()
