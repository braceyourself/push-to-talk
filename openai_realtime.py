#!/usr/bin/env python3
"""
OpenAI Realtime API integration with function calling for system interaction.
Provides low-latency voice-to-voice AI with tool execution.
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

# Claude CLI for delegation
CLAUDE_CLI = Path.home() / ".local" / "bin" / "claude"
CLAUDE_SESSION_DIR = Path.home() / ".local" / "share" / "push-to-talk" / "claude-session"

# Tools available to the AI
TOOLS = [
    {
        "type": "function",
        "name": "run_command",
        "description": "Execute a shell command and return the output. Use for system tasks like checking status, running scripts, git operations, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "type": "function",
        "name": "ask_claude",
        "description": "Delegate a complex task to Claude Code CLI. Use for coding tasks, complex reasoning, multi-step operations, or when you need Claude's expertise.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or question for Claude"
                }
            },
            "required": ["task"]
        }
    },
    {
        "type": "function",
        "name": "remember",
        "description": "Save information to memory for future reference. Use to remember preferences, facts, or context.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "A short identifier for this memory"
                },
                "value": {
                    "type": "string",
                    "description": "The information to remember"
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "type": "function",
        "name": "recall",
        "description": "Recall previously saved memories",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The memory key to recall, or 'all' for everything"
                }
            },
            "required": ["key"]
        }
    }
]

# Memory storage
MEMORY_FILE = Path.home() / ".local" / "share" / "push-to-talk" / "claude-session" / "memory.json"


def load_memory():
    """Load memory from file."""
    try:
        if MEMORY_FILE.exists():
            return json.loads(MEMORY_FILE.read_text())
    except:
        pass
    return {}


def save_memory(memory):
    """Save memory to file."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))


def execute_tool(name, arguments):
    """Execute a tool and return the result."""
    print(f"Executing tool: {name}({arguments})", flush=True)

    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments

        if name == "run_command":
            result = subprocess.run(
                args["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + result.stderr
            return json.dumps({"success": result.returncode == 0, "output": output[:2000]})

        elif name == "read_file":
            path = Path(args["path"]).expanduser()
            if path.exists():
                content = path.read_text()[:5000]
                return json.dumps({"success": True, "content": content})
            return json.dumps({"success": False, "error": "File not found"})

        elif name == "write_file":
            path = Path(args["path"]).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"])
            return json.dumps({"success": True, "message": f"Wrote {len(args['content'])} bytes"})

        elif name == "ask_claude":
            CLAUDE_SESSION_DIR.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [str(CLAUDE_CLI), '-c', '-p', args["task"],
                 '--permission-mode', 'acceptEdits',
                 '--add-dir', str(Path.home() / '.claude')],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(CLAUDE_SESSION_DIR)
            )
            return json.dumps({"response": result.stdout[:3000]})

        elif name == "remember":
            memory = load_memory()
            memory[args["key"]] = args["value"]
            save_memory(memory)
            return json.dumps({"success": True, "message": f"Remembered '{args['key']}'"})

        elif name == "recall":
            memory = load_memory()
            if args["key"] == "all":
                return json.dumps({"memories": memory})
            value = memory.get(args["key"])
            if value:
                return json.dumps({"key": args["key"], "value": value})
            return json.dumps({"error": f"No memory found for '{args['key']}'"})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Command timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


class RealtimeSession:
    """Manages a real-time voice conversation with OpenAI."""

    def __init__(self, api_key, on_status=None):
        self.api_key = api_key
        self.on_status = on_status or (lambda s: None)
        self.ws = None
        self.running = False
        self.audio_player = None
        self.playing_audio = False  # True when AI is speaking (to avoid echo)
        self.audio_done_time = 0  # Timestamp when audio finished (for cooldown)
        self._interrupt_requested = False

    def _set_status(self, status):
        self.on_status(status)

    async def connect(self):
        """Connect to OpenAI Realtime API."""
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

        # Configure session with tools
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """You are a voice assistant with direct access to the user's Linux computer.

IMPORTANT: You MUST use tools for ANY system information:
- Time/date: use run_command with 'date'
- Files: use read_file or run_command with 'ls'
- System info: use run_command
- NEVER guess or use your training data for system-specific info

Available tools: run_command, read_file, write_file, ask_claude, remember, recall.
Keep responses concise. When using tools, briefly state what you're doing, then report results.""",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700
                },
                "tools": TOOLS,
                "tool_choice": "auto"
            }
        }))

        print("Realtime API: Connected with tools", flush=True)
        return True

    async def disconnect(self):
        """Disconnect from the API."""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self.audio_player:
            self.audio_player.terminate()
            self.audio_player = None
        # Always unmute mic on disconnect
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                       capture_output=True)
        print("Realtime API: Disconnected", flush=True)

    def start_audio_player(self):
        """Start the audio player subprocess."""
        self.audio_player = subprocess.Popen(
            ['aplay', '-r', '24000', '-f', 'S16_LE', '-t', 'raw', '-q'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    async def send_audio(self, audio_data):
        """Send audio data to the API."""
        if self.ws:
            encoded = base64.b64encode(audio_data).decode('utf-8')
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": encoded
            }))

    async def handle_events(self):
        """Process incoming events from the API."""
        try:
            print("Realtime API: Event handler started", flush=True)
            async for message in self.ws:
                if not self.running:
                    break

                # Check for interrupt request
                if self._interrupt_requested:
                    self._interrupt_requested = False
                    await self.interrupt()

                data = json.loads(message)
                event_type = data.get("type", "")

                # Debug: log all event types (except audio deltas which are too frequent)
                if event_type not in ("response.audio.delta", "input_audio_buffer.append"):
                    print(f"Realtime API event: {event_type}", flush=True)

                if event_type == "response.audio.delta":
                    # Play audio chunk (and mute mic to avoid echo)
                    if not self.playing_audio:
                        self.playing_audio = True
                        self._set_status("speaking")  # Purple - AI speaking
                        # Physically mute the mic via PulseAudio
                        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '1'],
                                       capture_output=True)
                        print("Realtime API: Mic muted", flush=True)
                    audio_data = base64.b64decode(data.get("delta", ""))
                    if audio_data and self.audio_player and self.audio_player.stdin:
                        try:
                            self.audio_player.stdin.write(audio_data)
                            self.audio_player.stdin.flush()
                        except:
                            pass

                elif event_type == "response.audio_transcript.delta":
                    text = data.get("delta", "")
                    if text:
                        print(text, end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    print("", flush=True)

                elif event_type == "response.done":
                    # Check for function calls
                    response = data.get("response", {})
                    output_items = response.get("output", [])
                    print(f"Realtime API: response.done with {len(output_items)} output items", flush=True)

                    for item in output_items:
                        item_type = item.get("type")
                        print(f"Realtime API: Output item type: {item_type}", flush=True)

                        if item_type == "function_call":
                            call_id = item.get("call_id")
                            name = item.get("name")
                            arguments = item.get("arguments", "{}")

                            print(f"Realtime API: Function call - {name}({arguments})", flush=True)

                            # Execute the tool
                            self._set_status("processing")
                            result = execute_tool(name, arguments)
                            print(f"Realtime API: Tool result: {result[:200]}...", flush=True)

                            # Send result back
                            await self.ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result
                                }
                            }))

                            # Trigger response to speak the result
                            await self.ws.send(json.dumps({
                                "type": "response.create"
                            }))
                            print("Realtime API: Triggered response after tool call", flush=True)

                    self._set_status("success")
                    self.playing_audio = False  # Response done, resume mic
                    self.audio_done_time = time.time()  # Track when audio finished
                    # Delay unmute to let speaker audio clear
                    async def delayed_unmute():
                        await asyncio.sleep(1.5)
                        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                                       capture_output=True)
                        self._set_status("listening")  # Blue - ready to listen
                        print("Realtime API: Mic unmuted", flush=True)
                    asyncio.create_task(delayed_unmute())

                elif event_type == "input_audio_buffer.speech_started":
                    self.playing_audio = False  # User interrupted, resume mic
                    # Unmute mic
                    subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                                   capture_output=True)
                    self._set_status("listening")  # Blue - AI listening
                    print("\n[Listening...]", flush=True)

                elif event_type == "input_audio_buffer.speech_stopped":
                    # Mute mic immediately when user stops speaking to prevent echo
                    self.playing_audio = True
                    subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '1'],
                                   capture_output=True)
                    print("Realtime API: Mic muted (speech stopped)", flush=True)
                    self._set_status("speaking")  # Purple - AI will respond
                    print("[Processing...]", flush=True)

                elif event_type == "error":
                    error = data.get("error", {})
                    print(f"Realtime API Error: {error}", flush=True)
                    self._set_status("error")

        except websockets.exceptions.ConnectionClosed:
            print("Realtime API: Connection closed", flush=True)
        except Exception as e:
            print(f"Realtime API: Error: {e}", flush=True)

    async def record_and_send(self):
        """Record audio from microphone and send to API."""
        process = await asyncio.create_subprocess_exec(
            'pw-record', '--format', 's16', '--rate', '24000', '--channels', '1', '-',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print("Realtime API: Audio recording started", flush=True)
        chunks_sent = 0

        try:
            while self.running:
                audio_data = await process.stdout.read(CHUNK_SIZE)
                if audio_data:
                    # Skip sending while AI is speaking or during cooldown to avoid echo
                    time_since_audio = time.time() - self.audio_done_time
                    if not self.playing_audio and time_since_audio > 2.0:
                        await self.send_audio(audio_data)
                        chunks_sent += 1
                        if chunks_sent % 100 == 0:
                            print(f"Realtime API: Sent {chunks_sent} audio chunks", flush=True)
                else:
                    await asyncio.sleep(0.01)
        finally:
            process.terminate()
            await process.wait()
            print(f"Realtime API: Audio recording stopped (sent {chunks_sent} chunks)", flush=True)

    async def run(self):
        """Run the realtime session."""
        self._set_status("processing")
        self.running = True

        try:
            await self.connect()
            self.start_audio_player()
            self._set_status("listening")  # Blue - ready to listen

            # Run both tasks concurrently
            await asyncio.gather(
                self.handle_events(),
                self.record_and_send()
            )

        except Exception as e:
            print(f"Realtime session error: {e}", flush=True)
            self._set_status("error")
        finally:
            await self.disconnect()
            self._set_status("idle")

    def stop(self):
        """Stop the session."""
        self.running = False

    async def interrupt(self):
        """Interrupt the current response."""
        if self.ws and self.playing_audio:
            print("Realtime API: Interrupting response", flush=True)
            self.playing_audio = False
            # Cancel current response
            await self.ws.send(json.dumps({"type": "response.cancel"}))
            # Clear the input buffer to start fresh
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    def request_interrupt(self):
        """Thread-safe way to request an interrupt."""
        self._interrupt_requested = True


def get_api_key():
    """Get OpenAI API key."""
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


def is_available():
    """Check if Realtime API is available."""
    return WEBSOCKETS_AVAILABLE and get_api_key() is not None
