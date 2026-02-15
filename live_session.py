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

from task_manager import TaskManager, ClaudeTask, TaskStatus

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
KEEP_LAST_TURNS = 3
REALTIME_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}      # turns to preserve when summarizing

# Task management tools for OpenAI Realtime API function calling
TASK_TOOLS = [
    {
        "type": "function",
        "name": "spawn_task",
        "description": (
            "Start a Claude CLI task in the background. Use when the user asks you to do "
            "real work -- coding, refactoring, debugging, analysis. Return immediately with "
            "a brief acknowledgment. Keep acknowledgments short, every word takes time to speak."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short descriptive name, 2-4 words"
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed prompt for Claude CLI to execute"
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to the project directory where Claude should work"
                }
            },
            "required": ["name", "prompt", "project_dir"]
        }
    },
    {
        "type": "function",
        "name": "list_tasks",
        "description": (
            "List all tasks with their current status. Use when the user asks what tasks "
            "are running or wants a status update. Summarize concisely -- every word costs "
            "time to speak aloud."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "get_task_status",
        "description": "Get status of a specific task by name or number.",
        "parameters": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or ID number"
                }
            },
            "required": ["identifier"]
        }
    },
    {
        "type": "function",
        "name": "get_task_result",
        "description": (
            "Read a task's output. For completed tasks, summarizes what was accomplished. "
            "For running tasks, shows recent progress. Keep spoken summaries brief."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or ID number"
                },
                "tail_lines": {
                    "type": "integer",
                    "description": "Number of output lines to return from the end -- use fewer for quick summaries, more for detailed results"
                }
            },
            "required": ["identifier"]
        }
    },
    {
        "type": "function",
        "name": "cancel_task",
        "description": "Cancel a running task. No confirmation needed -- just do it.",
        "parameters": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or ID number"
                }
            },
            "required": ["identifier"]
        }
    },
]


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
        self.voice = voice if voice in REALTIME_VOICES else "ash"
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
        self._unmute_task = None  # Track pending delayed_unmute to cancel on new audio
        self.muted = False  # User-toggled mute via overlay click
        self.task_manager = TaskManager()
        self._notification_queue = []  # Task completion/failure notifications

        # Register task lifecycle callbacks
        self.task_manager.on('on_task_complete', self._on_task_complete)
        self.task_manager.on('on_task_failed', self._on_task_failed)

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

    def _resolve_task(self, identifier: str) -> ClaudeTask | None:
        """Resolve a task identifier (name or ID) to a ClaudeTask."""
        # Try as integer ID first
        try:
            task_id = int(identifier)
            return self.task_manager.get_task(task_id)
        except ValueError:
            pass
        # Try name match
        return self.task_manager.find_task_by_name(identifier)

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a task management tool and return JSON result."""
        if name == "spawn_task":
            task_name = args.get("name", "unnamed task")
            prompt = args.get("prompt", "")
            project_dir = args.get("project_dir", "")
            try:
                task = await self.task_manager.spawn_task(
                    task_name, prompt, Path(project_dir)
                )
                return json.dumps({
                    "id": task.id,
                    "name": task.name,
                    "status": task.status.value
                })
            except ValueError as e:
                return json.dumps({"error": str(e)})

        elif name == "list_tasks":
            tasks = self.task_manager.get_all_tasks()
            result = []
            now = time.time()
            for t in tasks:
                info = {"id": t.id, "name": t.name, "status": t.status.value}
                if t.status == TaskStatus.COMPLETED and t.completed_at and t.started_at:
                    duration = t.completed_at - t.started_at
                    info["duration"] = f"completed in {duration:.1f}s"
                elif t.status == TaskStatus.RUNNING and t.started_at:
                    duration = now - t.started_at
                    info["duration"] = f"running for {duration:.1f}s"
                elif t.status == TaskStatus.PENDING:
                    info["duration"] = "pending"
                result.append(info)
            return json.dumps(result)

        elif name == "get_task_status":
            identifier = args.get("identifier", "")
            task = self._resolve_task(identifier)
            if not task:
                return json.dumps({"error": f"No task found matching '{identifier}'"})
            now = time.time()
            info = {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "project_dir": str(task.project_dir),
            }
            if task.status == TaskStatus.COMPLETED and task.completed_at and task.started_at:
                info["duration"] = f"completed in {task.completed_at - task.started_at:.1f}s"
            elif task.status == TaskStatus.RUNNING and task.started_at:
                info["duration"] = f"running for {now - task.started_at:.1f}s"
            # Last 5 lines of output as recent_output
            output_lines = list(task.output_lines)
            info["recent_output"] = output_lines[-5:] if output_lines else []
            return json.dumps(info)

        elif name == "get_task_result":
            identifier = args.get("identifier", "")
            tail_lines = args.get("tail_lines", 50)
            task = self._resolve_task(identifier)
            if not task:
                return json.dumps({"error": f"No task found matching '{identifier}'"})
            output = self.task_manager.get_task_output(task.id)
            lines = output.split('\n') if output else []
            tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
            return json.dumps({
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "output": '\n'.join(tail),
                "total_lines": len(lines)
            })

        elif name == "cancel_task":
            identifier = args.get("identifier", "")
            task = self._resolve_task(identifier)
            if not task:
                return json.dumps({"error": f"No task found matching '{identifier}'"})
            success = await self.task_manager.cancel_task(task.id)
            if success:
                return json.dumps({"success": True, "message": f"Task '{task.name}' cancelled"})
            else:
                return json.dumps({"success": False, "message": f"Task '{task.name}' is not running"})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    async def _on_task_complete(self, task):
        """Handle task completion -- queue notification for delivery."""
        if not self.running or not self.ws:
            return
        duration = (task.completed_at or time.time()) - (task.started_at or task.created_at)
        # Get last few lines of output for summary context
        output_tail = '\n'.join(list(task.output_lines)[-20:])
        notification = {
            "type": "task_complete",
            "task_id": task.id,
            "task_name": task.name,
            "duration": f"{duration:.1f}s",
            "output_summary": output_tail[:1500],
            "project_dir": str(task.project_dir)
        }
        self._notification_queue.append(notification)
        print(f"Live session: Queued completion notification for task {task.id} '{task.name}'", flush=True)

        # If not currently speaking, flush immediately
        if not self.playing_audio:
            await self._flush_notifications()

    async def _on_task_failed(self, task):
        """Handle task failure -- queue notification for delivery."""
        if not self.running or not self.ws:
            return
        duration = (task.completed_at or time.time()) - (task.started_at or task.created_at)
        output_tail = '\n'.join(list(task.output_lines)[-20:])
        notification = {
            "type": "task_failed",
            "task_id": task.id,
            "task_name": task.name,
            "duration": f"{duration:.1f}s",
            "exit_code": task.return_code,
            "error_output": output_tail[:1500],
            "project_dir": str(task.project_dir)
        }
        self._notification_queue.append(notification)
        print(f"Live session: Queued failure notification for task {task.id} '{task.name}'", flush=True)

        if not self.playing_audio:
            await self._flush_notifications()

    async def _flush_notifications(self):
        """Deliver any queued task notifications to the conversation."""
        if not self._notification_queue or not self.ws or not self.running:
            return

        # Drain the queue
        notifications = self._notification_queue[:]
        self._notification_queue.clear()

        for notification in notifications:
            notif_type = notification["type"]
            task_name = notification["task_name"]

            if notif_type == "task_complete":
                message = (
                    f"[Task notification] Task '{task_name}' (ID {notification['task_id']}) "
                    f"completed successfully in {notification['duration']}. "
                    f"Project: {notification['project_dir']}. "
                    f"Output summary:\n{notification['output_summary']}\n\n"
                    f"Inform the user briefly that this task finished. One or two sentences max."
                )
            elif notif_type == "task_failed":
                message = (
                    f"[Task notification] Task '{task_name}' (ID {notification['task_id']}) "
                    f"failed after {notification['duration']} with exit code {notification['exit_code']}. "
                    f"Project: {notification['project_dir']}. "
                    f"Error output:\n{notification['error_output']}\n\n"
                    f"Inform the user briefly that this task failed and what went wrong. Keep it short."
                )
            else:
                continue

            print(f"Live session: Delivering notification for task '{task_name}'", flush=True)

            # Inject as system message so AI can speak about it
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": message}]
                }
            }))

            # Trigger AI to speak the notification
            await self.ws.send(json.dumps({
                "type": "response.create"
            }))

    async def _inject_task_context(self):
        """Inject current task status into conversation for ambient awareness."""
        if not self.ws or not self.running:
            return
        running = self.task_manager.get_running_tasks()
        recent_completed = [
            t for t in self.task_manager.get_all_tasks()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            and t.completed_at and (time.time() - t.completed_at) < 300  # last 5 min
        ]

        if not running and not recent_completed:
            return

        parts = []
        if running:
            task_strs = [f"'{t.name}' (running {time.time() - (t.started_at or t.created_at):.0f}s)" for t in running]
            parts.append(f"Running tasks: {', '.join(task_strs)}")
        if recent_completed:
            task_strs = [f"'{t.name}' ({t.status.value})" for t in recent_completed[:3]]
            parts.append(f"Recently finished: {', '.join(task_strs)}")

        context = "[Background task status] " + ". ".join(parts)

        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": context}]
            }
        }))

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
                "tools": TASK_TOOLS,
                "tool_choice": "auto"
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
                    # Cancel any pending delayed_unmute -- new audio is starting
                    if self._unmute_task and not self._unmute_task.done():
                        self._unmute_task.cancel()
                        self._unmute_task = None
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
                        # Bail if new audio started while we waited
                        if self._unmute_task and self._unmute_task.cancelled():
                            return
                        self.playing_audio = False
                        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'],
                                       capture_output=True)
                        self._set_status("listening")
                        print("Live session: Mic unmuted", flush=True)

                    self._unmute_task = asyncio.create_task(delayed_unmute())

                elif event_type == "response.done":
                    # Track conversation turn and token usage
                    response = data.get("response", {})
                    usage = response.get("usage", {})
                    total_tokens = usage.get("total_tokens", 0)
                    self.conversation.latest_tokens = total_tokens

                    # Extract assistant transcript from output items
                    output_items = response.get("output", [])
                    has_function_call = False
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

                    # Handle function calls from tool use
                    for item in output_items:
                        if item.get("type") == "function_call":
                            has_function_call = True
                            call_id = item.get("call_id")
                            fn_name = item.get("name")
                            arguments = item.get("arguments", "{}")
                            print(f"Live session: Tool call - {fn_name}({arguments})", flush=True)

                            # Execute tool asynchronously
                            try:
                                fn_args = json.loads(arguments)
                                result = await self._execute_tool(fn_name, fn_args)
                            except Exception as e:
                                result = json.dumps({"error": str(e)})

                            # Send result back to conversation
                            await self.ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result
                                }
                            }))

                            # Trigger AI to respond with the result
                            await self.ws.send(json.dumps({
                                "type": "response.create"
                            }))

                    self._reset_idle_timer()

                    # Skip summarize and fallback unmute during tool call cycles --
                    # those will fire on the final response.done (after tool results)
                    if not has_function_call:
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

                            self._unmute_task = asyncio.create_task(fallback_unmute())

                        # Flush pending task notifications after response completes
                        if self._notification_queue:
                            async def delayed_flush():
                                await asyncio.sleep(1.5)
                                if not self.playing_audio:
                                    await self._flush_notifications()
                            asyncio.create_task(delayed_flush())

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

                    # Inject ambient task awareness before AI responds
                    await self._inject_task_context()

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

        mute_signal = Path(__file__).parent / "live_mute_toggle"
        try:
            while self.running:
                # Check for mute toggle signal from overlay
                if mute_signal.exists():
                    try:
                        command = mute_signal.read_text().strip()
                        mute_signal.unlink()
                        if command == "stop":
                            print("Live session: Stop requested by user", flush=True)
                            self.running = False
                            break
                        elif command == "mute":
                            self.muted = True
                            self._set_status("muted")
                            self._reset_idle_timer()
                            print("Live session: Muted by user", flush=True)
                        else:
                            # Legacy toggle
                            self.muted = not self.muted
                            status = "muted" if self.muted else "listening"
                            self._set_status(status)
                            self._reset_idle_timer()
                            print(f"Live session: {'Muted' if self.muted else 'Unmuted'} by user", flush=True)
                    except Exception:
                        pass

                audio_data = await process.stdout.read(CHUNK_SIZE)
                if audio_data:
                    # Skip sending while AI is speaking, or while user-muted
                    time_since_audio = time.time() - self.audio_done_time
                    if not self.muted and not self.playing_audio and time_since_audio > 1.0:
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
        self.muted = False

        # Clean up stale toggle signal from previous session
        stale_signal = Path(__file__).parent / "live_mute_toggle"
        if stale_signal.exists():
            try:
                stale_signal.unlink()
            except Exception:
                pass

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
