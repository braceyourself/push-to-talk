#!/usr/bin/env python3
"""
Live voice conversation session using a cascaded pipeline:
  Streaming STT (Deepgram) -> Claude CLI (streaming + MCP tools) -> OpenAI TTS -> PyAudio playback

Uses the locally installed Claude CLI for LLM processing — no Anthropic API key required.
Tools are exposed via an MCP server that proxies calls back to this process over a Unix socket.
"""

import os
import sys
import json
import asyncio
import subprocess
import threading
import time
import re
import wave
import random
import tempfile
import shutil
from collections import deque
from pathlib import Path

import pysbd

from pipeline_frames import PipelineFrame, FrameType
from response_library import ResponseLibrary
from stream_composer import StreamComposer, AudioSegment, SegmentType
from task_manager import TaskManager, ClaudeTask, TaskStatus
from event_bus import EventBus, EventBusWriter, BusEvent, EventType, build_llm_context

# Module-level sentence segmenter for post-tool-buffer and end-of-turn flushing
_sentence_segmenter = pysbd.Segmenter(language="en", clean=False)

# Question patterns for detecting when AI asks user a question
_AI_QUESTION_PATTERNS = re.compile(
    r'\b(should I|do you want|would you like|shall I|can I|what do you think)\b',
    re.IGNORECASE,
)

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4096  # bytes per read from PulseAudio
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# TTS sentence buffer — accumulate text until a sentence boundary before sending to TTS
SENTENCE_END_RE = re.compile(r'[.!?]\s|[.!?]$|\n')

# Supported OpenAI TTS voices
TTS_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}

# Piper TTS configuration
PIPER_CMD = str(Path.home() / ".local" / "share" / "push-to-talk" / "venv" / "bin" / "piper")
PIPER_MODEL = str(Path.home() / ".local" / "share" / "push-to-talk" / "piper-voices" / "en_US-lessac-medium.onnx")
PIPER_SAMPLE_RATE = 22050

# Tool intent map: human-readable descriptions for overlay display during tool use
TOOL_INTENT_MAP = {
    "spawn_task": "Starting a task",
    "list_tasks": "Checking tasks",
    "get_task_status": "Checking task progress",
    "get_task_result": "Getting task results",
    "cancel_task": "Cancelling a task",
    "run_command": "Running a command",
    "read_file": "Reading a file",
}

# ── Security guards (mirror protect-reads.sh / protect-bash.sh) ──

_SENSITIVE_FILENAMES = {'.npmrc', '.netrc', '.pypirc', '.git-credentials'}

def _is_sensitive_path(path: str) -> 'tuple[bool, str]':
    """Check if a file path points to sensitive content. Mirrors protect-reads.sh."""
    p = path.rstrip('/')
    basename = os.path.basename(p)
    parts = p.split('/')

    # .env and .env.* files
    if basename == '.env' or basename.startswith('.env.'):
        return True, "Environment file"

    # credentials, secret, password in filename
    lower = basename.lower()
    for word in ('credentials', 'secret', 'password'):
        if word in lower:
            return True, f"Filename contains '{word}'"

    # .password-store directory
    if '.password-store' in parts:
        return True, "Password store"

    # /proc/*/environ
    if '/proc/' in p and p.endswith('/environ'):
        return True, "Process environment"

    # RC/credential files
    if basename in _SENSITIVE_FILENAMES:
        return True, f"Credential file ({basename})"

    # Cloud/SSH directories
    if '/.aws/' in p or ('/.ssh/' in p and ('id_' in basename or basename == 'authorized_keys')):
        return True, "Cloud/SSH credential"
    if '/.gcloud/' in p:
        return True, "GCloud credential"

    # Key file extensions
    for ext in ('.pem', '.p12', '.pfx', '.key'):
        if p.endswith(ext):
            return True, f"Key file ({ext})"

    return False, ""


_COMMAND_READ_CMDS = re.compile(r'\b(cat|head|tail|less|more)\b')
_COMMAND_SENSITIVE_ARGS = re.compile(
    r'(\.env\b|credentials|secret|password|\.pem\b|\.key\b|/proc/\S+/environ)'
)

def _is_sensitive_command(command: str) -> 'tuple[bool, str]':
    """Check if a shell command accesses sensitive content. Mirrors protect-bash.sh."""
    cmd = command.strip()

    # printenv
    if re.match(r'\bprintenv\b', cmd):
        return True, "printenv exposes environment variables"

    # pass show
    if re.match(r'\bpass\s+show\b', cmd):
        return True, "pass show exposes passwords"

    # gpg --decrypt / gpg -d
    if re.search(r'\bgpg\s+(--decrypt|-d)\b', cmd):
        return True, "gpg decrypt exposes encrypted content"

    # /proc/*/environ anywhere in command
    if re.search(r'/proc/\S+/environ', cmd):
        return True, "Accessing process environment"

    # cat/head/tail/less/more + sensitive file pattern
    if _COMMAND_READ_CMDS.search(cmd) and _COMMAND_SENSITIVE_ARGS.search(cmd):
        return True, "Reading sensitive file"

    return False, ""


class CircuitBreaker:
    """Track consecutive failures per service and trip to fallback."""

    def __init__(self, name, max_failures=3, recovery_time=60):
        self.name = name
        self.max_failures = max_failures
        self.recovery_time = recovery_time
        self.failures = 0
        self.tripped = False
        self.tripped_at = 0

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.max_failures:
            self.tripped = True
            self.tripped_at = time.time()
            print(f"Circuit breaker: {self.name} tripped after {self.failures} failures", flush=True)

    def record_success(self):
        self.failures = 0
        if self.tripped:
            self.tripped = False
            print(f"Circuit breaker: {self.name} recovered", flush=True)

    def is_tripped(self) -> bool:
        if self.tripped and (time.time() - self.tripped_at) > self.recovery_time:
            # Auto-retry after recovery period
            print(f"Circuit breaker: {self.name} attempting recovery", flush=True)
            self.tripped = False
            self.failures = 0
        return self.tripped


class LiveSession:
    """Cascaded voice pipeline: STT -> Claude CLI -> TTS -> Playback."""

    def __init__(self, openai_api_key=None, deepgram_api_key=None,
                 voice="ash", model="claude-sonnet-4-5-20250929", on_status=None,
                 fillers_enabled=True, barge_in_enabled=True, whisper_model=None,
                 idle_timeout=0, sse_dashboard=False, sse_port=9847):
        self.openai_api_key = openai_api_key
        self.deepgram_api_key = deepgram_api_key
        self.whisper_model = whisper_model
        self.voice = voice if voice in TTS_VOICES else "ash"
        self.model = model
        self.on_status = on_status or (lambda s: None)

        # SSE dashboard
        self._sse_dashboard = sse_dashboard
        self._sse_port = sse_port
        self._sse_clients: list[asyncio.StreamWriter] = []
        self._sse_server = None

        # Event bus (created in run())
        self._bus: EventBus | None = None

        self.running = False
        self.playing_audio = False
        self.audio_done_time = 0
        self._interrupt_requested = False
        self.muted = False

        # STT gating: mic stays live during playback, but STT ignores audio
        self._stt_gated = False
        self._was_stt_gated = False  # Tracks previous state for transition detection
        # VAD state for barge-in detection
        self._vad_speech_count = 0
        self._barge_in_cooldown_until = 0

        # Generation ID for interrupt coherence — all frames carry this.
        # On interrupt, ID increments; stages discard stale frames.
        self.generation_id = 0

        self.personality_prompt = self._build_personality()

        self._idle_timer = None
        self._idle_timeout = idle_timeout

        # Playback state
        self._playback_buffer = deque()
        self._bytes_played = 0

        # Task manager
        self.task_manager = TaskManager()
        self._notification_queue = []
        self.task_manager.on('on_task_complete', self._on_task_complete)
        self.task_manager.on('on_task_failed', self._on_task_failed)

        # Mic mute state tracking
        self._unmute_task = None

        # Conversation logging and learner
        self._session_log_path = None
        self._learner_process = None
        self._clip_factory_process = None

        # Filler system
        self.fillers_enabled = fillers_enabled
        self._filler_clips = {}  # {category: [pcm_bytes, ...]}
        self._last_filler = {}   # {category: index} for no-repeat guard
        self._filler_cancel = None  # asyncio.Event to cancel filler playback
        self._ack_cancel = None    # asyncio.Event to cancel acknowledgment playback
        if self.fillers_enabled:
            self._load_filler_clips()

        # Response library (replaces random filler selection)
        self._response_library = ResponseLibrary()
        self._classifier_process = None
        self._classifier_socket_path = None
        self._seed_process = None

        # AI question tracking (for trivial detection context)
        self._ai_asked_question = False

        # Stream composer (created in run())
        self._composer = None

        # Barge-in (VAD) system
        self.barge_in_enabled = barge_in_enabled
        self._vad_model = None
        self._vad_state = None

        # Barge-in: sentence tracking and interruption annotation
        self._spoken_sentences = []         # Sentences sent to TTS in current turn
        self._played_sentence_count = 0     # How many sentences had audio fully played
        self._full_response_text = ""       # Full LLM response for current turn
        self._was_interrupted = False        # Set by _trigger_barge_in, cleared at new turn
        self._barge_in_annotation = None    # Annotation prepended to next user message
        self._post_barge_in = False         # Shortened silence after barge-in

        # CLI response reader task — tracked so barge-in can cancel it
        self._response_reader_task = None

        # CLI session persistence — resume conversations across restarts
        self._cli_session_id = None

        # Speech activity tracking — filler manager checks this to avoid playing over speech
        self._last_speech_energy_time = 0.0

        # Circuit breakers for service fallback
        self._stt_breaker = CircuitBreaker("STT/Deepgram")
        self._tts_breaker = CircuitBreaker("TTS/OpenAI")

        # Pipeline queues (created in run())
        self._audio_in_q = None
        self._stt_out_q = None
        self._llm_out_q = None
        self._audio_out_q = None

        # STT flush signal — set when mic mutes to flush accumulated transcripts
        self._stt_flush_event = None  # Created in run()
        self._loop = None  # Event loop reference for thread-safe calls

        # Claude CLI subprocess and IPC
        self._cli_process = None
        self._cli_ready = False
        self._last_send_time = None
        self._tool_ipc_server = None
        self._tool_socket_path = None
        self._mcp_config_path = None

        # Find claude CLI
        self._claude_cli_path = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")

    # ── Personality (reused unchanged) ──────────────────────────────

    def _build_personality(self):
        """Load personality from .md files, memories, and CLAUDE.md context."""
        parts = []
        personality_dir = Path(__file__).parent / "personality"

        # 1. Core personality files
        if personality_dir.exists():
            for md_file in sorted(personality_dir.glob("*.md")):
                content = md_file.read_text().strip()
                if content:
                    parts.append(content)

        # 2. Persistent memories
        memories_dir = personality_dir / "memories"
        if memories_dir.exists():
            memory_files = sorted(memories_dir.glob("*.md"))
            if memory_files:
                memory_parts = []
                for md_file in memory_files:
                    content = md_file.read_text().strip()
                    if content:
                        memory_parts.append(content)
                if memory_parts:
                    parts.append("# Memories\n\n" + "\n\n".join(memory_parts))

        # 3. User global context
        global_claude = Path.home() / ".claude" / "CLAUDE.md"
        if global_claude.exists():
            content = global_claude.read_text().strip()
            if content:
                parts.append(f"# User Context\n\n{content}")

        # 4. Project context
        project_claude = Path(__file__).parent / "CLAUDE.md"
        if project_claude.exists():
            content = project_claude.read_text().strip()
            if content:
                parts.append(f"# Project Context\n\n{content}")

        return "\n\n".join(parts)

    # ── Conversation Logger ──────────────────────────────────────────

    def _log_event(self, event_type, **kwargs):
        """Emit an event to the bus (replaces direct JSONL log writes)."""
        if self._bus:
            self._bus.emit(event_type, gen=self.generation_id, **kwargs)

    def _spawn_learner(self):
        """Spawn the background learner daemon that watches the event bus."""
        learner_script = Path(__file__).parent / "learner.py"
        if not learner_script.exists():
            print("Live session: learner.py not found, skipping", flush=True)
            return
        # Pass session dir — learner resolves to events.jsonl
        session_dir = self._session_log_path.parent if self._session_log_path else None
        if not session_dir:
            return
        cmd = [sys.executable, str(learner_script), str(session_dir)]
        try:
            self._learner_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"Live session: Learner spawned (PID {self._learner_process.pid})", flush=True)
        except Exception as e:
            print(f"Live session: Failed to spawn learner: {e}", flush=True)

    def _spawn_clip_factory(self):
        """Spawn the clip factory to top up the acknowledgment filler pool."""
        factory_script = Path(__file__).parent / "clip_factory.py"
        if not factory_script.exists():
            print("Live session: clip_factory.py not found, skipping", flush=True)
            return
        cmd = [sys.executable, str(factory_script)]
        try:
            self._clip_factory_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"Live session: Clip factory spawned (PID {self._clip_factory_process.pid})", flush=True)
        except Exception as e:
            print(f"Live session: Failed to spawn clip factory: {e}", flush=True)

    # ── Classifier Daemon + Response Library ─────────────────────────

    def _spawn_classifier(self):
        """Spawn the classifier daemon process."""
        self._classifier_socket_path = f"/tmp/ptt-classifier-{os.getpid()}.sock"
        if os.path.exists(self._classifier_socket_path):
            os.unlink(self._classifier_socket_path)

        classifier_script = Path(__file__).parent / "input_classifier.py"
        if not classifier_script.exists():
            print("Live session: input_classifier.py not found, skipping", flush=True)
            return

        cmd = [sys.executable, str(classifier_script), self._classifier_socket_path]
        try:
            self._classifier_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait for readiness signal (up to 3 seconds)
            import select
            ready_fds, _, _ = select.select([self._classifier_process.stdout], [], [], 3.0)
            if ready_fds:
                line = self._classifier_process.stdout.readline().decode().strip()
                if "CLASSIFIER_READY" in line:
                    print(f"Live session: Classifier spawned (PID {self._classifier_process.pid})", flush=True)
                else:
                    print(f"Live session: Classifier unexpected output: {line}", flush=True)
            else:
                print("Live session: Classifier readiness timeout (3s), continuing", flush=True)
        except Exception as e:
            print(f"Live session: Failed to spawn classifier: {e}", flush=True)
            self._classifier_process = None

    async def _classify_input(self, text: str) -> dict:
        """Send text to classifier daemon, get classification result."""
        if not self._classifier_socket_path:
            return {"category": "acknowledgment", "confidence": 0.0}
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._classifier_socket_path),
                timeout=0.1
            )
            request = json.dumps({
                "text": text,
                "ai_asked_question": self._ai_asked_question,
            }) + "\n"
            writer.write(request.encode())
            await writer.drain()
            response = await asyncio.wait_for(reader.readline(), timeout=0.1)
            writer.close()
            return json.loads(response.decode().strip())
        except Exception as e:
            print(f"  [classifier] IPC error: {e}", flush=True)
            return {"category": "acknowledgment", "confidence": 0.0}

    def _load_response_library(self):
        """Load the categorized response library. Falls back to existing ack clips if not available."""
        self._response_library.load()
        if self._response_library.is_loaded():
            print(f"Live session: Response library loaded", flush=True)
        else:
            print("Live session: Response library empty, will use ack clip fallback", flush=True)

    def _ensure_seed_library(self):
        """Check if response library needs seed generation. Run in background if needed."""
        from response_library import LIBRARY_META
        if LIBRARY_META.exists():
            return  # Already seeded

        seed_phrases = Path(__file__).parent / "seed_phrases.json"
        if not seed_phrases.exists():
            return  # No seed phrases available

        factory_script = Path(__file__).parent / "clip_factory.py"
        if not factory_script.exists():
            return

        print("Live session: Seed library not found, generating in background...", flush=True)
        cmd = [sys.executable, str(factory_script), "--seed-responses"]
        try:
            self._seed_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"Live session: Seed generation started (PID {self._seed_process.pid})", flush=True)
        except Exception as e:
            print(f"Live session: Failed to start seed generation: {e}", flush=True)

    # ── Task tools (reused unchanged) ──────────────────────────────

    def _resolve_task(self, identifier: str) -> ClaudeTask | None:
        """Resolve a task identifier (name or ID) to a ClaudeTask."""
        try:
            task_id = int(identifier)
            return self.task_manager.get_task(task_id)
        except ValueError:
            pass
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
                    "id": task.id, "name": task.name, "status": task.status.value
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
                    info["duration"] = f"completed in {t.completed_at - t.started_at:.1f}s"
                elif t.status == TaskStatus.RUNNING and t.started_at:
                    info["duration"] = f"running for {now - t.started_at:.1f}s"
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
                "id": task.id, "name": task.name, "status": task.status.value,
                "project_dir": str(task.project_dir),
            }
            if task.status == TaskStatus.COMPLETED and task.completed_at and task.started_at:
                info["duration"] = f"completed in {task.completed_at - task.started_at:.1f}s"
            elif task.status == TaskStatus.RUNNING and task.started_at:
                info["duration"] = f"running for {now - task.started_at:.1f}s"
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
                "id": task.id, "name": task.name, "status": task.status.value,
                "output": '\n'.join(tail), "total_lines": len(lines)
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

        elif name == "run_command":
            command = args.get("command", "").strip()
            if not command:
                return json.dumps({"error": "No command provided"})
            blocked, reason = _is_sensitive_command(command)
            if blocked:
                print(f"Live session: Blocked command: {command!r} — {reason}", flush=True)
                return json.dumps({"error": f"Blocked: {reason}"})
            working_dir = args.get("working_dir", None)
            timeout = min(int(args.get("timeout", 30)), 120)
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                MAX_OUTPUT = 10240  # 10KB
                stdout_str = stdout.decode(errors='replace')
                if len(stdout_str) > MAX_OUTPUT:
                    stdout_str = stdout_str[:MAX_OUTPUT] + "\n[...truncated]"
                result = {"exit_code": proc.returncode, "stdout": stdout_str}
                if stderr:
                    stderr_str = stderr.decode(errors='replace')
                    if len(stderr_str) > MAX_OUTPUT:
                        stderr_str = stderr_str[:MAX_OUTPUT] + "\n[...truncated]"
                    result["stderr"] = stderr_str
                return json.dumps(result)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return json.dumps({"error": f"Command timed out after {timeout}s"})
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif name == "read_file":
            file_path = args.get("path", "")
            if not file_path:
                return json.dumps({"error": "No path provided"})
            blocked, reason = _is_sensitive_path(file_path)
            if blocked:
                print(f"Live session: Blocked file read: {file_path!r} — {reason}", flush=True)
                return json.dumps({"error": f"Blocked: {reason}"})
            try:
                p = Path(file_path)
                if p.is_dir():
                    return json.dumps({"error": f"Path is a directory: {file_path}"})
                if not p.exists():
                    return json.dumps({"error": f"File not found: {file_path}"})
                offset = int(args.get("offset", 0))
                limit = int(args.get("limit", 0))
                MAX_OUTPUT = 10240
                with open(p, 'r', errors='replace') as f:
                    lines = f.readlines()
                total_lines = len(lines)
                if offset > 0:
                    lines = lines[offset:]
                if limit > 0:
                    lines = lines[:limit]
                content = ''.join(lines)
                if len(content) > MAX_OUTPUT:
                    content = content[:MAX_OUTPUT] + "\n[...truncated]"
                result = {
                    "content": content,
                    "total_lines": total_lines,
                    "showing": f"lines {offset+1}-{offset+len(lines)} of {total_lines}"
                }
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif name == "send_notification":
            title = args.get("title", "Notification")
            body = args.get("body", "")
            urgency = args.get("urgency", "normal")
            try:
                import subprocess as _sp
                cmd = ['notify-send', '-u', urgency, title, body]
                _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                return json.dumps({"success": True})
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif name == "get_pipeline_events":
            last_n = int(args.get("last_n", 20))
            event_type = args.get("event_type", None)
            since_seconds = float(args.get("since_seconds", 300))
            since_ts = time.time() - since_seconds if since_seconds > 0 else None
            if self._bus:
                events = self._bus.read_recent(last_n=last_n, event_type=event_type,
                                                since_ts=since_ts)
                return json.dumps([{
                    "ts": e.ts, "type": e.type, "gen": e.gen, **e.payload
                } for e in events])
            return json.dumps([])

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    # ── Notifications ──────────────────────────────────────────────

    async def _on_task_complete(self, task):
        """Handle task completion — emit bus event + queue notification for delivery."""
        if not self.running:
            return
        duration = (task.completed_at or time.time()) - (task.started_at or task.created_at)
        output_tail = '\n'.join(list(task.output_lines)[-20:])
        if self._bus:
            self._bus.emit("task_complete", gen=self.generation_id,
                          task_id=task.id, task_name=task.name,
                          duration=f"{duration:.1f}s", output_summary=output_tail[:1500])
        self._notification_queue.append({
            "type": "task_complete",
            "task_id": task.id, "task_name": task.name,
            "duration": f"{duration:.1f}s",
            "output_summary": output_tail[:1500],
            "project_dir": str(task.project_dir)
        })
        print(f"Live session: Queued completion notification for task {task.id} '{task.name}'", flush=True)
        if not self.playing_audio:
            await self._flush_notifications()

    async def _on_task_failed(self, task):
        """Handle task failure — emit bus event + queue notification for delivery."""
        if not self.running:
            return
        duration = (task.completed_at or time.time()) - (task.started_at or task.created_at)
        output_tail = '\n'.join(list(task.output_lines)[-20:])
        if self._bus:
            self._bus.emit("task_failed", gen=self.generation_id,
                          task_id=task.id, task_name=task.name,
                          duration=f"{duration:.1f}s", error_output=output_tail[:1500])
        self._notification_queue.append({
            "type": "task_failed",
            "task_id": task.id, "task_name": task.name,
            "duration": f"{duration:.1f}s",
            "exit_code": task.return_code,
            "error_output": output_tail[:1500],
            "project_dir": str(task.project_dir)
        })
        print(f"Live session: Queued failure notification for task {task.id} '{task.name}'", flush=True)
        if not self.playing_audio:
            await self._flush_notifications()

    async def _flush_notifications(self):
        """Deliver queued task notifications by sending to CLI."""
        if not self._notification_queue or not self.running:
            return
        if not self._cli_process or not self._cli_ready:
            return

        notifications = self._notification_queue[:]
        self._notification_queue.clear()

        for notification in notifications:
            notif_type = notification["type"]

            if notif_type == "task_complete":
                task_name = notification["task_name"]
                message = (
                    f"[Task notification] Task '{task_name}' (ID {notification['task_id']}) "
                    f"completed successfully in {notification['duration']}. "
                    f"Project: {notification['project_dir']}. "
                    f"Output summary:\n{notification['output_summary']}\n\n"
                    f"Inform the user briefly that this task finished. One or two sentences max."
                )
            elif notif_type == "task_failed":
                task_name = notification["task_name"]
                message = (
                    f"[Task notification] Task '{task_name}' (ID {notification['task_id']}) "
                    f"failed after {notification['duration']} with exit code {notification['exit_code']}. "
                    f"Project: {notification['project_dir']}. "
                    f"Error output:\n{notification['error_output']}\n\n"
                    f"Inform the user briefly that this task failed and what went wrong. Keep it short."
                )
            elif notif_type == "learning":
                message = (
                    f"[Learning notification] You just learned something new about the user: "
                    f"{notification['summary']}\n\n"
                    f"Briefly and naturally mention what you learned. Don't be robotic about it."
                )
            else:
                continue

            print(f"Live session: Delivering {notif_type} notification", flush=True)
            await self._send_to_cli(message)
            await self._read_cli_response()

    def _build_task_context(self) -> str:
        """Build ambient task context string to prepend to user messages."""
        running = self.task_manager.get_running_tasks()
        recent_completed = [
            t for t in self.task_manager.get_all_tasks()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            and t.completed_at and (time.time() - t.completed_at) < 300
        ]
        if not running and not recent_completed:
            return ""

        parts = []
        if running:
            task_strs = [
                f"'{t.name}' (running {time.time() - (t.started_at or t.created_at):.0f}s)"
                for t in running
            ]
            parts.append(f"Running tasks: {', '.join(task_strs)}")
        if recent_completed:
            task_strs = [f"'{t.name}' ({t.status.value})" for t in recent_completed[:3]]
            parts.append(f"Recently finished: {', '.join(task_strs)}")

        return "[Background task status] " + ". ".join(parts)

    # ── Status and timers ──────────────────────────────────────────

    def _set_status(self, status, metadata=None):
        if metadata:
            self.on_status(json.dumps({"status": status, **metadata}))
        else:
            self.on_status(status)
        if self._bus:
            self._bus.emit("status", gen=self.generation_id, status=status, metadata=metadata)

    # ── SSE Dashboard ─────────────────────────────────────────────

    def _emit_event(self, event_type, **data):
        """Emit a pipeline event via the bus (persisted + callbacks)."""
        if self._bus:
            self._bus.emit(event_type, gen=self.generation_id, **data)

    def _emit_audio_rms(self, **data):
        """Ephemeral RMS emission via bus (callbacks only, no disk)."""
        if self._bus:
            self._bus.emit_ephemeral("audio_rms", gen=self.generation_id, **data)

    def _sse_broadcast(self, evt: BusEvent):
        """Bus callback that broadcasts events to SSE clients."""
        if not self._sse_clients:
            return
        envelope = json.dumps({
            "type": evt.type,
            "ts": evt.ts,
            "gen_id": evt.gen,
            **evt.payload,
        })
        msg = f"data: {envelope}\n\n".encode()
        dead = []
        for writer in self._sse_clients:
            try:
                writer.write(msg)
            except (ConnectionError, OSError):
                dead.append(writer)
        for w in dead:
            self._sse_clients.remove(w)

    def _get_queue_depths(self):
        """Return dict of queue sizes for all pipeline queues."""
        depths = {}
        if self._audio_in_q:
            depths['audio_in'] = self._audio_in_q.qsize()
        if self._stt_out_q:
            depths['stt_out'] = self._stt_out_q.qsize()
        if self._llm_out_q:
            depths['llm_out'] = self._llm_out_q.qsize()
        if self._audio_out_q:
            depths['audio_out'] = self._audio_out_q.qsize()
        if self._composer and hasattr(self._composer, '_segment_q'):
            depths['composer'] = self._composer._segment_q.qsize()
        return depths

    def _build_state_snapshot(self):
        """Build initial state snapshot for newly connected SSE clients."""
        return {
            "type": "snapshot",
            "ts": time.time(),
            "gen_id": self.generation_id,
            "running": self.running,
            "muted": self.muted,
            "playing_audio": self.playing_audio,
            "stt_gated": self._stt_gated,
            "model": self.model,
            "voice": self.voice,
            "queue_depths": self._get_queue_depths(),
        }

    async def _sse_server_stage(self):
        """SSE server stage — serves realtime dashboard events over HTTP."""
        if not self._sse_dashboard:
            return

        async def handle_client(reader, writer):
            # Parse request line
            try:
                request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not request_line:
                    writer.close()
                    return
                parts = request_line.decode().split()
                method = parts[0] if parts else ''
                path = parts[1] if len(parts) > 1 else '/'
            except (asyncio.TimeoutError, ConnectionError, UnicodeDecodeError):
                writer.close()
                return

            # Read headers
            content_length = 0
            try:
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                    if line == b"\r\n" or not line:
                        break
                    if b':' in line:
                        k, v = line.decode().split(':', 1)
                        if k.strip().lower() == 'content-length':
                            content_length = int(v.strip())
            except (asyncio.TimeoutError, ConnectionError):
                writer.close()
                return

            # Route requests
            if method == 'GET' and '/events' in path:
                await self._handle_sse_stream(reader, writer)
            elif method == 'POST' and '/cmd' in path:
                body = await reader.read(content_length) if content_length else b'{}'
                await self._handle_command(writer, body)
            elif method == 'OPTIONS':
                writer.write(
                    b"HTTP/1.1 204 No Content\r\n"
                    b"Access-Control-Allow-Origin: *\r\n"
                    b"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                    b"Access-Control-Allow-Headers: Content-Type\r\n"
                    b"Connection: close\r\n\r\n"
                )
                try:
                    await writer.drain()
                except (ConnectionError, OSError):
                    pass
                writer.close()
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n")
                try:
                    await writer.drain()
                except (ConnectionError, OSError):
                    pass
                writer.close()

        try:
            self._sse_server = await asyncio.start_server(
                handle_client, '127.0.0.1', self._sse_port
            )
            print(f"SSE dashboard server at http://127.0.0.1:{self._sse_port}/events", flush=True)

            # Periodic queue depth emission (ephemeral — no disk write)
            async def emit_queue_depths():
                while self.running:
                    await asyncio.sleep(0.5)
                    if self._sse_clients and self._bus:
                        self._bus.emit_ephemeral("queue_depths", gen=self.generation_id,
                                                 **self._get_queue_depths())

            depth_task = asyncio.create_task(emit_queue_depths())

            # Keep server running
            while self.running:
                await asyncio.sleep(0.5)

        finally:
            if self._sse_server:
                self._sse_server.close()
                await self._sse_server.wait_closed()
            for writer in list(self._sse_clients):
                writer.close()
            self._sse_clients.clear()

    async def _handle_sse_stream(self, reader, writer):
        """Handle GET /events — SSE stream connection."""
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: keep-alive\r\n"
            b"Access-Control-Allow-Origin: *\r\n"
            b"\r\n"
        )
        try:
            await writer.drain()
        except (ConnectionError, OSError):
            writer.close()
            return

        # Send initial snapshot
        snapshot = self._build_state_snapshot()
        msg = f"data: {json.dumps(snapshot)}\n\n".encode()
        try:
            writer.write(msg)
            await writer.drain()
        except (ConnectionError, OSError):
            writer.close()
            return

        self._sse_clients.append(writer)
        print(f"SSE: Client connected ({len(self._sse_clients)} total)", flush=True)

        # Keep connection alive with periodic pings
        try:
            while self.running:
                await asyncio.sleep(15)
                try:
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
                except (ConnectionError, OSError):
                    break
        finally:
            if writer in self._sse_clients:
                self._sse_clients.remove(writer)
            writer.close()
            print(f"SSE: Client disconnected ({len(self._sse_clients)} total)", flush=True)

    async def _handle_command(self, writer, body: bytes):
        """Handle POST /cmd requests from the dashboard."""
        try:
            cmd = json.loads(body)
            action = cmd.get('action', '')
            result = {"ok": True}

            if action in ('mute', 'unmute'):
                # Emit bus command — mic stage polls bus for mute/unmute
                if self._bus:
                    self._bus.emit("command", gen=self.generation_id, action=action)
                else:
                    self.set_muted(action == 'mute')
            elif action == 'interrupt':
                self.request_interrupt()
            elif action == 'restart':
                # Write restart_live signal (same as indicator "Start Session")
                # then stop the live session — config watcher will restart it
                if self._bus:
                    self._bus.emit("command", gen=self.generation_id, action="stop")
                status_file = Path(__file__).parent / "status"
                status_file.write_text("restart_live")
                self._send_json_response(writer, result)
                await writer.drain()
                writer.close()
                return
            elif action == 'stop':
                if self._bus:
                    self._bus.emit("command", gen=self.generation_id, action="stop")
                self._send_json_response(writer, result)
                await writer.drain()
                writer.close()
                return
            else:
                result = {"ok": False, "error": f"Unknown action: {action}"}

            self._send_json_response(writer, result)
            await writer.drain()
            writer.close()
        except Exception as e:
            try:
                self._send_json_response(writer, {"ok": False, "error": str(e)})
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            writer.close()

    def _send_json_response(self, writer, data: dict):
        """Write an HTTP JSON response to a StreamWriter."""
        body = json.dumps(data).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Access-Control-Allow-Origin: *\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )

    def _reset_idle_timer(self):
        self._cancel_idle_timer()
        if self._idle_timeout <= 0:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._idle_timer = loop.call_later(self._idle_timeout, self._on_idle_timeout)
        except RuntimeError:
            pass

    def _cancel_idle_timer(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _on_idle_timeout(self):
        print(f"Live session: Idle timeout ({self._idle_timeout}s), disconnecting", flush=True)
        self.running = False
        if self._composer:
            self._composer.stop()

    # ── Mic mute/unmute ────────────────────────────────────────────

    def _mute_mic(self):
        # No-op: mic stays physically live so VAD can detect barge-in speech.
        # _stt_gated prevents transcription during playback.
        pass

    def _unmute_mic(self):
        # No-op: mic is never physically muted.
        pass

    # ── Filler System ────────────────────────────────────────────────

    def _load_filler_clips(self):
        """Load acknowledgment WAV clips as raw PCM bytes."""
        # Load acknowledgment clips (verbal phrases like "let me check that")
        ack_dir = Path(__file__).parent / "audio" / "fillers" / "acknowledgment"
        if not ack_dir.exists():
            print("Live session: No acknowledgment clips found, fillers disabled", flush=True)
            self.fillers_enabled = False
            return

        ack_clips = []
        for wav_path in sorted(ack_dir.glob("*.wav")):
            try:
                with wave.open(str(wav_path), 'rb') as wf:
                    pcm = wf.readframes(wf.getnframes())
                    rate = wf.getframerate()
                if rate != SAMPLE_RATE:
                    pcm = self._resample_22050_to_24000(pcm)
                ack_clips.append(pcm)
            except Exception as e:
                print(f"Live session: Error loading ack clip {wav_path}: {e}", flush=True)

        if ack_clips:
            self._filler_clips["acknowledgment"] = ack_clips
            self._last_filler["acknowledgment"] = -1
            print(f"Live session: Loaded {len(ack_clips)} acknowledgment clips", flush=True)
        else:
            print("Live session: No acknowledgment clips loaded, fillers disabled", flush=True)
            self.fillers_enabled = False

    def _pick_filler(self, category: str) -> bytes | None:
        """Pick a random filler clip from category, avoiding consecutive repeats."""
        clips = self._filler_clips.get(category)
        if not clips:
            return None
        if len(clips) == 1:
            return clips[0]

        last = self._last_filler.get(category, -1)
        choices = [i for i in range(len(clips)) if i != last]
        idx = random.choice(choices)
        self._last_filler[category] = idx
        return clips[idx]

    async def _filler_manager(self, user_text: str, cancel_event: asyncio.Event):
        """Play a context-appropriate quick response while waiting for LLM."""
        # Hot-reload response library if seed generation completed
        if not self._response_library.is_loaded():
            from response_library import LIBRARY_META
            if LIBRARY_META.exists():
                self._load_response_library()

        # Step 1: Classify via daemon IPC (<5ms) -- includes ai_asked_question context
        classification = await self._classify_input(user_text)
        category = classification.get("category", "acknowledgment")
        confidence = classification.get("confidence", 0.0)
        subcategory = classification.get("subcategory", "")
        match_type = classification.get("match_type", "heuristic")
        trivial = classification.get("trivial", False)

        # Reset ai_asked_question after classification (user has responded)
        self._ai_asked_question = False

        # Step 2: Trivial input -> natural silence (no filler clip)
        if trivial:
            self._log_event("classification",
                input_text=user_text,
                category=category,
                confidence=confidence,
                subcategory=subcategory,
                match_type=match_type,
                trivial=True,
                clip_id=None,
                clip_phrase=None,
                fallback_used=False,
            )
            self._set_status("thinking")
            return  # Natural silence -- no filler

        # Step 3: Low confidence -> fall back to acknowledgment
        if confidence < 0.4:
            category = "acknowledgment"

        # Step 4: Lookup clip from response library
        response = None
        clip_pcm = None
        if self._response_library.is_loaded():
            response = self._response_library.lookup(category, subcategory=subcategory)
            if response:
                clip_pcm = self._response_library.get_clip_pcm(response.id)

        # Step 5: Log classification trace
        self._log_event("classification",
            input_text=user_text,
            category=category,
            confidence=confidence,
            subcategory=subcategory,
            match_type=match_type,
            trivial=False,
            clip_id=response.id if response else None,
            clip_phrase=response.phrase if response else None,
            fallback_used=response is None,
        )

        # Step 6: Gate -- skip if LLM responds fast (500ms)
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=0.5)
            return
        except asyncio.TimeoutError:
            pass

        if cancel_event.is_set():
            return

        # Step 6b: Skip if user is still speaking (speech energy within last 500ms)
        if time.time() - self._last_speech_energy_time < 0.5:
            print("  [filler] Skipped — user still speaking", flush=True)
            return

        # Step 7: Play clip via composer (or direct if composer not available)
        if clip_pcm:
            clip_pcm = self._resample_22050_to_24000(clip_pcm)
            # Mark social fillers as sufficient — composer will suppress LLM TTS
            meta = {"sufficient": True} if category == "social" else {}
            if self._composer:
                await self._composer.enqueue(
                    AudioSegment(SegmentType.FILLER_CLIP, data=clip_pcm, metadata=meta)
                )
            else:
                await self._play_filler_audio(clip_pcm, cancel_event)
            # Log filler phrase as assistant turn in conversation history
            if response and response.phrase:
                self._log_event("assistant", text=response.phrase, filler=True)
            if response:
                barged = cancel_event.is_set()
                self._response_library.record_usage(response.id, barged_in=barged)
            return

        # Step 8: Ultimate fallback -- existing random ack clip (old system)
        clip = self._pick_filler("acknowledgment")
        if clip:
            if self._composer:
                await self._composer.enqueue(AudioSegment(SegmentType.FILLER_CLIP, data=clip))
            else:
                await self._play_filler_audio(clip, cancel_event)

    async def _tts_to_pcm(self, text: str) -> bytes | None:
        """Convert text to PCM audio via Piper. Returns resampled 24kHz bytes."""
        try:
            process = await asyncio.create_subprocess_exec(
                PIPER_CMD, '--model', PIPER_MODEL, '--output-raw',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(input=text.encode()),
                timeout=3.0
            )
            if stdout:
                return self._resample_22050_to_24000(stdout)
        except Exception as e:
            print(f"  [filler] TTS error: {e}", flush=True)
        return None

    async def _play_filler_audio(self, pcm_data: bytes, cancel_event: asyncio.Event):
        """Push filler PCM to playback queue as FILLER type frames."""
        gen_id = self.generation_id
        offset = 0
        while offset < len(pcm_data):
            if cancel_event.is_set() or self.generation_id != gen_id:
                return
            chunk = pcm_data[offset:offset + 4096]
            offset += 4096
            await self._audio_out_q.put(PipelineFrame(
                type=FrameType.FILLER,
                generation_id=gen_id,
                data=chunk
            ))

    async def _play_gated_ack(self, cancel_event: asyncio.Event, gen_id: int):
        """Play an acknowledgment clip after 300ms gate. Skip if cancelled (fast tool)."""
        # Gate: wait 300ms — if tool completes fast, skip acknowledgment
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=0.3)
            return  # Tool completed fast, skip acknowledgment
        except asyncio.TimeoutError:
            pass

        if cancel_event.is_set() or self.generation_id != gen_id:
            return

        # Play acknowledgment clip
        clip = self._pick_filler("acknowledgment")
        if clip:
            await self._play_filler_audio(clip, cancel_event)

    async def _play_gated_ack_via_composer(self, cancel_event: asyncio.Event, gen_id: int):
        """Play an acknowledgment clip via composer after 300ms gate."""
        # Gate: wait 300ms -- if tool completes fast, skip acknowledgment
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=0.3)
            return  # Tool completed fast, skip acknowledgment
        except asyncio.TimeoutError:
            pass

        if cancel_event.is_set() or self.generation_id != gen_id:
            return

        # Play acknowledgment clip via composer for consistency
        clip = self._pick_filler("acknowledgment")
        if clip:
            if self._composer:
                await self._composer.enqueue(AudioSegment(SegmentType.FILLER_CLIP, data=clip))
            else:
                await self._play_filler_audio(clip, cancel_event)

    # ── VAD Barge-in ──────────────────────────────────────────────

    def _load_vad_model(self):
        """Load Silero VAD ONNX model."""
        model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
        if not model_path.exists():
            print(f"Live session: VAD model not found at {model_path}, barge-in disabled", flush=True)
            self.barge_in_enabled = False
            return

        try:
            import onnxruntime
            self._vad_model = onnxruntime.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            # Initialize VAD state: Silero VAD v5 uses single 'state' tensor [2, 1, 128]
            # and requires a 64-sample context prepended to each 512-sample input
            import numpy as np
            self._vad_state = {
                'state': np.zeros((2, 1, 128), dtype=np.float32),
                'sr': np.array(16000, dtype=np.int64),  # scalar; always 16kHz after resample
                'context': np.zeros(64, dtype=np.float32),  # 64-sample context for continuity
            }
            print("Live session: Silero VAD loaded", flush=True)
        except Exception as e:
            print(f"Live session: Failed to load VAD model: {e}", flush=True)
            self.barge_in_enabled = False

    def _run_vad(self, audio_bytes: bytes) -> float:
        """Run VAD inference on audio chunk, return max speech probability.

        Silero VAD v5 expects 512-sample windows at 16kHz with a 64-sample
        context prepended (total input: 576 samples). Each 4096-byte chunk
        at 24kHz yields ~1365 resampled samples, which we process as
        consecutive 512-sample windows, returning the max probability.
        """
        if not self._vad_model:
            return 0.0

        import numpy as np

        # Convert bytes to float32 samples
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16kHz if needed by taking 2 of every 3 samples
        if SAMPLE_RATE == 24000:
            indices = np.arange(0, len(samples), 1.5).astype(int)
            indices = indices[indices < len(samples)]
            samples = samples[indices]

        # Process all resampled samples in 512-sample windows with 64-sample context
        context = self._vad_state['context']
        max_prob = 0.0

        try:
            for i in range(0, len(samples) - 511, 512):
                window = samples[i:i + 512]
                # Prepend context (as Silero's own OnnxWrapper does)
                input_data = np.concatenate([context, window]).reshape(1, -1)

                ort_inputs = {
                    'input': input_data,
                    'state': self._vad_state['state'],
                    'sr': self._vad_state['sr']
                }
                ort_outputs = self._vad_model.run(None, ort_inputs)
                prob = ort_outputs[0].item()
                self._vad_state['state'] = ort_outputs[1]

                # Update context from the end of this window
                context = window[-64:]

                if prob > max_prob:
                    max_prob = prob

            self._vad_state['context'] = context
            return max_prob
        except Exception as e:
            print(f"VAD error: {e}", flush=True)
            return 0.0

    def _reset_vad_state(self):
        """Reset VAD hidden state to avoid contamination between segments."""
        if self._vad_state:
            import numpy as np
            self._vad_state['state'] = np.zeros((2, 1, 128), dtype=np.float32)
            self._vad_state['context'] = np.zeros(64, dtype=np.float32)

    # ── Fallback: Whisper STT ────────────────────────────────────

    async def _stt_whisper_fallback(self):
        """Fallback STT using local Whisper when Deepgram is unavailable."""
        import numpy as np

        print("STT: Running Whisper fallback mode", flush=True)
        audio_buffer = bytearray()
        SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
        SILENCE_DURATION = 0.7   # seconds of silence to trigger transcription

        silence_start = None

        try:
            while self.running:
                try:
                    frame = self._audio_in_q.get_nowait()
                    if frame.type == FrameType.AUDIO_RAW:
                        audio_buffer.extend(frame.data)

                        # Check RMS for silence detection
                        samples = np.frombuffer(frame.data, dtype=np.int16)
                        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

                        if rms < SILENCE_THRESHOLD:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > SILENCE_DURATION and len(audio_buffer) > SAMPLE_RATE * 2:
                                # Silence detected after speech — transcribe
                                pcm_data = bytes(audio_buffer)
                                audio_buffer.clear()
                                silence_start = None

                                # Run Whisper in executor
                                transcript = await asyncio.get_event_loop().run_in_executor(
                                    None, self._whisper_transcribe, pcm_data
                                )
                                if transcript:
                                    await self._stt_out_q.put(PipelineFrame(
                                        type=FrameType.END_OF_UTTERANCE,
                                        generation_id=self.generation_id
                                    ))
                                    await self._stt_out_q.put(PipelineFrame(
                                        type=FrameType.TRANSCRIPT,
                                        generation_id=self.generation_id,
                                        data=transcript
                                    ))
                        else:
                            silence_start = None
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.02)

        except Exception as e:
            print(f"Whisper fallback error: {e}", flush=True)

    def _whisper_transcribe(self, pcm_data: bytes) -> str | None:
        """Transcribe PCM audio using faster-whisper (blocking, run in executor)."""
        try:
            import numpy as np

            if not self.whisper_model:
                from faster_whisper import WhisperModel
                self.whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

            # Convert PCM bytes to float32 numpy array (faster-whisper expects this)
            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            segments_gen, info = self.whisper_model.transcribe(
                samples, language="en",
                beam_size=5,
                condition_on_previous_text=False,  # Prevent hallucination loops
                vad_filter=True,  # Built-in Silero VAD — filters silence before transcription
            )

            # Multi-layer segment filtering: catches throat clearing, coughs,
            # background noise, and Whisper hallucinations on non-speech audio.
            kept = []
            any_segments = False
            for s in segments_gen:
                any_segments = True

                # Layer 1: no_speech_prob (high = Whisper thinks no speech)
                if s.no_speech_prob >= 0.6:
                    print(f"STT: Rejected (no_speech={s.no_speech_prob:.2f}): \"{s.text[:40]}\"", flush=True)
                    continue

                # Layer 2: low transcription confidence (catches coughs/clears
                # that have acoustic energy but produce gibberish text)
                if s.avg_logprob < -1.0:
                    print(f"STT: Rejected (logprob={s.avg_logprob:.2f}): \"{s.text[:40]}\"", flush=True)
                    continue

                # Layer 3: repetitive hallucination pattern (e.g. "thank you
                # thank you thank you" from sustained noise)
                if s.compression_ratio > 2.4:
                    print(f"STT: Rejected (compression={s.compression_ratio:.2f}): \"{s.text[:40]}\"", flush=True)
                    continue

                kept.append(s.text.strip())

            if not kept and any_segments:
                # All segments rejected — signal to overlay
                self._set_status("stt_rejected")

            text = " ".join(kept).strip()

            # Note: single-word transcripts are kept — the no_speech_prob
            # filter above is sufficient to catch non-speech sounds.

            return text if text else None

        except Exception as e:
            print(f"Whisper transcribe error: {e}", flush=True)
            return None

    # ── Fallback: Piper TTS ──────────────────────────────────────

    async def _tts_piper_fallback(self, text: str, gen_id: int):
        """Fallback TTS using local Piper (used by filler system)."""
        try:
            process = await asyncio.create_subprocess_exec(
                PIPER_CMD, '--model', PIPER_MODEL, '--output-raw',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await process.communicate(input=text.encode())
            if stdout and self.generation_id == gen_id:
                resampled = self._resample_22050_to_24000(stdout)
                offset = 0
                while offset < len(resampled):
                    if self.generation_id != gen_id:
                        break
                    chunk = resampled[offset:offset + 4096]
                    offset += 4096
                    await self._audio_out_q.put(PipelineFrame(
                        type=FrameType.TTS_AUDIO,
                        generation_id=gen_id,
                        data=chunk
                    ))
        except Exception as e:
            print(f"Piper TTS fallback error: {e}", flush=True)

    # ── Claude CLI Management ──────────────────────────────────────

    async def _start_tool_ipc_server(self):
        """Start Unix socket server for MCP tool call proxying."""
        self._tool_socket_path = f"/tmp/ptt-tools-{os.getpid()}.sock"

        # Clean up stale socket
        if os.path.exists(self._tool_socket_path):
            os.unlink(self._tool_socket_path)

        async def handle_client(reader, writer):
            try:
                data = await asyncio.wait_for(reader.readline(), timeout=30)
                if data:
                    request = json.loads(data.decode().strip())
                    tool_name = request.get("tool", "")
                    args = request.get("args", {})
                    result = await self._execute_tool(tool_name, args)
                    writer.write(result.encode() + b"\n")
                    await writer.drain()
            except Exception as e:
                error = json.dumps({"error": str(e)})
                writer.write(error.encode() + b"\n")
                try:
                    await writer.drain()
                except Exception:
                    pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        self._tool_ipc_server = await asyncio.start_unix_server(
            handle_client, self._tool_socket_path
        )
        print(f"Live session: Tool IPC server started at {self._tool_socket_path}", flush=True)

    async def _stop_tool_ipc_server(self):
        """Stop the Unix socket server."""
        if self._tool_ipc_server:
            self._tool_ipc_server.close()
            try:
                await self._tool_ipc_server.wait_closed()
            except Exception:
                pass
        if self._tool_socket_path and os.path.exists(self._tool_socket_path):
            try:
                os.unlink(self._tool_socket_path)
            except Exception:
                pass

    def _generate_mcp_config(self) -> str:
        """Generate a temporary MCP config file for the CLI."""
        mcp_server_script = str(Path(__file__).parent / "task_tools_mcp.py")

        config = {
            "mcpServers": {
                "ptt-task-tools": {
                    "command": "python3",
                    "args": [mcp_server_script],
                    "env": {
                        "PTT_TOOL_SOCKET": self._tool_socket_path
                    }
                }
            }
        }

        # Write to temp file
        fd, path = tempfile.mkstemp(suffix=".json", prefix="ptt-mcp-")
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f)

        self._mcp_config_path = path
        return path

    async def _start_cli_process(self):
        """Start the Claude CLI subprocess for LLM processing."""
        t0 = time.time()
        # Start tool IPC server first
        await self._start_tool_ipc_server()

        # Generate MCP config pointing to our tool server
        mcp_config = self._generate_mcp_config()

        # Build command
        cmd = [
            self._claude_cli_path,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model", self.model,
            "--system-prompt", self.personality_prompt,
            "--mcp-config", mcp_config,
            "--strict-mcp-config",
        ]

        # Resume previous conversation if available
        session_file = Path("~/.local/share/push-to-talk/cli_session_id").expanduser()
        if session_file.exists():
            prev_session = session_file.read_text().strip()
            if prev_session:
                cmd.extend(["--resume", prev_session])
                print(f"Live session: Resuming CLI session {prev_session}", flush=True)

        # Disable all built-in tools — only MCP tools available
        cmd.extend(["--tools", ""])

        env = {**os.environ}
        # Unset Claude Code env vars to avoid nesting error
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        env.pop("CLAUDECODE", None)

        self._cli_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._cli_ready = True
        print(f"Live session: Claude CLI started (PID {self._cli_process.pid}, model={self.model}) [{time.time()-t0:.2f}s]", flush=True)

        # Start stderr reader to log CLI errors
        asyncio.create_task(self._read_cli_stderr())

    async def _stop_cli_process(self):
        """Stop the Claude CLI subprocess and clean up."""
        self._cli_ready = False

        if self._cli_process:
            try:
                self._cli_process.stdin.close()
            except Exception:
                pass
            try:
                self._cli_process.terminate()
                await asyncio.wait_for(self._cli_process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._cli_process.kill()
                except Exception:
                    pass
            self._cli_process = None

        await self._stop_tool_ipc_server()

        # Clean up temp MCP config
        if self._mcp_config_path and os.path.exists(self._mcp_config_path):
            try:
                os.unlink(self._mcp_config_path)
            except Exception:
                pass

    async def _read_cli_stderr(self):
        """Read and log stderr from the CLI process."""
        if not self._cli_process or not self._cli_process.stderr:
            return
        try:
            while True:
                line = await self._cli_process.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    print(f"CLI stderr: {text}", flush=True)
        except Exception:
            pass

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip markdown formatting for TTS output."""
        import re
        s = text
        # Filter stage directions: "(quiet)", "(silence)", "(no response)", etc.
        s = re.sub(r'^\s*\([^)]{1,30}\)\s*$', '', s)
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)  # **bold**
        s = re.sub(r'\*(.+?)\*', r'\1', s)        # *italic*
        s = re.sub(r'__(.+?)__', r'\1', s)        # __bold__
        s = re.sub(r'_(.+?)_', r'\1', s)          # _italic_
        s = re.sub(r'`(.+?)`', r'\1', s)          # `code`
        s = re.sub(r'^#{1,6}\s+', '', s)           # # headings
        s = re.sub(r'^[-*+]\s+', '', s)            # - bullet points
        s = re.sub(r'^\d+\.\s+', '', s)            # 1. numbered lists
        s = re.sub(r'^>\s+', '', s)                # > blockquotes
        s = s.replace('```', '')                    # code fences
        return s.strip()

    async def _send_to_cli(self, text: str):
        """Send a user message to the CLI subprocess via stream-json."""
        if not self._cli_process or not self._cli_ready:
            print("Live session: CLI not ready, dropping message", flush=True)
            return

        message = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": text}
        })
        try:
            self._last_send_time = time.time()
            self._cli_process.stdin.write(message.encode() + b"\n")
            await self._cli_process.stdin.drain()
        except Exception as e:
            print(f"Live session: Error sending to CLI: {e}", flush=True)
            self._cli_ready = False

    async def _drain_stale_cli_output(self):
        """Drain leftover CLI output from a previous cancelled read until we see a result message."""
        if not self._cli_process or not self._cli_process.stdout:
            return
        drained = 0
        while True:
            try:
                line = await asyncio.wait_for(
                    self._cli_process.stdout.readline(),
                    timeout=0.1  # Very short timeout — if nothing waiting, we're clean
                )
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue
                drained += 1
                try:
                    event_data = json.loads(line_str)
                    if event_data.get("type") == "result":
                        break  # Old turn fully drained
                except json.JSONDecodeError:
                    continue
            except asyncio.TimeoutError:
                break  # Nothing waiting — stdout is clean
        if drained:
            print(f"Live session: Drained {drained} stale CLI events", flush=True)

    async def _read_cli_response(self):
        """Read streaming events from CLI until turn is complete. Route text through composer."""
        if not self._cli_process or not self._cli_process.stdout:
            return

        gen_id = self.generation_id
        sentence_buffer = ""
        full_response = ""
        first_text_time = None
        saw_tool_use = False  # Track if model is making tool calls
        sse_char_counter = 0  # for decimated text delta events

        # Reset sentence tracking for this turn
        self._spoken_sentences = []
        self._played_sentence_count = 0
        self._full_response_text = ""
        self._was_interrupted = False
        # After tool use, hold all text until turn ends so we only speak
        # the final coherent response — not intermediate narration.
        post_tool_buffer = ""

        try:
            while self.running:
                try:
                    line = await asyncio.wait_for(
                        self._cli_process.stdout.readline(),
                        timeout=120  # 2 min timeout for long tool operations
                    )
                except asyncio.TimeoutError:
                    print("Live session: CLI response timeout", flush=True)
                    break

                if not line:
                    # CLI process ended
                    print("Live session: CLI process ended unexpectedly", flush=True)
                    self._cli_ready = False
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    event_data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Capture CLI session ID from any event that carries it
                cli_sid = event_data.get("session_id")
                if cli_sid and not self._cli_session_id:
                    self._cli_session_id = cli_sid
                    session_file = Path("~/.local/share/push-to-talk/cli_session_id").expanduser()
                    session_file.parent.mkdir(parents=True, exist_ok=True)
                    session_file.write_text(cli_sid)
                    print(f"Live session: CLI session ID: {cli_sid}", flush=True)

                # Check for interrupt
                if self.generation_id != gen_id:
                    # Keep reading until we see the result message, but discard output
                    if event_data.get("type") == "result":
                        break
                    continue

                event_type = event_data.get("type", "")

                if event_type == "stream_event":
                    event = event_data.get("event", {})
                    inner_type = event.get("type", "")

                    if inner_type == "content_block_start":
                        content_block = event.get("content_block", {})
                        if content_block.get("type") == "tool_use":
                            tool_name = content_block.get("name", "unknown")
                            # Strip MCP prefix (e.g. "mcp__ptt-task-tools__spawn_task" -> "spawn_task")
                            bare_name = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name
                            intent = TOOL_INTENT_MAP.get(bare_name, bare_name.replace('_', ' ').title())

                            if not saw_tool_use:
                                saw_tool_use = True
                                # First tool call — reset composer to drain
                                # any pre-tool sentences, then drain audio_out
                                if self._composer:
                                    self._composer.reset()
                                drained = 0
                                while not self._audio_out_q.empty():
                                    try:
                                        f = self._audio_out_q.get_nowait()
                                        if f.type == FrameType.END_OF_TURN:
                                            await self._audio_out_q.put(f)
                                            break
                                        drained += 1
                                    except asyncio.QueueEmpty:
                                        break
                                if drained:
                                    print(f"  [tool] Drained {drained} pre-tool frames", flush=True)

                                # Play acknowledgment clip with 300ms gate
                                if self.fillers_enabled:
                                    ack_cancel = asyncio.Event()
                                    self._ack_cancel = ack_cancel
                                    asyncio.create_task(
                                        self._play_gated_ack_via_composer(ack_cancel, gen_id)
                                    )

                            # Discard any accumulated text (pre-tool or inter-tool narration)
                            if sentence_buffer.strip():
                                print(f"  [tool] Suppressed text: \"{sentence_buffer.strip()[:60]}\"", flush=True)
                                sentence_buffer = ""
                            # Also discard inter-tool post_tool_buffer
                            if post_tool_buffer.strip():
                                print(f"  [tool] Suppressed inter-tool text: \"{post_tool_buffer.strip()[:60]}\"", flush=True)
                                post_tool_buffer = ""
                            self._set_status("tool_use", {"intent": intent})
                            self._emit_event("llm_tool_use", tool_name=bare_name, intent=intent)

                    elif inner_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if first_text_time is None:
                                first_text_time = time.time()
                                ft_ms = (first_text_time - (self._last_send_time or first_text_time)) * 1000
                                self._emit_event("llm_first_token", latency_ms=round(ft_ms, 1))
                                print(f"\n  [timing] first token: {ft_ms/1000:.2f}s", flush=True)
                            full_response += text
                            sse_char_counter += len(text)
                            if sse_char_counter >= 50:
                                self._emit_event("llm_text_delta", chars=sse_char_counter, total_chars=len(full_response))
                                sse_char_counter = 0
                            print(text, end="", flush=True)

                            if saw_tool_use:
                                # After tool use: hold text, don't send to TTS yet
                                post_tool_buffer += text
                                # Cancel ack on first post-tool text
                                if self._ack_cancel and not self._ack_cancel.is_set():
                                    self._ack_cancel.set()
                            else:
                                # Normal path: stream text to TTS
                                sentence_buffer += text

                                # Cancel filler on first text from LLM
                                if self._filler_cancel and not self._filler_cancel.is_set():
                                    self._filler_cancel.set()
                                # Cancel acknowledgment if still pending
                                if self._ack_cancel and not self._ack_cancel.is_set():
                                    self._ack_cancel.set()

                                # Flush complete sentences to composer
                                while True:
                                    match = SENTENCE_END_RE.search(sentence_buffer)
                                    if not match:
                                        break
                                    end_pos = match.end()
                                    sentence = sentence_buffer[:end_pos].strip()
                                    sentence_buffer = sentence_buffer[end_pos:]
                                    if sentence:
                                        # Strip markdown for TTS
                                        clean = self._strip_markdown(sentence)
                                        if clean:
                                            if self._composer:
                                                await self._composer.enqueue(
                                                    AudioSegment(SegmentType.TTS_SENTENCE, data=clean)
                                                )
                                            else:
                                                await self._llm_out_q.put(PipelineFrame(
                                                    type=FrameType.TEXT_DELTA,
                                                    generation_id=gen_id,
                                                    data=clean
                                                ))
                                            self._spoken_sentences.append(clean)

                elif event_type == "result":
                    # CLI finished processing this message (including any tool use rounds)
                    total_ms = (time.time() - (first_text_time or time.time())) * 1000
                    self._emit_event("llm_complete", total_chars=len(full_response),
                                     sentences=len(self._spoken_sentences), latency_ms=round(total_ms, 1))
                    print("", flush=True)  # newline after streaming
                    break

        except Exception as e:
            print(f"Live session: Error reading CLI response: {e}", flush=True)

        # Log full assistant response (skip stage directions like "(quiet)")
        logged_response = self._strip_markdown(full_response)
        if logged_response and self.generation_id == gen_id:
            self._log_event("assistant", text=logged_response)

        # Flush post-tool text now that the turn is complete (use pysbd for proper splitting)
        if saw_tool_use and post_tool_buffer.strip() and self.generation_id == gen_id:
            clean = self._strip_markdown(post_tool_buffer)
            if clean:
                sentences = _sentence_segmenter.segment(clean)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        if self._composer:
                            await self._composer.enqueue(
                                AudioSegment(SegmentType.TTS_SENTENCE, data=sent)
                            )
                        else:
                            await self._llm_out_q.put(PipelineFrame(
                                type=FrameType.TEXT_DELTA,
                                generation_id=gen_id,
                                data=sent
                            ))
                        self._spoken_sentences.append(sent)

        # Flush remaining text (non-tool-use path, use pysbd for final flush)
        if not saw_tool_use and sentence_buffer.strip() and self.generation_id == gen_id:
            clean = self._strip_markdown(sentence_buffer)
            if clean:
                sentences = _sentence_segmenter.segment(clean)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        if self._composer:
                            await self._composer.enqueue(
                                AudioSegment(SegmentType.TTS_SENTENCE, data=sent)
                            )
                        else:
                            await self._llm_out_q.put(PipelineFrame(
                                type=FrameType.TEXT_DELTA,
                                generation_id=gen_id,
                                data=sent
                            ))
                        self._spoken_sentences.append(sent)

        # Track ai_asked_question: check if last sentence is a question
        if full_response.strip() and self.generation_id == gen_id:
            self._full_response_text = full_response.strip()
            last_sentence = self._spoken_sentences[-1] if self._spoken_sentences else ""
            if last_sentence.rstrip().endswith("?") or _AI_QUESTION_PATTERNS.search(last_sentence):
                self._ai_asked_question = True

        # Signal end of turn
        if self.generation_id == gen_id:
            if self._composer:
                await self._composer.enqueue_end_of_turn()
            else:
                await self._llm_out_q.put(PipelineFrame(
                    type=FrameType.END_OF_TURN,
                    generation_id=gen_id
                ))

    # ── Pipeline Stage 1: Audio Capture ────────────────────────────

    async def _audio_capture_stage(self):
        """Record audio from mic via PulseAudio Simple API and push to audio_in queue.

        Uses a daemon thread for blocking pa.read() calls, with an async loop
        for signal file polling and thread health monitoring.
        Automatically reconnects if PulseAudio errors occur.
        """
        import pasimple

        mute_signal = Path(__file__).parent / "live_mute_toggle"  # Legacy fallback
        chunks_sent = 0
        _last_bus_check_ts = time.time()
        restarts = 0
        loop = asyncio.get_event_loop()

        while self.running:
            stop_event = threading.Event()
            error_holder = [None]

            def _enqueue_audio(frame):
                try:
                    self._audio_in_q.put_nowait(frame)
                except asyncio.QueueFull:
                    pass  # Drop frame rather than block

            def record_thread():
                try:
                    with pasimple.PaSimple(
                        pasimple.PA_STREAM_RECORD,
                        pasimple.PA_SAMPLE_S16LE,
                        CHANNELS, SAMPLE_RATE,
                        app_name='push-to-talk',
                    ) as pa:
                        while not stop_event.is_set():
                            data = pa.read(CHUNK_SIZE)
                            loop.call_soon_threadsafe(
                                _enqueue_audio,
                                PipelineFrame(
                                    type=FrameType.AUDIO_RAW,
                                    generation_id=self.generation_id,
                                    data=data
                                )
                            )
                except Exception as e:
                    if not stop_event.is_set():
                        error_holder[0] = e

            thread = threading.Thread(target=record_thread, daemon=True)
            thread.start()

            if restarts > 0:
                print(f"Live session: Audio capture restarted (attempt {restarts + 1})", flush=True)
            else:
                print("Live session: Audio capture started (PulseAudio)", flush=True)

            try:
                while self.running:
                    # Check for commands via bus events
                    if self._bus:
                        cmd_events = self._bus.read_recent(last_n=10, event_type="command",
                                                           since_ts=_last_bus_check_ts)
                        for cmd_evt in cmd_events:
                            action = cmd_evt.payload.get("action", "toggle")
                            if action == "stop":
                                print("Live session: Stop requested by user", flush=True)
                                self.running = False
                                break
                            elif action == "mute":
                                self.muted = True
                                self._set_status("muted")
                                self._reset_idle_timer()
                                print("Live session: Muted by user", flush=True)
                            elif action == "unmute":
                                self.muted = False
                                self._set_status("listening")
                                self._reset_idle_timer()
                                print("Live session: Unmuted by user", flush=True)
                            else:  # toggle
                                self.muted = not self.muted
                                status = "muted" if self.muted else "listening"
                                self._set_status(status)
                                self._reset_idle_timer()
                                print(f"Live session: {'Muted' if self.muted else 'Unmuted'} by user", flush=True)
                        if cmd_events:
                            _last_bus_check_ts = cmd_events[-1].ts + 0.001
                        if not self.running:
                            break

                    # Legacy: Check for mute toggle signal file (backward compat)
                    if mute_signal.exists():
                        try:
                            command = mute_signal.read_text().strip()
                            mute_signal.unlink()
                            if command == "stop":
                                print("Live session: Stop requested by user (signal file)", flush=True)
                                self.running = False
                                break
                            elif command == "mute":
                                self.muted = True
                                self._set_status("muted")
                                self._reset_idle_timer()
                            else:
                                self.muted = not self.muted
                                status = "muted" if self.muted else "listening"
                                self._set_status(status)
                                self._reset_idle_timer()
                        except Exception:
                            pass

                    # Check for learner notifications via bus
                    if self._bus:
                        learner_events = self._bus.read_recent(last_n=5, event_type="learner_notify",
                                                                since_ts=_last_bus_check_ts)
                        for lrn_evt in learner_events:
                            summary = lrn_evt.payload.get("summary", "")
                            if summary:
                                self._notification_queue.append({"type": "learning", "summary": summary})
                                print(f"Live session: Learner notification: {summary[:80]}...", flush=True)
                                if not self.playing_audio:
                                    asyncio.ensure_future(self._flush_notifications())

                    # Legacy: Check for learner notification signal file
                    learner_notify = Path(__file__).parent / "learner_notify"
                    if learner_notify.exists():
                        try:
                            summary = learner_notify.read_text().strip()
                            learner_notify.unlink()
                            if summary:
                                self._notification_queue.append({"type": "learning", "summary": summary})
                                print(f"Live session: Learner notification (file): {summary[:80]}...", flush=True)
                                if not self.playing_audio:
                                    asyncio.ensure_future(self._flush_notifications())
                        except Exception:
                            pass

                    # Check thread health
                    if not thread.is_alive():
                        if error_holder[0]:
                            print(f"Live session: Audio capture error: {error_holder[0]}", flush=True)
                        break

                    # Track chunks for debug logging
                    qsize = self._audio_in_q.qsize()
                    if qsize > chunks_sent + 200:
                        chunks_sent = qsize
                        print(f"Live session: ~{chunks_sent} audio chunks captured", flush=True)

                    await asyncio.sleep(0.05)  # 50ms poll
            finally:
                stop_event.set()
                thread.join(timeout=2)

            if self.running:
                restarts += 1
                await asyncio.sleep(1)

        print(f"Live session: Audio capture stopped ({chunks_sent} chunks, {restarts} restarts)", flush=True)

    # ── Pipeline Stage 2: STT (Whisper local) ───────────────────

    async def _stt_stage(self):
        """Accumulate audio, detect silence, transcribe with Whisper."""
        import numpy as np

        print("STT: Using local Whisper", flush=True)
        audio_buffer = bytearray()
        SILENCE_THRESHOLD = 150  # RMS below this = silence (ambient noise ~20-100)
        SILENCE_DURATION_NORMAL = 0.8     # seconds of silence to trigger transcription
        SILENCE_DURATION_POST_BARGE = 0.4 # Faster response after interruption
        SPEECH_ENERGY_MIN = 200  # Per-chunk RMS to flag speech (well above ambient)
        SPEECH_CHUNKS_MIN = 3    # Need ~255ms of speech-level audio
        MIN_BUFFER_SECONDS = 0.5 # Minimum buffer length to transcribe
        MAX_BUFFER_SECONDS = 10  # Safety cap: force transcription after this long
        # Whisper hallucinates these on silence/noise — reject them
        HALLUCINATION_PHRASES = {
            "thank you", "thanks for watching", "thanks for listening",
            "thank you for watching", "thanks for your time",
            "goodbye", "bye", "you", "the end", "to", "so",
            "please subscribe", "like and subscribe", "i'm sorry",
            "hmm", "uh", "um", "oh",
        }

        silence_start = None
        has_speech = False  # True if enough chunks exceeded SPEECH_ENERGY_MIN
        speech_chunk_count = 0  # Number of chunks exceeding SPEECH_ENERGY_MIN
        peak_rms = 0.0  # Track peak for debugging

        def _is_hallucination(text):
            return text.lower().strip().rstrip('.!?') in HALLUCINATION_PHRASES

        try:
            while self.running:
                try:
                    frame = self._audio_in_q.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.02)

                    # Check for flush signal (key released / mic muted)
                    if self._stt_flush_event and self._stt_flush_event.is_set():
                        self._stt_flush_event.clear()
                        if len(audio_buffer) > int(SAMPLE_RATE * MIN_BUFFER_SECONDS * BYTES_PER_SAMPLE):
                            pcm_data = bytes(audio_buffer)
                            audio_buffer.clear()
                            silence_start = None
                            has_speech = False
                            speech_chunk_count = 0
                            transcript = await asyncio.get_event_loop().run_in_executor(
                                None, self._whisper_transcribe, pcm_data
                            )
                            if transcript and not _is_hallucination(transcript):
                                print(f"STT: Flushed on mute: \"{transcript[:60]}\"", flush=True)
                                await self._stt_out_q.put(PipelineFrame(
                                    type=FrameType.END_OF_UTTERANCE,
                                    generation_id=self.generation_id
                                ))
                                await self._stt_out_q.put(PipelineFrame(
                                    type=FrameType.TRANSCRIPT,
                                    generation_id=self.generation_id,
                                    data=transcript
                                ))
                                self._post_barge_in = False
                            elif transcript:
                                print(f"STT: Rejected hallucination: \"{transcript}\"", flush=True)
                        else:
                            audio_buffer.clear()
                            silence_start = None
                            has_speech = False
                            speech_chunk_count = 0
                    continue

                if frame.type != FrameType.AUDIO_RAW:
                    continue

                # Branch 1: User pressed mute — discard everything
                if self.muted:
                    audio_buffer.clear()
                    silence_start = None
                    has_speech = False
                    speech_chunk_count = 0
                    self._was_stt_gated = self._stt_gated
                    continue

                # Branch 2: STT gated (AI playback) — run VAD but don't transcribe
                if self._stt_gated:
                    audio_buffer.clear()
                    silence_start = None
                    has_speech = False
                    speech_chunk_count = 0
                    self._was_stt_gated = True

                    # Run VAD to detect barge-in speech
                    if self.barge_in_enabled and self._vad_model and self.playing_audio:
                        if time.time() < self._barge_in_cooldown_until:
                            continue
                        prob = self._run_vad(frame.data)
                        if prob > 0.5:
                            self._vad_speech_count += 1
                            # Show "listening" indicator on first detected speech chunk
                            if self._vad_speech_count == 1:
                                self._set_status("listening")
                            # ~0.5s sustained speech: 4096 bytes at 24kHz 16-bit = ~85ms per chunk
                            # 0.5s / 0.085s ~ 6 chunks
                            if self._vad_speech_count >= 6:
                                print(f"Barge-in: Sustained speech detected ({self._vad_speech_count} chunks, prob={prob:.2f})", flush=True)
                                await self._trigger_barge_in()
                        else:
                            if self._vad_speech_count > 0:
                                self._set_status("speaking")
                            self._vad_speech_count = 0
                    continue

                # Branch 3: Gated->ungated transition — reset silence tracking for clean start
                if self._was_stt_gated:
                    self._was_stt_gated = False
                    audio_buffer.clear()
                    silence_start = None
                    has_speech = False
                    speech_chunk_count = 0
                    peak_rms = 0.0
                    # Don't continue — fall through to normal audio processing below

                audio_buffer.extend(frame.data)

                # Check RMS for silence detection
                samples = np.frombuffer(frame.data, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

                # SSE: decimated audio RMS
                buf_seconds_approx = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                self._emit_audio_rms(rms=round(float(rms), 1), has_speech=has_speech,
                                     speech_chunks=speech_chunk_count, buf_seconds=round(buf_seconds_approx, 2))

                # Track whether we've seen real speech energy
                if rms > SPEECH_ENERGY_MIN:
                    speech_chunk_count += 1
                    self._last_speech_energy_time = time.time()
                    if speech_chunk_count >= SPEECH_CHUNKS_MIN:
                        has_speech = True
                if rms > peak_rms:
                    peak_rms = rms

                # Determine if we should transcribe
                should_transcribe = False
                trigger_reason = ""

                buf_bytes = len(audio_buffer)
                buf_seconds = buf_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                min_buf = int(SAMPLE_RATE * MIN_BUFFER_SECONDS * BYTES_PER_SAMPLE)

                if rms < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    else:
                        current_silence_duration = SILENCE_DURATION_POST_BARGE if self._post_barge_in else SILENCE_DURATION_NORMAL
                        if time.time() - silence_start > current_silence_duration and has_speech and buf_bytes > min_buf:
                            should_transcribe = True
                            trigger_reason = "silence"
                else:
                    silence_start = None

                # Safety cap: force transcription if buffer is too long
                # No has_speech requirement — Whisper + hallucination filter handles empty audio
                if not should_transcribe and buf_seconds > MAX_BUFFER_SECONDS:
                    should_transcribe = True
                    trigger_reason = "max_buffer"

                if should_transcribe:
                    pcm_data = bytes(audio_buffer)
                    actual_seconds = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                    print(f"STT: Transcribing {actual_seconds:.1f}s buffer (peak RMS: {peak_rms:.0f}, trigger: {trigger_reason})", flush=True)
                    self._emit_event("stt_start", trigger=trigger_reason,
                                     buf_seconds=round(actual_seconds, 2), peak_rms=round(peak_rms, 0))
                    stt_t0 = time.time()
                    audio_buffer.clear()
                    silence_start = None
                    has_speech = False
                    speech_chunk_count = 0
                    peak_rms = 0.0

                    # Run Whisper in executor to avoid blocking event loop
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None, self._whisper_transcribe, pcm_data
                    )
                    stt_ms = (time.time() - stt_t0) * 1000
                    if transcript and not _is_hallucination(transcript):
                        self._emit_event("stt_complete", text=transcript[:60],
                                         latency_ms=round(stt_ms, 1), rejected=False)
                        print(f"STT [whisper]: {transcript}", flush=True)
                        await self._stt_out_q.put(PipelineFrame(
                            type=FrameType.END_OF_UTTERANCE,
                            generation_id=self.generation_id
                        ))
                        await self._stt_out_q.put(PipelineFrame(
                            type=FrameType.TRANSCRIPT,
                            generation_id=self.generation_id,
                            data=transcript
                        ))
                        self._post_barge_in = False
                    elif transcript:
                        self._emit_event("stt_complete", text=transcript[:60],
                                         latency_ms=round(stt_ms, 1), rejected=True)
                        print(f"STT: Rejected hallucination: \"{transcript}\"", flush=True)

        except Exception as e:
            print(f"STT Whisper error: {e}", flush=True)

    # ── Pipeline Stage 3: LLM (Claude CLI) ────────────────────────

    async def _llm_stage(self):
        """Consume transcripts, send to Claude CLI, emit text deltas."""
        # Start CLI subprocess
        await self._start_cli_process()

        try:
            while self.running:
                try:
                    frame = await asyncio.wait_for(self._stt_out_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Check for pending notifications during idle
                    if self._notification_queue and not self.playing_audio:
                        await self._flush_notifications()
                    continue

                if frame.generation_id != self.generation_id:
                    continue  # Stale frame

                if frame.type == FrameType.END_OF_UTTERANCE:
                    self._set_status("thinking")
                    self._mute_mic()
                    continue

                if frame.type != FrameType.TRANSCRIPT:
                    continue

                transcript = frame.data
                t_start = time.time()
                print(f"User: {transcript}", flush=True)
                self._log_event("user", text=transcript)
                self._reset_idle_timer()

                # Start filler manager (will be cancelled when LLM responds)
                self._filler_cancel = asyncio.Event()
                filler_task = None
                if self.fillers_enabled:
                    filler_task = asyncio.create_task(
                        self._filler_manager(transcript, self._filler_cancel)
                    )

                # Build user message with ambient task context + pipeline context
                task_context = self._build_task_context()
                user_content = f"{task_context}\n\n{transcript}" if task_context else transcript

                # Inject pipeline context from event bus
                if self._bus:
                    pipeline_ctx = build_llm_context(self._bus)
                    if pipeline_ctx:
                        user_content = f"{pipeline_ctx}\n\n{user_content}"

                # Prepend barge-in annotation if the AI was interrupted
                if self._barge_in_annotation:
                    user_content = f"{self._barge_in_annotation}\n\n{user_content}"
                    self._barge_in_annotation = None

                # Drain any stale CLI output from a previous cancelled read
                await self._drain_stale_cli_output()

                # Send to CLI and read response
                await self._send_to_cli(user_content)
                self._emit_event("llm_send", text_preview=transcript[:60])
                t_sent = time.time()
                print(f"  [timing] CLI send: {t_sent - t_start:.2f}s", flush=True)

                # Wrap reader in a task so barge-in can cancel it
                self._response_reader_task = asyncio.create_task(self._read_cli_response())
                try:
                    await self._response_reader_task
                except asyncio.CancelledError:
                    print("Live session: CLI read cancelled by barge-in", flush=True)
                finally:
                    self._response_reader_task = None
                t_done = time.time()
                print(f"  [timing] CLI response: {t_done - t_sent:.2f}s, total: {t_done - t_start:.2f}s", flush=True)

                # Cancel filler and acknowledgment if still running
                if self._filler_cancel:
                    self._filler_cancel.set()
                if self._ack_cancel:
                    self._ack_cancel.set()
                if filler_task and not filler_task.done():
                    filler_task.cancel()

                self._reset_idle_timer()

        finally:
            await self._stop_cli_process()

    # ── Pipeline Stage 4: TTS (Piper local) ─────────────────────

    def _resample_22050_to_24000(self, pcm_data: bytes) -> bytes:
        """Resample 16-bit PCM from 22050Hz to 24000Hz using linear interpolation."""
        import struct
        samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
        ratio = PIPER_SAMPLE_RATE / SAMPLE_RATE
        out_len = int(len(samples) / ratio)
        out = []
        for i in range(out_len):
            src = i * ratio
            idx = int(src)
            frac = src - idx
            if idx + 1 < len(samples):
                val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            else:
                val = samples[idx] if idx < len(samples) else 0
            out.append(int(val))
        return struct.pack(f'<{len(out)}h', *out)

    async def _tts_stage(self):
        """Consume text deltas, convert to PCM audio via Piper TTS, push to playback."""
        while self.running:
            try:
                frame = await asyncio.wait_for(self._llm_out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if frame.generation_id != self.generation_id:
                continue

            if frame.type == FrameType.END_OF_TURN:
                await self._audio_out_q.put(PipelineFrame(
                    type=FrameType.END_OF_TURN,
                    generation_id=frame.generation_id
                ))
                continue

            if frame.type != FrameType.TEXT_DELTA:
                continue

            text = frame.data
            if not text.strip():
                continue

            gen_id = frame.generation_id

            try:
                tts_start = time.time()
                process = await asyncio.create_subprocess_exec(
                    PIPER_CMD, '--model', PIPER_MODEL, '--output-raw',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )

                # Feed text and close stdin to trigger synthesis
                process.stdin.write(text.encode())
                process.stdin.close()

                # Stream chunks as they arrive
                first_chunk = True
                while self.generation_id == gen_id:
                    chunk = await process.stdout.read(4096)
                    if not chunk:
                        break
                    if first_chunk:
                        print(f"  [timing] TTS first byte: {time.time() - tts_start:.2f}s for \"{text[:40]}...\"", flush=True)
                        first_chunk = False
                    # Resample 22050 → 24000
                    resampled = self._resample_22050_to_24000(chunk)
                    await self._audio_out_q.put(PipelineFrame(
                        type=FrameType.TTS_AUDIO,
                        generation_id=gen_id,
                        data=resampled
                    ))

                await process.wait()

                # Mark sentence boundary for playback stage to count
                if self.generation_id == gen_id:
                    await self._audio_out_q.put(PipelineFrame(
                        type=FrameType.CONTROL,
                        generation_id=gen_id,
                        data="sentence_done"
                    ))
            except Exception as e:
                print(f"Piper TTS error: {e}", flush=True)

    # ── Pipeline Stage 5: Playback (PyAudio) ──────────────────────

    async def _playback_stage(self):
        """Consume TTS audio frames and play through PyAudio."""
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=1024,
        )

        print("Live session: Playback started", flush=True)

        try:
            while self.running:
                try:
                    frame = await asyncio.wait_for(self._audio_out_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if frame.generation_id != self.generation_id:
                    continue

                # Count completed sentences via sentinel from composer (or legacy TTS stage)
                if frame.type == FrameType.SENTENCE_DONE or (frame.type == FrameType.CONTROL and frame.data == "sentence_done"):
                    self._played_sentence_count += 1
                    continue

                if frame.type == FrameType.END_OF_TURN:
                    # Audio done — unmute mic after brief cooldown
                    self.audio_done_time = time.time()

                    async def delayed_unmute():
                        await asyncio.sleep(0.5)
                        if self.playing_audio:
                            self.playing_audio = False
                            self._stt_gated = False
                            self._unmute_mic()
                            if not self.muted:
                                self._set_status("listening")
                            else:
                                self._set_status("muted")
                            print("Live session: Mic unmuted", flush=True)

                            # Flush pending notifications
                            if self._notification_queue:
                                await asyncio.sleep(1.0)
                                if not self.playing_audio:
                                    await self._flush_notifications()

                    # Cancel previous unmute if pending
                    if self._unmute_task and not self._unmute_task.done():
                        self._unmute_task.cancel()
                    self._unmute_task = asyncio.create_task(delayed_unmute())
                    continue

                if frame.type in (FrameType.TTS_AUDIO, FrameType.FILLER):
                    # Cancel any pending unmute — new audio arrived
                    if self._unmute_task and not self._unmute_task.done():
                        self._unmute_task.cancel()

                    # First audio chunk — gate STT and set status
                    # Mic stays physically live so VAD can detect barge-in speech
                    # Skip re-gating for post-barge-in trail clip (user is already talking)
                    if not self.playing_audio:
                        self.playing_audio = True
                        if not self._post_barge_in:
                            self._set_status("speaking")
                            self._stt_gated = True

                    self._reset_idle_timer()

                    # Write audio to PyAudio stream (runs in executor to avoid blocking)
                    audio_data = frame.data
                    await asyncio.get_event_loop().run_in_executor(
                        None, stream.write, audio_data
                    )
                    if frame.type == FrameType.TTS_AUDIO:
                        self._bytes_played += len(audio_data)

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print("Live session: Playback stopped", flush=True)

    # ── Interrupt ──────────────────────────────────────────────────

    def set_muted(self, muted: bool):
        """Thread-safe mute/unmute from UI thread."""
        if not self.running:
            return  # Pipeline not ready yet
        self.muted = muted
        if muted:
            self._set_status("muted")
            # Signal STT to flush accumulated transcripts (thread-safe)
            if self._stt_flush_event:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._stt_flush_event.set)
                else:
                    self._stt_flush_event.set()
            print("Live session: Muted (key released)", flush=True)
        else:
            self._set_status("listening")
            self._reset_idle_timer()
            print("Live session: Unmuted (key held)", flush=True)

    def request_interrupt(self):
        """Thread-safe way to request an interrupt (from UI thread)."""
        self._interrupt_requested = True

    async def _check_interrupt(self):
        """Check and handle pending interrupt request."""
        if self._interrupt_requested:
            self._interrupt_requested = False
            print("Live session: Interrupting response", flush=True)
            self.generation_id += 1
            self.playing_audio = False
            self._stt_gated = False
            self._vad_speech_count = 0
            # Reset composer and drain audio queue
            if self._composer:
                self._composer.reset()
            while not self._audio_out_q.empty():
                try:
                    self._audio_out_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._unmute_mic()
            self._set_status("listening")

    async def _trigger_barge_in(self):
        """Barge-in detected: fade out playback, play trailing filler, reset state."""
        import numpy as np

        print("Barge-in: Triggering interruption", flush=True)

        # 1. Cancel pending delayed_unmute task
        if self._unmute_task and not self._unmute_task.done():
            self._unmute_task.cancel()

        # 2. Increment generation_id to discard all queued frames
        self.generation_id += 1

        # 2b. Cancel the CLI response reader task so _llm_stage unblocks
        if self._response_reader_task and not self._response_reader_task.done():
            self._response_reader_task.cancel()

        # 3. Pause and reset composer (returns unplayed segments for annotation)
        held_segments = []
        if self._composer:
            held_segments = self._composer.pause()
            self._composer.reset()

        # 4. Drain audio_out queue (stale frames from before composer pause)
        while not self._audio_out_q.empty():
            try:
                self._audio_out_q.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 5. Cancel filler and acknowledgment if running
        if self._filler_cancel and not self._filler_cancel.is_set():
            self._filler_cancel.set()
        if self._ack_cancel and not self._ack_cancel.is_set():
            self._ack_cancel.set()

        # 6. Set cooldown (1.5s)
        self._barge_in_cooldown_until = time.time() + 1.5

        # 7. Reset VAD speech counter and state
        self._vad_speech_count = 0
        self._reset_vad_state()

        # 8. Transition state — ungating triggers _was_stt_gated reset in STT stage
        self.playing_audio = False
        self._stt_gated = False

        # 9. Unmute mic (undo the _llm_stage mute from thinking phase)
        self._unmute_mic()

        # 10. Play a trailing acknowledgment filler clip for naturalness
        # (bypasses composer intentionally -- one-shot trail directly to audio_out)
        if self.fillers_enabled:
            clip = self._pick_filler("acknowledgment")
            if clip:
                gen_id = self.generation_id
                # Only queue the first ~150ms worth of clip for a brief trail
                trail_bytes = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.15)
                trail = clip[:trail_bytes] if len(clip) > trail_bytes else clip
                # Apply fade-out to the trail
                samples = np.frombuffer(trail, dtype=np.int16).copy()
                if len(samples) > 0:
                    fade = np.linspace(0.8, 0.0, len(samples))
                    samples = (samples.astype(np.float64) * fade).astype(np.int16)
                    await self._audio_out_q.put(PipelineFrame(
                        type=FrameType.FILLER,
                        generation_id=gen_id,
                        data=samples.tobytes()
                    ))
                    # END_OF_TURN so playback stage resets playing_audio
                    await self._audio_out_q.put(PipelineFrame(
                        type=FrameType.END_OF_TURN,
                        generation_id=gen_id,
                    ))

        # 11. Build interruption annotation for next turn
        # Use held_segments from composer for unspoken count, combined with sentence tracking
        self._was_interrupted = True
        unspoken_from_composer = [
            s for s in held_segments
            if s.type == SegmentType.TTS_SENTENCE and isinstance(s.data, str)
        ]
        spoken = self._spoken_sentences[:self._played_sentence_count]
        # Combine tracked unspoken with composer held segments for complete picture
        unspoken = self._spoken_sentences[self._played_sentence_count:]
        if unspoken_from_composer and not unspoken:
            unspoken = [s.data for s in unspoken_from_composer]
        spoken_text = " ".join(spoken) if spoken else "(nothing)"
        unspoken_text = " ".join(unspoken) if unspoken else "(nothing)"

        self._barge_in_annotation = (
            f"[The user interrupted you. "
            f"They heard up to: \"{spoken_text}\". "
            f"Your unspoken response was: \"{unspoken_text}\". "
            f"Adjust based on what the user says next.]"
        )
        print(f"Barge-in: Annotation ({self._played_sentence_count}/{len(self._spoken_sentences)} sentences spoken)", flush=True)

        # 11. Set shortened post-barge-in silence threshold
        self._post_barge_in = True

        # 12. Set status to listening
        self._set_status("listening")

        # Log barge-in event (bus handles both persistence + SSE)
        self._log_event("barge_in",
            spoken_sentences=self._played_sentence_count,
            total_sentences=len(self._spoken_sentences),
            cooldown_until=self._barge_in_cooldown_until
        )
        print("Barge-in: Interruption complete, listening for user", flush=True)

    # ── Main loop ──────────────────────────────────────────────────

    async def run(self):
        """Run the live session — 5 concurrent pipeline stages."""
        self._set_status("thinking")
        self.running = True
        self._started_at = time.time()
        self.muted = False  # Start unmuted — key is held when session launches

        # Clean up stale toggle signal
        stale_signal = Path(__file__).parent / "live_mute_toggle"
        if stale_signal.exists():
            try:
                stale_signal.unlink()
            except Exception:
                pass

        # Set up session logging via event bus
        session_id = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path("~/Audio/push-to-talk/sessions").expanduser() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_log_path = session_dir / "events.jsonl"

        self._bus = EventBus(session_dir, "live_session", session_id)
        self._bus.open()

        # Wire SSE dashboard to bus callbacks
        if self._sse_dashboard:
            self._bus.on("*", self._sse_broadcast)

        self._log_event("session_start", model=self.model, session_id=session_id)

        # Write active session pointer (points to session dir, not specific file)
        active_file = Path("~/.local/share/push-to-talk/active_session").expanduser()
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(str(session_dir))
        self._spawn_learner()
        self._spawn_clip_factory()
        self._spawn_classifier()
        self._ensure_seed_library()
        self._load_response_library()

        # Load VAD model once at startup (used by STT stage for barge-in)
        if self.barge_in_enabled:
            self._load_vad_model()

        # Create pipeline queues
        self._audio_in_q = asyncio.Queue(maxsize=100)
        self._stt_out_q = asyncio.Queue(maxsize=50)
        self._llm_out_q = asyncio.Queue(maxsize=50)  # kept for fallback compatibility
        self._audio_out_q = asyncio.Queue(maxsize=200)
        self._stt_flush_event = asyncio.Event()
        self._loop = asyncio.get_event_loop()

        # Create StreamComposer -- replaces _tts_stage for TTS generation
        async def _tts_callback(text: str) -> bytes | None:
            return await self._tts_to_pcm(text)

        self._composer = StreamComposer(
            self._audio_out_q,
            _tts_callback,
            lambda: self.generation_id,
            on_event=lambda etype, **d: self._bus.emit(etype, gen=self.generation_id, **d) if self._bus else None,
        )

        self._set_status("listening")
        self._reset_idle_timer()
        print("Live session: Pipeline started (with StreamComposer)", flush=True)

        try:
            # Run interrupt checker as background task
            async def interrupt_loop():
                while self.running:
                    await self._check_interrupt()
                    await asyncio.sleep(0.05)

            stages = [
                self._audio_capture_stage(),
                self._stt_stage(),
                self._llm_stage(),
                self._composer.run(),  # Composer replaces _tts_stage
                self._playback_stage(),
                interrupt_loop(),
                self._sse_server_stage(),
            ]

            await asyncio.gather(*stages, return_exceptions=True)
        except Exception as e:
            print(f"Live session error: {e}", flush=True)
            self._set_status("error")
        finally:
            self._log_event("session_end",
                            duration_s=round(time.time() - self._started_at, 1))
            self.running = False
            if self._composer:
                self._composer.stop()
            if self._bus:
                self._bus.close()
                self._bus = None
            self._cancel_idle_timer()
            self._unmute_mic()
            self._set_status("idle")

            # Clean up active session pointer
            try:
                active_file = Path("~/.local/share/push-to-talk/active_session").expanduser()
                if active_file.exists():
                    active_file.unlink(missing_ok=True)
            except Exception:
                pass

            # Clean up background processes
            for name, proc in [("learner", self._learner_process),
                               ("clip factory", self._clip_factory_process)]:
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    print(f"Live session: {name} stopped", flush=True)

            # Kill classifier daemon
            if self._classifier_process:
                try:
                    self._classifier_process.terminate()
                    self._classifier_process.wait(timeout=3)
                except Exception:
                    try:
                        self._classifier_process.kill()
                    except Exception:
                        pass
                print("Live session: classifier stopped", flush=True)
            if self._classifier_socket_path and os.path.exists(self._classifier_socket_path):
                try:
                    os.unlink(self._classifier_socket_path)
                except Exception:
                    pass

            # Kill seed generation if still running
            if self._seed_process and self._seed_process.poll() is None:
                try:
                    self._seed_process.terminate()
                except Exception:
                    pass

            # Save response library usage data
            if self._response_library.is_loaded():
                try:
                    self._response_library.save()
                except Exception as e:
                    print(f"Live session: Failed to save response library: {e}", flush=True)

            print("Live session: Pipeline stopped", flush=True)

    def stop(self):
        """Stop the session gracefully."""
        self.running = False
        if self._composer:
            self._composer.stop()
        self._cancel_idle_timer()
