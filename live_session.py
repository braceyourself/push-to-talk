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
import time
import re
import wave
import random
import tempfile
import shutil
from collections import deque
from pathlib import Path

from pipeline_frames import PipelineFrame, FrameType
from task_manager import TaskManager, ClaudeTask, TaskStatus

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4096  # bytes per read from pw-record
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# TTS sentence buffer — accumulate text until a sentence boundary before sending to TTS
SENTENCE_END_RE = re.compile(r'[.!?]\s|[.!?]$|\n')

# Supported OpenAI TTS voices
TTS_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}

# Piper TTS configuration
PIPER_CMD = str(Path.home() / ".local" / "share" / "push-to-talk" / "venv" / "bin" / "piper")
PIPER_MODEL = str(Path.home() / ".local" / "share" / "push-to-talk" / "piper-voices" / "en_US-lessac-medium.onnx")
PIPER_SAMPLE_RATE = 22050


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
                 fillers_enabled=True, barge_in_enabled=True, whisper_model=None):
        self.openai_api_key = openai_api_key
        self.deepgram_api_key = deepgram_api_key
        self.whisper_model = whisper_model
        self.voice = voice if voice in TTS_VOICES else "ash"
        self.model = model
        self.on_status = on_status or (lambda s: None)

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
        self._idle_timeout = 120  # seconds

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
        if self.fillers_enabled:
            self._load_filler_clips()

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
        """Append a JSONL event to the session log file."""
        if not self._session_log_path:
            return
        entry = {"ts": time.time(), "type": event_type, **kwargs}
        try:
            with open(self._session_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Live session: Log write error: {e}", flush=True)

    def _spawn_learner(self):
        """Spawn the background learner daemon that watches the conversation log."""
        learner_script = Path(__file__).parent / "learner.py"
        if not learner_script.exists():
            print("Live session: learner.py not found, skipping", flush=True)
            return
        cmd = [sys.executable, str(learner_script), str(self._session_log_path)]
        try:
            self._learner_process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"Live session: Learner spawned (PID {self._learner_process.pid})", flush=True)
        except Exception as e:
            print(f"Live session: Failed to spawn learner: {e}", flush=True)

    def _spawn_clip_factory(self):
        """Spawn the clip factory to top up the non-verbal filler pool."""
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

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    # ── Notifications ──────────────────────────────────────────────

    async def _on_task_complete(self, task):
        """Handle task completion — queue notification for delivery."""
        if not self.running:
            return
        duration = (task.completed_at or time.time()) - (task.started_at or task.created_at)
        output_tail = '\n'.join(list(task.output_lines)[-20:])
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
        """Handle task failure — queue notification for delivery."""
        if not self.running:
            return
        duration = (task.completed_at or time.time()) - (task.started_at or task.created_at)
        output_tail = '\n'.join(list(task.output_lines)[-20:])
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

    def _set_status(self, status):
        self.on_status(status)

    def _reset_idle_timer(self):
        self._cancel_idle_timer()
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

    # ── Mic mute/unmute ────────────────────────────────────────────

    def _mute_mic(self):
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '1'], capture_output=True)

    def _unmute_mic(self):
        subprocess.run(['pactl', 'set-source-mute', '@DEFAULT_SOURCE@', '0'], capture_output=True)

    # ── Filler System ────────────────────────────────────────────────

    def _load_filler_clips(self):
        """Load non-verbal filler WAV files as raw PCM bytes."""
        filler_dir = Path(__file__).parent / "audio" / "fillers" / "nonverbal"
        if not filler_dir.exists():
            print("Live session: No nonverbal filler directory found, fillers disabled", flush=True)
            self.fillers_enabled = False
            return

        clips = []
        for wav_path in sorted(filler_dir.glob("*.wav")):
            try:
                with wave.open(str(wav_path), 'rb') as wf:
                    pcm = wf.readframes(wf.getnframes())
                    rate = wf.getframerate()
                if rate != SAMPLE_RATE:
                    pcm = self._resample_22050_to_24000(pcm)
                clips.append(pcm)
            except Exception as e:
                print(f"Live session: Error loading filler {wav_path}: {e}", flush=True)

        if clips:
            self._filler_clips["nonverbal"] = clips
            self._last_filler["nonverbal"] = -1
            print(f"Live session: Loaded {len(clips)} non-verbal filler clips", flush=True)
        else:
            print("Live session: No filler clips loaded, fillers disabled", flush=True)
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
        """Play a non-verbal filler clip while waiting for LLM response."""
        # Stage 1: Wait 300ms gate — skip filler if LLM responds fast
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=0.3)
            return
        except asyncio.TimeoutError:
            pass

        if cancel_event.is_set():
            return

        # Play a non-verbal clip
        clip = self._pick_filler("nonverbal")
        if clip:
            await self._play_filler_audio(clip, cancel_event)

        # Stage 2: If still waiting after 4s, play another clip
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=4.0)
            return
        except asyncio.TimeoutError:
            pass

        if not cancel_event.is_set():
            clip = self._pick_filler("nonverbal")
            if clip:
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
            # Initialize VAD state: h and c tensors (2, 1, 64)
            import numpy as np
            self._vad_state = {
                'h': np.zeros((2, 1, 64), dtype=np.float32),
                'c': np.zeros((2, 1, 64), dtype=np.float32),
                'sr': np.array([SAMPLE_RATE], dtype=np.int64)
            }
            print("Live session: Silero VAD loaded", flush=True)
        except Exception as e:
            print(f"Live session: Failed to load VAD model: {e}", flush=True)
            self.barge_in_enabled = False

    def _run_vad(self, audio_bytes: bytes) -> float:
        """Run VAD inference on audio chunk, return speech probability."""
        if not self._vad_model:
            return 0.0

        import numpy as np

        # Convert bytes to float32 samples
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Silero VAD expects 512 samples at 16kHz, or 768 at 24kHz
        # Resample to 16kHz if needed by taking every 1.5th sample (simple decimation)
        if SAMPLE_RATE == 24000:
            # Approximate resample 24k->16k by taking 2 of every 3
            indices = np.arange(0, len(samples), 1.5).astype(int)
            indices = indices[indices < len(samples)]
            samples = samples[indices]

        # Pad or trim to 512 samples
        if len(samples) < 512:
            samples = np.pad(samples, (0, 512 - len(samples)))
        else:
            samples = samples[:512]

        input_data = samples.reshape(1, -1)

        try:
            ort_inputs = {
                'input': input_data,
                'h': self._vad_state['h'],
                'c': self._vad_state['c'],
                'sr': self._vad_state['sr']
            }
            ort_outputs = self._vad_model.run(None, ort_inputs)
            prob = ort_outputs[0].item()
            self._vad_state['h'] = ort_outputs[1]
            self._vad_state['c'] = ort_outputs[2]
            return prob
        except Exception:
            return 0.0

    def _reset_vad_state(self):
        """Reset VAD hidden state to avoid contamination between segments."""
        if self._vad_state:
            import numpy as np
            self._vad_state['h'] = np.zeros((2, 1, 64), dtype=np.float32)
            self._vad_state['c'] = np.zeros((2, 1, 64), dtype=np.float32)

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
        """Transcribe PCM audio using Whisper (blocking, run in executor)."""
        try:
            import numpy as np
            import wave as wave_mod

            if not self.whisper_model:
                import whisper
                self.whisper_model = whisper.load_model("small")

            # Write to temp WAV file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                with wave_mod.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(BYTES_PER_SAMPLE)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm_data)

            result = self.whisper_model.transcribe(
                tmp_path, language="en",
                condition_on_previous_text=False,  # Prevent hallucination loops
            )

            # Filter out segments where Whisper thinks there's no speech
            # (catches throat clearing, coughs, background noise)
            segments = result.get("segments", [])
            if segments:
                kept = [s["text"] for s in segments if s.get("no_speech_prob", 0) < 0.6]
                text = "".join(kept).strip()
                if not kept and segments:
                    rejected_text = "".join(s["text"] for s in segments).strip()
                    probs = [f'{s.get("no_speech_prob", 0):.2f}' for s in segments]
                    print(f"STT: Rejected non-speech (probs={probs}): \"{rejected_text}\"", flush=True)
            else:
                text = result.get("text", "").strip()

            os.unlink(tmp_path)

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

    async def _read_cli_response(self):
        """Read streaming events from CLI until turn is complete. Route text to TTS."""
        if not self._cli_process or not self._cli_process.stdout:
            return

        gen_id = self.generation_id
        sentence_buffer = ""
        full_response = ""
        first_text_time = None
        saw_tool_use = False  # Track if model is making tool calls

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
                            if not saw_tool_use:
                                saw_tool_use = True
                                # First tool call — drain any text already
                                # sent to TTS so we don't speak pre-tool narration
                                drained = 0
                                for q in (self._llm_out_q, self._audio_out_q):
                                    while not q.empty():
                                        try:
                                            f = q.get_nowait()
                                            if f.type == FrameType.END_OF_TURN:
                                                await q.put(f)
                                                break
                                            drained += 1
                                        except asyncio.QueueEmpty:
                                            break
                                if drained:
                                    print(f"  [tool] Drained {drained} pre-tool frames", flush=True)
                                # Play a nonverbal filler clip
                                if self.fillers_enabled:
                                    clip = self._pick_filler("nonverbal")
                                    if clip:
                                        asyncio.create_task(
                                            self._play_filler_audio(clip, asyncio.Event())
                                        )

                            # Discard any accumulated text (pre-tool or inter-tool narration)
                            if sentence_buffer.strip():
                                print(f"  [tool] Suppressed text: \"{sentence_buffer.strip()[:60]}\"", flush=True)
                                sentence_buffer = ""
                            # Also discard inter-tool post_tool_buffer
                            if post_tool_buffer.strip():
                                print(f"  [tool] Suppressed inter-tool text: \"{post_tool_buffer.strip()[:60]}\"", flush=True)
                                post_tool_buffer = ""
                            self._set_status("tool_use")

                    elif inner_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if first_text_time is None:
                                first_text_time = time.time()
                                print(f"\n  [timing] first token: {first_text_time - (self._last_send_time or first_text_time):.2f}s", flush=True)
                            full_response += text
                            print(text, end="", flush=True)

                            if saw_tool_use:
                                # After tool use: hold text, don't send to TTS yet
                                post_tool_buffer += text
                            else:
                                # Normal path: stream text to TTS
                                sentence_buffer += text

                                # Cancel filler on first text from LLM
                                if self._filler_cancel and not self._filler_cancel.is_set():
                                    self._filler_cancel.set()

                                # Flush complete sentences to TTS
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
                                            await self._llm_out_q.put(PipelineFrame(
                                                type=FrameType.TEXT_DELTA,
                                                generation_id=gen_id,
                                                data=clean
                                            ))
                                            self._spoken_sentences.append(clean)

                elif event_type == "result":
                    # CLI finished processing this message (including any tool use rounds)
                    print("", flush=True)  # newline after streaming
                    break

        except Exception as e:
            print(f"Live session: Error reading CLI response: {e}", flush=True)

        # Log full assistant response
        if full_response.strip() and self.generation_id == gen_id:
            self._log_event("assistant", text=full_response.strip())

        # Flush post-tool text now that the turn is complete
        if saw_tool_use and post_tool_buffer.strip() and self.generation_id == gen_id:
            clean = self._strip_markdown(post_tool_buffer)
            if clean:
                await self._llm_out_q.put(PipelineFrame(
                    type=FrameType.TEXT_DELTA,
                    generation_id=gen_id,
                    data=clean
                ))
                self._spoken_sentences.append(clean)

        # Flush remaining text (non-tool-use path)
        if not saw_tool_use and sentence_buffer.strip() and self.generation_id == gen_id:
            clean = self._strip_markdown(sentence_buffer)
            if clean:
                await self._llm_out_q.put(PipelineFrame(
                    type=FrameType.TEXT_DELTA,
                    generation_id=gen_id,
                    data=clean
                ))
                self._spoken_sentences.append(clean)

        # Signal end of turn
        if self.generation_id == gen_id:
            await self._llm_out_q.put(PipelineFrame(
                type=FrameType.END_OF_TURN,
                generation_id=gen_id
            ))

    # ── Pipeline Stage 1: Audio Capture ────────────────────────────

    async def _audio_capture_stage(self):
        """Record audio from mic via pw-record and push to audio_in queue."""
        process = await asyncio.create_subprocess_exec(
            'pw-record', '--format', 's16', '--rate', str(SAMPLE_RATE), '--channels', str(CHANNELS), '-',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        print("Live session: Audio capture started", flush=True)

        mute_signal = Path(__file__).parent / "live_mute_toggle"
        chunks_sent = 0

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
                            self.muted = not self.muted
                            status = "muted" if self.muted else "listening"
                            self._set_status(status)
                            self._reset_idle_timer()
                            print(f"Live session: {'Muted' if self.muted else 'Unmuted'} by user", flush=True)
                    except Exception:
                        pass

                # Check for learner notification
                learner_notify = Path(__file__).parent / "learner_notify"
                if learner_notify.exists():
                    try:
                        summary = learner_notify.read_text().strip()
                        learner_notify.unlink()
                        if summary:
                            self._notification_queue.append({"type": "learning", "summary": summary})
                            print(f"Live session: Learner notification: {summary[:80]}...", flush=True)
                            if not self.playing_audio:
                                asyncio.ensure_future(self._flush_notifications())
                    except Exception:
                        pass

                audio_data = await process.stdout.read(CHUNK_SIZE)
                if not audio_data:
                    await asyncio.sleep(0.01)
                    continue

                # Always send audio to the STT stage. During playback, the mic
                # stays live (for VAD barge-in detection) but STT is gated —
                # audio is consumed for VAD but not accumulated for transcription.
                await self._audio_in_q.put(PipelineFrame(
                    type=FrameType.AUDIO_RAW,
                    generation_id=self.generation_id,
                    data=audio_data
                ))
                chunks_sent += 1
                if chunks_sent % 200 == 0:
                    print(f"Live session: Sent {chunks_sent} audio chunks to STT", flush=True)

        finally:
            process.terminate()
            await process.wait()
            print(f"Live session: Audio capture stopped ({chunks_sent} chunks)", flush=True)

    # ── Pipeline Stage 2: STT (Whisper local) ───────────────────

    async def _stt_stage(self):
        """Accumulate audio, detect silence, transcribe with Whisper."""
        import numpy as np

        print("STT: Using local Whisper", flush=True)
        audio_buffer = bytearray()
        SILENCE_THRESHOLD = 18   # RMS below this = silence (ambient ~10)
        SILENCE_DURATION = 0.8   # seconds of silence to trigger transcription
        SPEECH_ENERGY_MIN = 25   # Per-chunk RMS to flag speech (just above ambient)
        SPEECH_CHUNKS_MIN = 5    # Need ~425ms of speech-level audio
        MIN_BUFFER_SECONDS = 0.5 # Minimum buffer length to transcribe
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
                            # ~0.5s sustained speech: 4096 bytes at 24kHz 16-bit = ~85ms per chunk
                            # 0.5s / 0.085s ~ 6 chunks
                            if self._vad_speech_count >= 6:
                                print(f"Barge-in: Sustained speech detected ({self._vad_speech_count} chunks, prob={prob:.2f})", flush=True)
                                await self._trigger_barge_in()
                        else:
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

                # Track whether we've seen real speech energy
                if rms > SPEECH_ENERGY_MIN:
                    speech_chunk_count += 1
                    if speech_chunk_count >= SPEECH_CHUNKS_MIN:
                        has_speech = True
                if rms > peak_rms:
                    peak_rms = rms

                if rms < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION and has_speech and len(audio_buffer) > int(SAMPLE_RATE * MIN_BUFFER_SECONDS * BYTES_PER_SAMPLE):
                        # Silence detected after speech — transcribe
                        pcm_data = bytes(audio_buffer)
                        buf_seconds = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        print(f"STT: Transcribing {buf_seconds:.1f}s buffer (peak RMS: {peak_rms:.0f})", flush=True)
                        audio_buffer.clear()
                        silence_start = None
                        has_speech = False
                        speech_chunk_count = 0
                        peak_rms = 0.0

                        # Run Whisper in executor to avoid blocking event loop
                        transcript = await asyncio.get_event_loop().run_in_executor(
                            None, self._whisper_transcribe, pcm_data
                        )
                        if transcript and not _is_hallucination(transcript):
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
                        elif transcript:
                            print(f"STT: Rejected hallucination: \"{transcript}\"", flush=True)
                else:
                    silence_start = None

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

                # Build user message with ambient task context
                task_context = self._build_task_context()
                user_content = f"{task_context}\n\n{transcript}" if task_context else transcript

                # Prepend barge-in annotation if the AI was interrupted
                if self._barge_in_annotation:
                    user_content = f"{self._barge_in_annotation}\n\n{user_content}"
                    self._barge_in_annotation = None

                # Send to CLI and read response
                await self._send_to_cli(user_content)
                t_sent = time.time()
                print(f"  [timing] CLI send: {t_sent - t_start:.2f}s", flush=True)
                await self._read_cli_response()
                t_done = time.time()
                print(f"  [timing] CLI response: {t_done - t_sent:.2f}s, total: {t_done - t_start:.2f}s", flush=True)

                # Cancel filler if still running
                if self._filler_cancel:
                    self._filler_cancel.set()
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

                # Count completed sentences via sentinel from TTS stage
                if frame.type == FrameType.CONTROL and frame.data == "sentence_done":
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
                    if not self.playing_audio:
                        self.playing_audio = True
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
            # Drain pending audio and LLM queues
            for q in (self._audio_out_q, self._llm_out_q):
                while not q.empty():
                    try:
                        q.get_nowait()
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

        # 3. Drain audio_out and llm_out queues (stale frames)
        for q in (self._audio_out_q, self._llm_out_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # 4. Cancel filler if running
        if self._filler_cancel and not self._filler_cancel.is_set():
            self._filler_cancel.set()

        # 5. Set cooldown (1.5s)
        self._barge_in_cooldown_until = time.time() + 1.5

        # 6. Reset VAD speech counter and state
        self._vad_speech_count = 0
        self._reset_vad_state()

        # 7. Transition state — ungating triggers _was_stt_gated reset in STT stage
        self.playing_audio = False
        self._stt_gated = False

        # 8. Unmute mic (undo the _llm_stage mute from thinking phase)
        self._unmute_mic()

        # 9. Play a trailing non-verbal filler clip for naturalness
        if self.fillers_enabled:
            clip = self._pick_filler("nonverbal")
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

        # 10. Build interruption annotation for next turn
        self._was_interrupted = True
        spoken = self._spoken_sentences[:self._played_sentence_count]
        unspoken = self._spoken_sentences[self._played_sentence_count:]
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

        # Log it
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

        # Set up session logging
        session_id = time.strftime("%Y%m%d_%H%M%S")
        session_dir = Path("~/Audio/push-to-talk/sessions").expanduser() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_log_path = session_dir / "conversation.jsonl"
        self._log_event("session_start", model=self.model, session_id=session_id)

        # Write active session pointer and spawn learner
        active_file = Path("~/.local/share/push-to-talk/active_session").expanduser()
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(str(self._session_log_path))
        self._spawn_learner()
        self._spawn_clip_factory()

        # Load VAD model once at startup (used by STT stage for barge-in)
        if self.barge_in_enabled:
            self._load_vad_model()

        # Create pipeline queues
        self._audio_in_q = asyncio.Queue(maxsize=100)
        self._stt_out_q = asyncio.Queue(maxsize=50)
        self._llm_out_q = asyncio.Queue(maxsize=50)
        self._audio_out_q = asyncio.Queue(maxsize=200)
        self._stt_flush_event = asyncio.Event()
        self._loop = asyncio.get_event_loop()

        self._set_status("listening")
        self._reset_idle_timer()
        print("Live session: Pipeline started", flush=True)

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
                self._tts_stage(),
                self._playback_stage(),
                interrupt_loop(),
            ]

            await asyncio.gather(*stages, return_exceptions=True)
        except Exception as e:
            print(f"Live session error: {e}", flush=True)
            self._set_status("error")
        finally:
            self._log_event("session_end")
            self.running = False
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

            print("Live session: Pipeline stopped", flush=True)

    def stop(self):
        """Stop the session gracefully."""
        self.running = False
        self._cancel_idle_timer()
