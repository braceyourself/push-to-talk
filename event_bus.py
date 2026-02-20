"""
Unified JSONL event bus for push-to-talk pipeline.

All processes (live_session, learner, indicator, clip_factory) write to a single
JSONL file per session. Consumers tail it or read recent entries. In-process
callbacks provide real-time delivery for SSE dashboard.

Writer atomicity: POSIX O_APPEND guarantees atomic writes under PIPE_BUF (4096 bytes).
Each JSON line + newline stays under that limit.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# POSIX PIPE_BUF — lines must stay under this for atomic multi-writer appends
_PIPE_BUF = 4096

# Event types excluded from build_llm_context (high-frequency, no replay value)
_LLM_CONTEXT_EXCLUDE = {"audio_rms", "queue_depths", "llm_text_delta", "snapshot"}

# Max age for build_llm_context (5 minutes)
_LLM_CONTEXT_MAX_AGE = 300


class EventType(str, Enum):
    """All event types in the bus catalog."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    STATUS = "status"
    AUDIO_RMS = "audio_rms"
    QUEUE_DEPTHS = "queue_depths"
    STT_START = "stt_start"
    STT_COMPLETE = "stt_complete"
    USER = "user"
    ASSISTANT = "assistant"
    LLM_SEND = "llm_send"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_TEXT_DELTA = "llm_text_delta"
    LLM_TOOL_USE = "llm_tool_use"
    LLM_COMPLETE = "llm_complete"
    TTS_START = "tts_start"
    TTS_COMPLETE = "tts_complete"
    FILLER_PLAYED = "filler_played"
    BARGE_IN = "barge_in"
    COMMAND = "command"
    LEARNER_NOTIFY = "learner_notify"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    ERROR = "error"
    SNAPSHOT = "snapshot"


# Core fields that are not part of the payload
_CORE_FIELDS = {"ts", "src", "type", "gen", "sid"}


@dataclass
class BusEvent:
    """A single event on the bus."""
    ts: float
    src: str
    type: str
    gen: int
    sid: str
    payload: dict = field(default_factory=dict)

    def __init__(self, ts: float, src: str, type: str, gen: int, sid: str, **kwargs):
        self.ts = ts
        self.src = src
        self.type = type
        self.gen = gen
        self.sid = sid
        self.payload = kwargs

    def to_json_line(self) -> str:
        """Serialize to a single JSON line with trailing newline.

        Truncates payload if the line would exceed PIPE_BUF.
        """
        data = {"ts": self.ts, "src": self.src, "type": self.type,
                "gen": self.gen, "sid": self.sid, **self.payload}
        line = json.dumps(data, separators=(',', ':')) + "\n"

        if len(line.encode()) > _PIPE_BUF:
            # Truncate large string values in payload
            truncated = dict(data)
            for key, val in list(truncated.items()):
                if key in _CORE_FIELDS:
                    continue
                if isinstance(val, str) and len(val) > 200:
                    truncated[key] = val[:200] + "...[truncated]"
            line = json.dumps(truncated, separators=(',', ':')) + "\n"

            # If still too large, drop all payload
            if len(line.encode()) > _PIPE_BUF:
                minimal = {k: data[k] for k in _CORE_FIELDS}
                minimal["_truncated"] = True
                line = json.dumps(minimal, separators=(',', ':')) + "\n"

        return line

    @classmethod
    def from_json_line(cls, line: str) -> "BusEvent":
        """Deserialize from a JSON line."""
        data = json.loads(line.strip())
        core = {k: data.pop(k) for k in list(_CORE_FIELDS) if k in data}
        return cls(**core, **data)


class EventBusWriter:
    """Append-only writer for the bus JSONL file."""

    def __init__(self, bus_path: Path, src: str, sid: str):
        self._bus_path = bus_path
        self._src = src
        self._sid = sid
        self._file = None

    def open(self):
        """Open the JSONL file for appending (O_APPEND for atomicity)."""
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._bus_path, "a")

    def emit(self, event_type: str, gen: int = 0, **payload):
        """Write one event line to the JSONL file."""
        if self._file is None:
            self.open()
        evt = BusEvent(ts=time.time(), src=self._src, type=event_type,
                       gen=gen, sid=self._sid, **payload)
        line = evt.to_json_line()
        self._file.write(line)
        self._file.flush()

    def close(self):
        """Close the file handle."""
        if self._file:
            self._file.close()
            self._file = None


class EventBusTailer:
    """Reads events from the bus JSONL file.

    Provides both sync read_recent() and async tail() generator.
    """

    def __init__(self, bus_path: Path):
        self._bus_path = bus_path

    def read_recent(self, last_n: int = 50, event_type: str | None = None,
                    since_ts: float | None = None) -> list[BusEvent]:
        """Read recent events from the JSONL file (sync).

        Args:
            last_n: Maximum number of events to return.
            event_type: Filter to only this event type.
            since_ts: Only events after this timestamp.
        """
        if not self._bus_path.exists():
            return []

        events = []
        try:
            with open(self._bus_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = BusEvent.from_json_line(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if event_type and evt.type != event_type:
                        continue
                    if since_ts and evt.ts < since_ts:
                        continue
                    events.append(evt)
        except (OSError, IOError):
            return []

        if last_n:
            events = events[-last_n:]
        return events

    async def tail(self, poll_interval: float = 0.2):
        """Async generator that yields events as they appear.

        Starts from the beginning of the file and follows new lines.
        """
        import asyncio

        while not self._bus_path.exists():
            await asyncio.sleep(poll_interval)

        with open(self._bus_path, "r") as f:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            evt = BusEvent.from_json_line(line)
                            yield evt
                        except (json.JSONDecodeError, KeyError):
                            continue
                else:
                    await asyncio.sleep(poll_interval)


class EventBus:
    """Unified event bus combining writer, in-process callbacks, and tailer.

    Usage:
        bus = EventBus(session_dir, "live_session", session_id)
        bus.open()
        bus.on("*", my_callback)            # Register listener
        bus.emit("status", gen=1, status="listening")  # Write + callbacks
        bus.emit_ephemeral("audio_rms", gen=0, rms=150)  # Callbacks only
        events = bus.read_recent(last_n=10)  # Read from file
        bus.close()
    """

    def __init__(self, session_dir: Path, src: str, sid: str):
        self._session_dir = session_dir
        self._src = src
        self._sid = sid
        self._bus_path = session_dir / "events.jsonl"
        self._writer: EventBusWriter | None = None
        self._callbacks: dict[str, list[Callable]] = {}  # type -> [callback]

    @property
    def bus_path(self) -> Path:
        return self._bus_path

    def open(self):
        """Open the writer for this bus."""
        self._writer = EventBusWriter(self._bus_path, self._src, self._sid)
        self._writer.open()

    def close(self):
        """Close the writer."""
        if self._writer:
            self._writer.close()
            self._writer = None

    def on(self, event_type: str, callback: Callable):
        """Register an in-process callback.

        Args:
            event_type: Event type to listen for, or "*" for all events.
            callback: Called with BusEvent as argument.
        """
        self._callbacks.setdefault(event_type, []).append(callback)

    def _fire_callbacks(self, evt: BusEvent):
        """Fire registered callbacks for an event."""
        for cb_type in (evt.type, "*"):
            for cb in self._callbacks.get(cb_type, []):
                try:
                    cb(evt)
                except Exception as e:
                    logger.error("Bus callback error for %s: %s", evt.type, e)

    def emit(self, event_type: str, gen: int = 0, **payload):
        """Write event to JSONL file and fire in-process callbacks."""
        evt = BusEvent(ts=time.time(), src=self._src, type=event_type,
                       gen=gen, sid=self._sid, **payload)

        # Write to disk
        if self._writer:
            line = evt.to_json_line()
            self._writer._file.write(line)
            self._writer._file.flush()

        # Fire callbacks
        self._fire_callbacks(evt)

    def emit_ephemeral(self, event_type: str, gen: int = 0, **payload):
        """Fire callbacks only, skip disk write. For high-frequency events."""
        evt = BusEvent(ts=time.time(), src=self._src, type=event_type,
                       gen=gen, sid=self._sid, **payload)
        self._fire_callbacks(evt)

    def read_recent(self, last_n: int = 50, event_type: str | None = None,
                    since_ts: float | None = None) -> list[BusEvent]:
        """Read recent events from the JSONL file."""
        tailer = EventBusTailer(self._bus_path)
        return tailer.read_recent(last_n=last_n, event_type=event_type, since_ts=since_ts)

    def tailer(self) -> EventBusTailer:
        """Create a new tailer for this bus."""
        return EventBusTailer(self._bus_path)


def build_llm_context(bus: EventBus, max_events: int = 30) -> str:
    """Build a concise pipeline context string for LLM system prompt injection.

    Returns recent key events formatted for the LLM, excluding high-frequency
    events like audio_rms and queue_depths.
    """
    cutoff = time.time() - _LLM_CONTEXT_MAX_AGE
    events = bus.read_recent(last_n=200, since_ts=cutoff)

    if not events:
        return ""

    # Filter out high-frequency/noise events
    filtered = [e for e in events if e.type not in _LLM_CONTEXT_EXCLUDE]
    if not filtered:
        return ""

    # Take the most recent max_events
    filtered = filtered[-max_events:]

    lines = ["[Pipeline Context - recent events]"]
    for evt in filtered:
        ts_str = time.strftime("%H:%M:%S", time.localtime(evt.ts))
        payload_parts = []
        for k, v in evt.payload.items():
            if isinstance(v, str) and len(v) > 100:
                v = v[:100] + "..."
            payload_parts.append(f"{k}={v}")
        payload_str = ", ".join(payload_parts) if payload_parts else ""
        lines.append(f"  {ts_str} [{evt.type}] {payload_str}")

    return "\n".join(lines)
