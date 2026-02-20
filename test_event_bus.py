#!/usr/bin/env python3
"""Tests for the unified event bus module.

Tests: JSONL writing, BusEvent round-trip, read_recent filtering,
       in-process callbacks, ephemeral events, build_llm_context.

Run: python3 test_event_bus.py
"""

import asyncio
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

PASSED = 0
FAILED = 0
ERRORS = []


def test(name):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global PASSED, FAILED
    name = getattr(fn, '_test_name', fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  PASS: {name}")
    except AssertionError as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL: {name} -- {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR: {name} -- {type(e).__name__}: {e}")


# ======================================================================
# Test Group 1: BusEvent dataclass
# ======================================================================

@test("BusEvent to_json_line produces valid JSON with newline")
def test_bus_event_to_json():
    from event_bus import BusEvent
    evt = BusEvent(ts=1708444800.0, src="live_session", type="status",
                   gen=1, sid="20260220_141500", status="listening")
    line = evt.to_json_line()
    assert line.endswith("\n"), "Line must end with newline"
    parsed = json.loads(line)
    assert parsed["ts"] == 1708444800.0
    assert parsed["src"] == "live_session"
    assert parsed["type"] == "status"
    assert parsed["gen"] == 1
    assert parsed["sid"] == "20260220_141500"
    assert parsed["status"] == "listening"


@test("BusEvent round-trips through JSON")
def test_bus_event_roundtrip():
    from event_bus import BusEvent
    original = BusEvent(ts=1708444800.5, src="learner", type="learner_notify",
                        gen=0, sid="20260220_141500", summary="learned your name")
    line = original.to_json_line()
    restored = BusEvent.from_json_line(line)
    assert restored.ts == original.ts
    assert restored.src == original.src
    assert restored.type == original.type
    assert restored.gen == original.gen
    assert restored.sid == original.sid
    assert restored.payload["summary"] == "learned your name"


@test("BusEvent from_json_line handles extra payload fields")
def test_bus_event_extra_payload():
    from event_bus import BusEvent
    line = json.dumps({"ts": 1.0, "src": "x", "type": "foo", "gen": 0,
                       "sid": "s", "bar": 42, "baz": "hello"}) + "\n"
    evt = BusEvent.from_json_line(line)
    assert evt.payload["bar"] == 42
    assert evt.payload["baz"] == "hello"


@test("BusEvent to_json_line stays under 4096 bytes")
def test_bus_event_truncation():
    from event_bus import BusEvent
    # Create an event with a very long payload
    evt = BusEvent(ts=1.0, src="test", type="big", gen=0, sid="s",
                   text="x" * 5000)
    line = evt.to_json_line()
    assert len(line.encode()) <= 4096, f"Line is {len(line.encode())} bytes, exceeds 4096"


# ======================================================================
# Test Group 2: EventBusWriter
# ======================================================================

@test("Writer produces valid JSONL file")
def test_writer_produces_jsonl():
    from event_bus import EventBusWriter
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_path = Path(tmpdir) / "events.jsonl"
        writer = EventBusWriter(bus_path, "test_src", "test_sid")
        writer.open()
        writer.emit("status", gen=1, status="listening")
        writer.emit("user", gen=2, text="hello")
        writer.close()

        lines = bus_path.read_text().strip().split("\n")
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        for line in lines:
            parsed = json.loads(line)
            assert "ts" in parsed
            assert "src" in parsed
            assert parsed["src"] == "test_src"
            assert parsed["sid"] == "test_sid"


@test("Writer emit sets timestamp automatically")
def test_writer_auto_timestamp():
    from event_bus import EventBusWriter
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_path = Path(tmpdir) / "events.jsonl"
        writer = EventBusWriter(bus_path, "test", "sid1")
        writer.open()
        before = time.time()
        writer.emit("test_event", gen=0)
        after = time.time()
        writer.close()

        line = bus_path.read_text().strip()
        parsed = json.loads(line)
        assert before <= parsed["ts"] <= after


@test("Writer works without open (auto-opens)")
def test_writer_auto_open():
    from event_bus import EventBusWriter
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_path = Path(tmpdir) / "events.jsonl"
        writer = EventBusWriter(bus_path, "test", "sid1")
        writer.emit("hello", gen=0)
        writer.close()
        assert bus_path.exists()
        parsed = json.loads(bus_path.read_text().strip())
        assert parsed["type"] == "hello"


# ======================================================================
# Test Group 3: EventBus (writer + callbacks)
# ======================================================================

@test("EventBus.emit writes to disk and fires callbacks")
def test_bus_emit():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "live_session", "sid1")
        bus.open()

        events = []
        bus.on("*", lambda evt: events.append(evt))

        bus.emit("status", gen=1, status="listening")

        # Check disk
        bus_path = Path(tmpdir) / "events.jsonl"
        assert bus_path.exists()
        parsed = json.loads(bus_path.read_text().strip())
        assert parsed["type"] == "status"
        assert parsed["status"] == "listening"

        # Check callback
        assert len(events) == 1
        assert events[0].type == "status"

        bus.close()


@test("EventBus.emit_ephemeral fires callbacks but skips disk")
def test_bus_emit_ephemeral():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "live_session", "sid1")
        bus.open()

        events = []
        bus.on("*", lambda evt: events.append(evt))

        bus.emit_ephemeral("audio_rms", gen=0, rms=150.5)

        # Callback should fire
        assert len(events) == 1
        assert events[0].type == "audio_rms"

        # Disk should be empty
        bus_path = Path(tmpdir) / "events.jsonl"
        content = bus_path.read_text() if bus_path.exists() else ""
        assert content == "", f"Ephemeral should not write to disk, got: {content[:100]}"

        bus.close()


@test("EventBus.on with specific type only fires for that type")
def test_bus_on_specific_type():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "live_session", "sid1")
        bus.open()

        status_events = []
        bus.on("status", lambda evt: status_events.append(evt))

        bus.emit("status", gen=1, status="listening")
        bus.emit("user", gen=1, text="hello")
        bus.emit("status", gen=2, status="thinking")

        assert len(status_events) == 2, f"Expected 2 status events, got {len(status_events)}"
        assert status_events[0].payload["status"] == "listening"
        assert status_events[1].payload["status"] == "thinking"

        bus.close()


@test("EventBus wildcard callback fires for all events")
def test_bus_wildcard_callback():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "live_session", "sid1")
        bus.open()

        all_events = []
        bus.on("*", lambda evt: all_events.append(evt))

        bus.emit("status", gen=0)
        bus.emit("user", gen=0, text="hi")
        bus.emit_ephemeral("audio_rms", gen=0, rms=100)

        assert len(all_events) == 3

        bus.close()


# ======================================================================
# Test Group 4: read_recent
# ======================================================================

@test("read_recent returns last N events")
def test_read_recent_last_n():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        for i in range(10):
            bus.emit("counter", gen=0, i=i)

        recent = bus.read_recent(last_n=3)
        assert len(recent) == 3
        assert recent[0].payload["i"] == 7
        assert recent[2].payload["i"] == 9

        bus.close()


@test("read_recent filters by event_type")
def test_read_recent_type_filter():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        bus.emit("user", gen=0, text="hello")
        bus.emit("status", gen=0, status="thinking")
        bus.emit("assistant", gen=0, text="hi there")
        bus.emit("status", gen=0, status="listening")
        bus.emit("user", gen=0, text="bye")

        recent = bus.read_recent(event_type="user")
        assert len(recent) == 2
        assert recent[0].payload["text"] == "hello"
        assert recent[1].payload["text"] == "bye"

        bus.close()


@test("read_recent filters by since_ts")
def test_read_recent_since_ts():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        bus.emit("early", gen=0)
        cutoff = time.time()
        time.sleep(0.01)
        bus.emit("late", gen=0)

        recent = bus.read_recent(since_ts=cutoff)
        assert len(recent) == 1
        assert recent[0].type == "late"

        bus.close()


@test("read_recent with no file returns empty list")
def test_read_recent_no_file():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        # Don't open (no file created)
        recent = bus.read_recent(last_n=5)
        assert recent == []


# ======================================================================
# Test Group 5: build_llm_context
# ======================================================================

@test("build_llm_context returns formatted string with key events")
def test_build_llm_context():
    from event_bus import EventBus, build_llm_context
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        bus.emit("session_start", gen=0, model="claude-sonnet")
        bus.emit("user", gen=1, text="hello world")
        bus.emit("assistant", gen=1, text="hi there!")
        bus.emit("stt_complete", gen=1, text="hello world", latency_ms=340)
        bus.emit("llm_complete", gen=1, total_chars=20, sentences=1, latency_ms=500)

        ctx = build_llm_context(bus)
        assert isinstance(ctx, str)
        assert len(ctx) > 0
        # Should include user/assistant turns
        assert "hello world" in ctx or "user" in ctx

        bus.close()


@test("build_llm_context returns empty for no events")
def test_build_llm_context_empty():
    from event_bus import EventBus, build_llm_context
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        ctx = build_llm_context(bus)
        assert ctx == ""


@test("build_llm_context excludes high-frequency events")
def test_build_llm_context_excludes_ephemeral():
    from event_bus import EventBus, build_llm_context
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        # These are persisted to disk for the test (not ephemeral)
        # but build_llm_context should still exclude them
        bus.emit("user", gen=1, text="hello")
        # Manually write an audio_rms line to test exclusion
        bus_path = Path(tmpdir) / "events.jsonl"
        with open(bus_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "src": "test", "type": "audio_rms",
                                "gen": 0, "sid": "sid1", "rms": 100}) + "\n")

        ctx = build_llm_context(bus)
        assert "audio_rms" not in ctx

        bus.close()


# ======================================================================
# Test Group 6: EventBusTailer (async tail generator)
# ======================================================================

@test("Tailer yields events as they are written")
async def test_tailer_yields_events():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        # Write some events first
        bus.emit("user", gen=1, text="hello")
        bus.emit("assistant", gen=1, text="hi")

        # Create tailer
        tailer = bus.tailer()
        received = []

        async def read_events():
            async for evt in tailer.tail():
                received.append(evt)
                if len(received) >= 3:
                    break

        # Start reading and add one more event
        read_task = asyncio.create_task(read_events())
        await asyncio.sleep(0.1)
        bus.emit("user", gen=2, text="bye")

        try:
            await asyncio.wait_for(read_task, timeout=3.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) >= 2, f"Expected at least 2 events, got {len(received)}"
        assert received[0].type == "user"
        assert received[0].payload["text"] == "hello"

        bus.close()


@test("Tailer read_recent works synchronously")
def test_tailer_read_recent():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        bus.emit("user", gen=1, text="one")
        bus.emit("user", gen=2, text="two")
        bus.emit("user", gen=3, text="three")

        tailer = bus.tailer()
        recent = tailer.read_recent(last_n=2)
        assert len(recent) == 2
        assert recent[0].payload["text"] == "two"
        assert recent[1].payload["text"] == "three"

        bus.close()


# ======================================================================
# Test Group 7: EventType enum
# ======================================================================

@test("EventType has all catalog entries")
def test_event_type_catalog():
    from event_bus import EventType
    expected = [
        "session_start", "session_end", "status", "audio_rms", "queue_depths",
        "stt_start", "stt_complete", "user", "assistant",
        "llm_send", "llm_first_token", "llm_text_delta", "llm_tool_use", "llm_complete",
        "tts_start", "tts_complete", "filler_played",
        "barge_in", "command", "learner_notify",
        "task_complete", "task_failed", "error", "snapshot",
    ]
    for name in expected:
        assert hasattr(EventType, name.upper()), f"Missing EventType.{name.upper()}"
        assert EventType[name.upper()].value == name, f"EventType.{name.upper()} != '{name}'"


# ======================================================================
# Test Group 8: Multiple callbacks
# ======================================================================

@test("Multiple callbacks for same type all fire")
def test_multiple_callbacks():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        results_a = []
        results_b = []
        bus.on("status", lambda evt: results_a.append(evt))
        bus.on("status", lambda evt: results_b.append(evt))

        bus.emit("status", gen=0, status="listening")

        assert len(results_a) == 1
        assert len(results_b) == 1

        bus.close()


@test("Callback error doesn't prevent other callbacks")
def test_callback_error_isolation():
    from event_bus import EventBus
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = EventBus(Path(tmpdir), "test", "sid1")
        bus.open()

        results = []

        def bad_callback(evt):
            raise ValueError("boom")

        def good_callback(evt):
            results.append(evt)

        bus.on("*", bad_callback)
        bus.on("*", good_callback)

        bus.emit("test", gen=0)

        assert len(results) == 1, "Good callback should still fire despite bad callback"

        bus.close()


# ======================================================================
# Run all tests
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Event Bus Tests")
    print("=" * 60)

    tests = [
        obj for name, obj in sorted(globals().items())
        if callable(obj) and hasattr(obj, '_test_name')
    ]

    print(f"\nRunning {len(tests)} tests...\n")

    for fn in tests:
        run_test(fn)

    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED}")

    if ERRORS:
        print(f"\nFailures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")

    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)
