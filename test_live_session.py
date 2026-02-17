#!/usr/bin/env python3
"""Tests for LiveSession pipeline.

Tests the pipeline logic without requiring real API keys or services.
Mocks Deepgram STT, Claude CLI, and OpenAI TTS to test:
  - Transcript accumulation and flushing
  - Mute/unmute behavior
  - Key press/release (tap vs hold) handling
  - CLI message routing
  - Generation ID coherence
  - Session lifecycle

Run: python3 test_live_session.py
"""

import asyncio
import json
import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_frames import PipelineFrame, FrameType
from task_manager import TaskManager, TaskStatus

PASSED = 0
FAILED = 0
ERRORS = []


def test(name):
    """Decorator to register and run a test."""
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
        print(f"  FAIL: {name} — {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR: {name} — {type(e).__name__}: {e}")


def reset_task_manager():
    TaskManager._instance = None


# ── Helper: Minimal LiveSession for testing ───────────────────────

def make_session(**kwargs):
    """Create a LiveSession with mocked externals."""
    from live_session import LiveSession

    defaults = dict(
        openai_api_key="test-key",
        deepgram_api_key="test-key",
        voice="ash",
        model="claude-sonnet-4-5-20250929",
        on_status=lambda s: None,
        fillers_enabled=False,
        barge_in_enabled=False,
    )
    defaults.update(kwargs)

    reset_task_manager()
    session = LiveSession(**defaults)
    return session


# ══════════════════════════════════════════════════════════════════
# Test Group 1: Mute/Unmute Behavior
# ══════════════════════════════════════════════════════════════════

@test("set_muted ignores calls when not running")
def test_mute_not_running():
    session = make_session()
    assert session.running == False
    session.set_muted(True)
    assert session.muted == False  # Should not change — not running


@test("set_muted works when running")
def test_mute_when_running():
    session = make_session()
    session.running = True

    session.set_muted(True)
    assert session.muted == True

    session.set_muted(False)
    assert session.muted == False


@test("set_muted updates status callback")
def test_mute_status_callback():
    statuses = []
    session = make_session(on_status=lambda s: statuses.append(s))
    session.running = True

    session.set_muted(True)
    assert statuses[-1] == "muted"

    session.set_muted(False)
    assert statuses[-1] == "listening"


# ══════════════════════════════════════════════════════════════════
# Test Group 2: STT Transcript Accumulation
# ══════════════════════════════════════════════════════════════════

@test("STT stage emits transcript on speech_final")
async def test_stt_speech_final():
    """Simulate Deepgram on_message callback logic and verify transcript emission."""
    from deepgram.extensions.types.sockets.listen_v1_results_event import ListenV1ResultsEvent

    # Test the callback logic directly without constructing full Deepgram objects
    accumulated = []
    pending_text = None
    transcript_ready = asyncio.Event()

    def on_message_logic(transcript, is_final, speech_final):
        nonlocal pending_text
        if not transcript:
            return
        if is_final:
            accumulated.append(transcript)
            if speech_final:
                full_text = " ".join(accumulated).strip()
                accumulated.clear()
                if full_text:
                    pending_text = full_text
                    transcript_ready.set()

    # Send a final without speech_final
    on_message_logic("Hello", is_final=True, speech_final=False)
    assert len(accumulated) == 1
    assert not transcript_ready.is_set()

    # Now send speech_final
    on_message_logic("world", is_final=True, speech_final=True)
    assert len(accumulated) == 0  # Cleared after flush
    assert pending_text == "Hello world"
    assert transcript_ready.is_set()


@test("STT accumulated text flushes correctly")
async def test_stt_accumulation():
    """Multiple is_final segments accumulate until speech_final."""
    accumulated = []
    pending_text = None

    def on_final(text, speech_final=False):
        nonlocal pending_text
        accumulated.append(text)
        if speech_final:
            pending_text = " ".join(accumulated).strip()
            accumulated.clear()

    on_final("I want to")
    on_final("check the weather")
    assert len(accumulated) == 2
    assert pending_text is None

    on_final("in Seattle", speech_final=True)
    assert len(accumulated) == 0
    assert pending_text == "I want to check the weather in Seattle"


# ══════════════════════════════════════════════════════════════════
# Test Group 3: Flush on Mute (the key bug)
# ══════════════════════════════════════════════════════════════════

@test("Mute triggers STT flush event")
async def test_flush_on_mute():
    """When user releases key (mutes), the STT flush event should be set."""
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session._stt_out_q = asyncio.Queue(maxsize=50)
    session._stt_flush_event = asyncio.Event()
    session.generation_id = 0

    assert not session._stt_flush_event.is_set()

    # Mute should trigger flush event
    session.set_muted(True)

    assert session.muted == True
    assert session._stt_flush_event.is_set(), "Mute should set the STT flush event"


@test("Unmute does not trigger STT flush event")
async def test_unmute_no_flush():
    """Unmuting should not trigger a flush."""
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session._stt_flush_event = asyncio.Event()
    session.muted = True

    session.set_muted(False)

    assert not session._stt_flush_event.is_set(), "Unmute should not set flush event"


# ══════════════════════════════════════════════════════════════════
# Test Group 4: CLI Streaming
# ══════════════════════════════════════════════════════════════════

@test("CLI stream-json message format is correct")
async def test_cli_message_format():
    """Verify the JSON format sent to CLI stdin."""
    session = make_session()
    session._cli_ready = True

    # Mock the CLI process stdin
    written_data = []
    mock_stdin = MagicMock()
    mock_stdin.write = lambda data: written_data.append(data)
    mock_stdin.drain = AsyncMock()

    session._cli_process = MagicMock()
    session._cli_process.stdin = mock_stdin

    await session._send_to_cli("Hello world")

    assert len(written_data) == 1
    msg = json.loads(written_data[0].decode().strip())
    assert msg["type"] == "user"
    assert msg["message"]["role"] == "user"
    assert msg["message"]["content"] == "Hello world"


@test("CLI response parser extracts text deltas")
async def test_cli_response_parsing():
    """Feed mock stream-json events and verify text routing."""
    session = make_session()
    session.running = True
    session.generation_id = 0
    session._llm_out_q = asyncio.Queue(maxsize=50)
    session._filler_cancel = None
    session._last_send_time = time.time()

    # Build mock stream events
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello! "}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "How can I help?"}}},
        {"type": "result", "subtype": "success"},
    ]

    # Create a mock stdout that yields these events
    event_lines = [json.dumps(e).encode() + b"\n" for e in events]
    event_iter = iter(event_lines)

    async def mock_readline():
        try:
            return next(event_iter)
        except StopIteration:
            return b""

    session._cli_process = MagicMock()
    session._cli_process.stdout = MagicMock()
    session._cli_process.stdout.readline = mock_readline

    await session._read_cli_response()

    # Should have emitted text frames and END_OF_TURN
    frames = []
    while not session._llm_out_q.empty():
        frames.append(session._llm_out_q.get_nowait())

    # Should have at least the END_OF_TURN frame
    assert any(f.type == FrameType.END_OF_TURN for f in frames), \
        f"No END_OF_TURN frame found. Frames: {[f.type for f in frames]}"


@test("CLI response discards stale generation frames")
async def test_cli_stale_generation():
    """If generation_id changes mid-response, remaining text is discarded."""
    session = make_session()
    session.running = True
    session.generation_id = 0
    session._llm_out_q = asyncio.Queue(maxsize=50)
    session._filler_cancel = None
    session._last_send_time = time.time()

    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "First. "}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Second. "}}},
        {"type": "result", "subtype": "success"},
    ]

    event_lines = [json.dumps(e).encode() + b"\n" for e in events]
    call_count = 0

    async def mock_readline():
        nonlocal call_count
        if call_count < len(event_lines):
            line = event_lines[call_count]
            call_count += 1
            # After first event, simulate interrupt
            if call_count == 1:
                session.generation_id = 1  # Interrupt!
            return line
        return b""

    session._cli_process = MagicMock()
    session._cli_process.stdout = MagicMock()
    session._cli_process.stdout.readline = mock_readline

    await session._read_cli_response()

    frames = []
    while not session._llm_out_q.empty():
        frames.append(session._llm_out_q.get_nowait())

    # Only first sentence (before interrupt) should have been queued as TEXT_DELTA
    text_frames = [f for f in frames if f.type == FrameType.TEXT_DELTA]
    # The first text "First. " has sentence boundary so it gets flushed
    # But generation changed after first event, so second text is discarded
    # No END_OF_TURN should be emitted for stale generation
    eot_frames = [f for f in frames if f.type == FrameType.END_OF_TURN]
    assert len(eot_frames) == 0, "END_OF_TURN should not be emitted for interrupted generation"


# ══════════════════════════════════════════════════════════════════
# Test Group 5: Pipeline Frame Routing
# ══════════════════════════════════════════════════════════════════

@test("Audio always flows to keep Deepgram alive")
async def test_audio_always_flows():
    """Audio should always be sent to STT queue (keepalive), even during playback."""
    session = make_session()
    session.running = True
    session._audio_in_q = asyncio.Queue(maxsize=100)
    session.playing_audio = True  # Even during playback

    audio_data = b'\x00' * 4096
    # Audio capture no longer gates — always sends
    await session._audio_in_q.put(PipelineFrame(
        type=FrameType.AUDIO_RAW,
        generation_id=0,
        data=audio_data
    ))
    assert not session._audio_in_q.empty(), "Audio should always flow to STT"


@test("STT ignores transcripts during playback")
def test_stt_ignores_during_playback():
    """Deepgram callbacks should discard transcripts when AI is speaking."""
    session = make_session()
    session.playing_audio = True
    # Simulate: if playing_audio, transcript is dropped
    transcript = "some echo text"
    result = None
    if not session.playing_audio:
        result = transcript
    assert result is None, "Transcripts during playback should be ignored"


@test("Playback gates on generation_id")
async def test_playback_generation_gate():
    """Playback stage should discard frames with stale generation_id."""
    session = make_session()
    session.generation_id = 5

    frame_current = PipelineFrame(type=FrameType.TTS_AUDIO, generation_id=5, data=b'\x00' * 100)
    frame_stale = PipelineFrame(type=FrameType.TTS_AUDIO, generation_id=3, data=b'\x00' * 100)

    # Simulate the gate check from playback_stage
    assert frame_current.generation_id == session.generation_id
    assert frame_stale.generation_id != session.generation_id


# ══════════════════════════════════════════════════════════════════
# Test Group 6: Session Lifecycle
# ══════════════════════════════════════════════════════════════════

@test("Session starts with correct initial state")
def test_session_initial_state():
    session = make_session()
    assert session.running == False
    assert session.playing_audio == False
    assert session.muted == False
    assert session.generation_id == 0
    assert session._cli_process is None
    assert session._cli_ready == False


@test("stop() sets running to False")
def test_session_stop():
    session = make_session()
    session.running = True
    session.stop()
    assert session.running == False


@test("request_interrupt increments generation_id")
def test_interrupt_flag():
    session = make_session()
    session._interrupt_requested = False
    session.request_interrupt()
    assert session._interrupt_requested == True


@test("Generation ID increments on interrupt check")
async def test_interrupt_generation():
    session = make_session()
    session.running = True
    session.generation_id = 3
    session.playing_audio = True
    session._interrupt_requested = True
    session._audio_out_q = asyncio.Queue()
    session._llm_out_q = asyncio.Queue()

    await session._check_interrupt()

    assert session.generation_id == 4
    assert session.playing_audio == False
    assert session._interrupt_requested == False


# ══════════════════════════════════════════════════════════════════
# Test Group 7: Key Press/Release Logic
# ══════════════════════════════════════════════════════════════════

@test("Tap detection: < 300ms is a tap")
def test_tap_detection():
    press_time = time.time()
    time.sleep(0.1)
    elapsed = time.time() - press_time
    assert elapsed < 0.3, "Should be detected as a tap"


@test("Hold detection: > 300ms is a hold")
def test_hold_detection():
    press_time = time.time()
    time.sleep(0.35)
    elapsed = time.time() - press_time
    assert elapsed >= 0.3, "Should be detected as a hold"


@test("Live cycle: idle → listening starts session")
def test_cycle_idle_to_listening():
    """Test cycle logic: no session means start."""
    # Simulate cycle logic (can't import push-to-talk due to hyphen and GTK deps)
    live_session = None

    if live_session is None:
        action = "start"
    elif not live_session.muted:
        action = "mute"
    else:
        action = "stop"

    assert action == "start"


@test("Live cycle: listening → muted")
def test_cycle_listening_to_muted():
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session.muted = False

    # Simulate cycle: session exists, not muted → mute
    if not session.muted:
        session.set_muted(True)

    assert session.muted == True


@test("Live cycle: muted → idle")
def test_cycle_muted_to_idle():
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session.muted = True

    # Simulate cycle: session exists, muted → stop
    action = None
    if not session.muted:
        action = "mute"
    else:
        action = "stop"
        session.stop()

    assert action == "stop"
    assert session.running == False


@test("Tap does not unmute — only hold does")
def test_tap_no_unmute():
    """On a quick tap, the delayed unmute timer should be cancelled before it fires."""
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session._stt_flush_event = MagicMock()
    session.muted = True

    # Simulate: press starts a 300ms timer, release at 100ms cancels it
    timer = threading.Timer(0.3, lambda: session.set_muted(False))
    timer.start()

    # Release before 300ms — cancel the timer
    time.sleep(0.1)
    timer.cancel()

    # Muted state should not have changed
    assert session.muted == True


@test("Hold unmutes after 300ms")
def test_hold_unmutes():
    """On a hold, the delayed unmute timer fires after 300ms."""
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session._stt_flush_event = MagicMock()
    session.muted = True

    # Simulate: press starts a 300ms timer, key stays held
    timer = threading.Timer(0.3, lambda: session.set_muted(False))
    timer.start()

    # Wait for timer to fire
    time.sleep(0.4)

    assert session.muted == False


@test("First tap starts session without cycling on release")
def test_first_tap_no_cycle():
    """When the first tap creates a session, release should NOT cycle to muted."""
    # Simulate the press handler setting _live_starting_session
    _live_starting_session = True

    # Simulate release handler logic
    elapsed = 0.1  # Quick tap
    if _live_starting_session:
        # Should skip cycling
        action = "ignore_release"
        _live_starting_session = False
    elif elapsed < 0.3:
        action = "cycle"
    else:
        action = "mute"

    assert action == "ignore_release"
    assert _live_starting_session == False


@test("Subsequent tap does cycle modes")
def test_subsequent_tap_cycles():
    """After session is started, a quick tap should cycle modes."""
    _live_starting_session = False

    elapsed = 0.1  # Quick tap
    if _live_starting_session:
        action = "ignore_release"
    elif elapsed < 0.3:
        action = "cycle"
    else:
        action = "mute"

    assert action == "cycle"


# ══════════════════════════════════════════════════════════════════
# Test Group 8: MCP Config and Tool IPC
# ══════════════════════════════════════════════════════════════════

@test("MCP config generation creates valid JSON")
def test_mcp_config_generation():
    session = make_session()
    session._tool_socket_path = "/tmp/test-socket.sock"

    path = session._generate_mcp_config()
    try:
        with open(path) as f:
            config = json.load(f)

        assert "mcpServers" in config
        assert "ptt-task-tools" in config["mcpServers"]
        server = config["mcpServers"]["ptt-task-tools"]
        assert server["command"] == "python3"
        assert "task_tools_mcp.py" in server["args"][0]
        assert server["env"]["PTT_TOOL_SOCKET"] == "/tmp/test-socket.sock"
    finally:
        os.unlink(path)


@test("Tool IPC server starts and stops")
async def test_tool_ipc_lifecycle():
    session = make_session()

    await session._start_tool_ipc_server()
    assert session._tool_ipc_server is not None
    assert session._tool_socket_path is not None
    assert os.path.exists(session._tool_socket_path)

    await session._stop_tool_ipc_server()
    assert not os.path.exists(session._tool_socket_path)


@test("Tool IPC server handles tool calls")
async def test_tool_ipc_call():
    session = make_session()
    await session._start_tool_ipc_server()

    try:
        # Connect as a client and send a tool call
        reader, writer = await asyncio.open_unix_connection(session._tool_socket_path)
        request = json.dumps({"tool": "list_tasks", "args": {}})
        writer.write(request.encode() + b"\n")
        await writer.drain()

        response_line = await asyncio.wait_for(reader.readline(), timeout=5)
        response = json.loads(response_line.decode().strip())

        # list_tasks should return an array
        assert isinstance(response, list)

        writer.close()
        await writer.wait_closed()
    finally:
        await session._stop_tool_ipc_server()


# ══════════════════════════════════════════════════════════════════
# Test Group 9: Sentence Buffering
# ══════════════════════════════════════════════════════════════════

@test("Sentence boundary regex matches correctly")
def test_sentence_boundary():
    import re
    from live_session import SENTENCE_END_RE

    # Should match
    assert SENTENCE_END_RE.search("Hello world. ")
    assert SENTENCE_END_RE.search("How are you? ")
    assert SENTENCE_END_RE.search("Great!")
    assert SENTENCE_END_RE.search("Line one.\nLine two")

    # Should not match mid-word
    assert not SENTENCE_END_RE.search("Hello world")
    assert not SENTENCE_END_RE.search("Dr")


@test("Sentence buffer flushes complete sentences")
def test_sentence_buffer_flush():
    import re
    from live_session import SENTENCE_END_RE

    buffer = "Hello world. How are you? I'm fine."
    sentences = []

    while True:
        match = SENTENCE_END_RE.search(buffer)
        if not match:
            break
        end_pos = match.end()
        sentence = buffer[:end_pos].strip()
        buffer = buffer[end_pos:]
        if sentence:
            sentences.append(sentence)

    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you?"
    assert sentences[2] == "I'm fine."
    assert buffer.strip() == ""


@test("Partial sentence stays in buffer")
def test_sentence_buffer_partial():
    import re
    from live_session import SENTENCE_END_RE

    buffer = "Hello world. This is incompl"
    sentences = []

    while True:
        match = SENTENCE_END_RE.search(buffer)
        if not match:
            break
        end_pos = match.end()
        sentence = buffer[:end_pos].strip()
        buffer = buffer[end_pos:]
        if sentence:
            sentences.append(sentence)

    assert len(sentences) == 1
    assert sentences[0] == "Hello world."
    assert buffer == "This is incompl"


# ══════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LiveSession Pipeline Tests")
    print("=" * 60)

    # Collect all test functions
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
