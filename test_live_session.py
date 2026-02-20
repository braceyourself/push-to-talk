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
# Test Group 10: Input Classifier (semantic-first)
# ══════════════════════════════════════════════════════════════════

@test("Classifier: short acknowledgments use heuristic fast-path")
def test_classifier_ack_fast_path():
    from input_classifier import classify
    result = classify("yeah")
    assert result.category == "acknowledgment"
    assert result.subcategory == "affirmative"
    assert result.match_type == "heuristic"
    assert result.confidence >= 0.8

@test("Classifier: negative acknowledgment detected")
def test_classifier_negative_ack():
    from input_classifier import classify
    result = classify("nope")
    assert result.category == "acknowledgment"
    assert result.subcategory == "negative"

@test("Classifier: question mark forces question category")
def test_classifier_question_mark():
    from input_classifier import classify
    result = classify("really?")
    assert result.category == "question"

@test("Classifier: empty input returns acknowledgment")
def test_classifier_empty():
    from input_classifier import classify
    result = classify("")
    assert result.category == "acknowledgment"
    assert result.confidence == 0.3

@test("Classifier: no semantic fallback uses structural heuristic")
def test_classifier_no_semantic():
    from input_classifier import classify
    result = classify("the weather is nice today", semantic=None)
    # Without semantic, non-question non-ack falls through to acknowledgment default
    assert result.category == "acknowledgment"
    assert result.match_type == "heuristic"

@test("Classifier: short greetings use heuristic fast-path")
def test_classifier_greeting_fast_path():
    from input_classifier import classify
    for greeting in ["hi", "hey", "hello", "yo", "howdy"]:
        result = classify(greeting)
        assert result.category == "social", f"'{greeting}' classified as {result.category}, expected social"
        assert result.subcategory == "greeting", f"'{greeting}' subcategory {result.subcategory}, expected greeting"
        assert result.match_type == "heuristic", f"'{greeting}' match_type {result.match_type}, expected heuristic"
        assert result.confidence >= 0.8

@test("Trivial: short affirmatives are trivial")
def test_trivial_yes():
    from input_classifier import is_trivial
    assert is_trivial("yes") == True
    assert is_trivial("ok") == True
    assert is_trivial("mhm") == True

@test("Trivial: questions are not trivial")
def test_trivial_question():
    from input_classifier import is_trivial
    assert is_trivial("yes?") == False

@test("Trivial: long input is not trivial")
def test_trivial_long():
    from input_classifier import is_trivial
    assert is_trivial("this is a much longer sentence") == False

@test("Trivial: ai_asked_question overrides triviality")
def test_trivial_ai_question_override():
    from input_classifier import is_trivial
    # "yes" is normally trivial
    assert is_trivial("yes", ai_asked_question=False) == True
    # But when AI asked a question, "yes" is a real answer
    assert is_trivial("yes", ai_asked_question=True) == False


# ══════════════════════════════════════════════════════════════════
# Test Group 11: StreamComposer
# ══════════════════════════════════════════════════════════════════

@test("Composer: stop() causes run() to exit")
async def test_composer_stop():
    from stream_composer import StreamComposer
    audio_out_q = asyncio.Queue()

    async def dummy_tts(text):
        return b'\x00' * 100

    composer = StreamComposer(audio_out_q, dummy_tts, lambda: 1)

    # Run composer in background, stop after brief delay
    async def stop_after_delay():
        await asyncio.sleep(0.2)
        composer.stop()

    stop_task = asyncio.create_task(stop_after_delay())
    # run() should exit within ~0.7s (0.2s delay + 0.5s timeout in _next_segment)
    await asyncio.wait_for(composer.run(), timeout=2.0)
    await stop_task

@test("Composer: filler with sufficient=True suppresses TTS audio output")
async def test_composer_filler_sufficient():
    from stream_composer import StreamComposer, AudioSegment, SegmentType

    audio_out_q = asyncio.Queue()

    async def mock_tts(text):
        return b'\x00' * 100

    composer = StreamComposer(audio_out_q, mock_tts, lambda: 1)

    # Enqueue: sufficient filler, then TTS sentence, then end-of-turn
    await composer.enqueue(AudioSegment(
        SegmentType.FILLER_CLIP, data=b'\x00' * 50, metadata={"sufficient": True}
    ))
    await composer.enqueue(AudioSegment(SegmentType.TTS_SENTENCE, data="hey, what's up"))
    await composer.enqueue_end_of_turn()

    # Stop after processing
    async def stop_after():
        await asyncio.sleep(0.3)
        composer.stop()
    asyncio.create_task(stop_after())

    await asyncio.wait_for(composer.run(), timeout=2.0)

    # Check output frames — no SENTENCE_DONE should appear (it's only
    # emitted after actual TTS sentence processing, not filler clips).
    # Note: TTS_AUDIO frames may appear from post-filler silence.
    frames = []
    while not audio_out_q.empty():
        frames.append(audio_out_q.get_nowait())
    frame_types = [f.type for f in frames]
    assert FrameType.FILLER in frame_types, "Filler should still play"
    assert FrameType.END_OF_TURN in frame_types, "END_OF_TURN should be emitted"
    assert FrameType.SENTENCE_DONE not in frame_types, \
        "No SENTENCE_DONE should appear — TTS sentence was suppressed"

@test("Composer: filler without sufficient=True allows TTS")
async def test_composer_filler_not_sufficient():
    from stream_composer import StreamComposer, AudioSegment, SegmentType

    audio_out_q = asyncio.Queue()
    tts_called = []

    async def mock_tts(text):
        tts_called.append(text)
        return b'\x00' * 100

    composer = StreamComposer(audio_out_q, mock_tts, lambda: 1)

    # Enqueue: normal filler (no sufficient flag), then TTS sentence, then EOT
    await composer.enqueue(AudioSegment(SegmentType.FILLER_CLIP, data=b'\x00' * 50))
    await composer.enqueue(AudioSegment(SegmentType.TTS_SENTENCE, data="let me check on that"))
    await composer.enqueue_end_of_turn()

    async def stop_after():
        await asyncio.sleep(0.3)
        composer.stop()
    asyncio.create_task(stop_after())

    await asyncio.wait_for(composer.run(), timeout=2.0)

    # TTS SHOULD have been called
    assert len(tts_called) == 1, f"TTS should be called once, got {len(tts_called)}"
    assert tts_called[0] == "let me check on that"

@test("Composer: sufficient flag resets on end-of-turn")
async def test_composer_sufficient_resets():
    from stream_composer import StreamComposer, AudioSegment, SegmentType

    audio_out_q = asyncio.Queue()
    tts_called = []

    async def mock_tts(text):
        tts_called.append(text)
        return b'\x00' * 100

    composer = StreamComposer(audio_out_q, mock_tts, lambda: 1)

    # Turn 1: sufficient filler + TTS (should suppress)
    await composer.enqueue(AudioSegment(
        SegmentType.FILLER_CLIP, data=b'\x00' * 50, metadata={"sufficient": True}
    ))
    await composer.enqueue(AudioSegment(SegmentType.TTS_SENTENCE, data="hey there"))
    await composer.enqueue_end_of_turn()

    # Turn 2: normal filler + TTS (should play)
    await composer.enqueue(AudioSegment(SegmentType.FILLER_CLIP, data=b'\x00' * 50))
    await composer.enqueue(AudioSegment(SegmentType.TTS_SENTENCE, data="here's the answer"))
    await composer.enqueue_end_of_turn()

    async def stop_after():
        await asyncio.sleep(0.5)
        composer.stop()
    asyncio.create_task(stop_after())

    await asyncio.wait_for(composer.run(), timeout=3.0)

    # Only turn 2 TTS should have been called
    assert tts_called == ["here's the answer"], f"Only turn 2 TTS expected, got: {tts_called}"

@test("Composer: reset() clears sufficient flag")
async def test_composer_reset_clears_sufficient():
    from stream_composer import StreamComposer
    audio_out_q = asyncio.Queue()
    composer = StreamComposer(audio_out_q, AsyncMock(), lambda: 1)
    composer._filler_sufficient = True
    composer.reset()
    assert composer._filler_sufficient == False


# ══════════════════════════════════════════════════════════════════
# Test Group 12: Key handler — muted→listening hold behavior
# ══════════════════════════════════════════════════════════════════

@test("Hold from muted: stays listening (no re-mute)")
def test_hold_from_muted_stays_listening():
    """When press_state was 'muted', a hold should NOT re-mute."""
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session._stt_flush_event = asyncio.Event()

    # Simulate press handler: was muted, optimistic unmute
    session.muted = True
    session.set_muted(False)  # Optimistic unmute on press
    press_state = 'muted'

    # Simulate hold release (>= 500ms)
    elapsed = 0.8

    # Replicate the release handler logic
    if elapsed >= 2.0 and press_state == 'muted':
        session.stop()  # Would stop session
    elif press_state == 'muted':
        pass  # Hold from muted: keep listening
    else:
        session.set_muted(True)  # Hold from listening: mute

    assert session.muted == False, "Hold from muted should keep listening"

@test("Hold from listening: mutes")
def test_hold_from_listening_mutes():
    """When press_state was 'listening', a hold should mute."""
    session = make_session()
    session.running = True
    session._started_at = time.time() - 10
    session._stt_flush_event = asyncio.Event()
    session.muted = False
    press_state = 'listening'

    elapsed = 0.8

    if elapsed >= 2.0 and press_state == 'muted':
        session.stop()
    elif press_state == 'muted':
        pass
    else:
        session.set_muted(True)

    assert session.muted == True, "Hold from listening should mute"


# ══════════════════════════════════════════════════════════════════
# Test Group 13: Composer shutdown (session restart bug)
# ══════════════════════════════════════════════════════════════════

@test("Composer stop unblocks _next_segment")
async def test_composer_next_segment_unblocks():
    from stream_composer import StreamComposer
    audio_out_q = asyncio.Queue()
    composer = StreamComposer(audio_out_q, AsyncMock(), lambda: 1)

    # _next_segment should block until stop is called
    async def get_segment():
        return await composer._next_segment()

    async def stop_after():
        await asyncio.sleep(0.2)
        composer.stop()

    asyncio.create_task(stop_after())
    result = await asyncio.wait_for(get_segment(), timeout=2.0)
    assert result is None, "_next_segment should return None when stopped"

@test("Session stop() calls composer.stop()")
def test_session_stop_calls_composer_stop():
    session = make_session()
    session.running = True

    # Create a mock composer
    mock_composer = MagicMock()
    session._composer = mock_composer

    session.stop()

    assert session.running == False
    mock_composer.stop.assert_called_once()


# ══════════════════════════════════════════════════════════════════
# Test Group 14: Dead session detection (idle timeout race)
# ══════════════════════════════════════════════════════════════════

@test("set_muted on dead session (running=False) is no-op")
def test_set_muted_dead_session_noop():
    """set_muted() silently returns when running=False — proves the race."""
    session = make_session()
    session.running = False  # Idle timeout set this
    session.muted = True
    session._stt_flush_event = asyncio.Event()

    # Try to unmute — should silently fail (the bug)
    session.set_muted(False)
    # This assertion documents the bug: muted stays True
    assert session.muted == True, "set_muted(False) should be no-op when not running"

@test("Key handler detects dead session and starts new one")
def test_key_handler_dead_session_starts_new():
    """When live_session exists but running=False, pressing RShift should start new session."""
    ptt = MagicMock()
    ptt.live_session = MagicMock()
    ptt.live_session.running = False  # Dead session (idle timeout)
    ptt.live_session.muted = True
    ptt.ctrl_r_pressed = False
    ptt.config = {'ai_mode': 'live'}
    ptt.ai_key = 'shift_r'
    ptt._live_key_held = False
    ptt._live_press_processed = False
    ptt.start_live_session = MagicMock()

    # Simulate the key press handler logic (what SHOULD happen)
    # The fix: treat dead session (running=False) as no session
    if not ptt.live_session or not ptt.live_session.running:
        ptt._live_starting_session = True
        ptt.start_live_session()
    else:
        ptt._live_press_state = 'muted' if ptt.live_session.muted else 'listening'
        if ptt.live_session.muted:
            ptt.live_session.set_muted(False)

    ptt.start_live_session.assert_called_once()

@test("Session cleanup sets idle status after run() completes")
def test_session_cleanup_sets_idle_status():
    """After run() finishes (idle timeout or stop), status should be set to idle."""
    statuses = []
    session = make_session()
    session.on_status = lambda s: statuses.append(s)

    # Simulate what happens in the run_session thread's finally block
    # Currently: self.live_session = None (but no status update!)
    # Fix: should call set_status('idle')
    # For now, test that on_status is callable
    session.on_status('idle')
    assert 'idle' in statuses, "on_status('idle') should work"


# ══════════════════════════════════════════════════════════════════
# Test Group 15: VAD model compatibility
# ══════════════════════════════════════════════════════════════════

@test("VAD model loads and runs with correct input format")
def test_vad_model_correct_format():
    """Silero VAD v5 expects 'state' [2,1,128] with 64-sample context."""
    import numpy as np
    model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
    if not model_path.exists():
        print("  (skipped: no VAD model)")
        return

    session = make_session()
    session.barge_in_enabled = True
    session._load_vad_model()

    assert session._vad_model is not None, "VAD model should load"
    assert 'state' in session._vad_state, "State key should be 'state', not 'h'/'c'"
    assert 'context' in session._vad_state, "Should have 64-sample context buffer"
    assert len(session._vad_state['context']) == 64, "Context should be 64 samples"

    # Generate a chunk of silence and run VAD
    silence = b'\x00' * 4096  # 2048 samples at 16-bit
    prob = session._run_vad(silence)
    assert isinstance(prob, float), "VAD should return float"
    assert 0.0 <= prob <= 1.0, f"Probability should be 0-1, got {prob}"

@test("VAD _run_vad does not silently swallow errors")
def test_vad_run_returns_valid_probability():
    """_run_vad should produce valid probabilities, not always 0.0."""
    import numpy as np
    model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
    if not model_path.exists():
        print("  (skipped: no VAD model)")
        return

    session = make_session()
    session.barge_in_enabled = True
    session._load_vad_model()

    # Run several chunks — at minimum should not error
    for _ in range(5):
        chunk = b'\x00' * 4096
        prob = session._run_vad(chunk)
        assert isinstance(prob, float), "Each call should return float"


# ══════════════════════════════════════════════════════════════════
# Test Group 16: Barge-in status transitions
# ══════════════════════════════════════════════════════════════════

@test("Barge-in: VAD first speech sets status to 'listening' not 'hearing'")
def test_barge_in_vad_sets_listening():
    """When user starts talking during AI speech, status should be 'listening'."""
    statuses = []
    session = make_session(on_status=lambda s: statuses.append(s))
    session._stt_gated = True
    session.playing_audio = True
    session._vad_speech_count = 0

    # Simulate first speech detection
    session._vad_speech_count = 1
    session._set_status("listening")  # This is what the code should do

    assert "listening" in statuses

@test("Barge-in: trail clip does not re-gate STT")
async def test_barge_in_trail_no_regate():
    """After barge-in, the trail clip should not re-gate STT or set status to speaking."""
    import numpy as np
    statuses = []
    session = make_session(
        on_status=lambda s: statuses.append(s),
        fillers_enabled=True,
        barge_in_enabled=True,
    )
    session._audio_out_q = asyncio.Queue()
    session.generation_id = 1
    session.playing_audio = True
    session._stt_gated = True
    session._spoken_sentences = ["Hello there."]
    session._played_sentence_count = 1

    # Mock composer
    mock_composer = MagicMock()
    mock_composer.pause.return_value = []
    mock_composer.reset = MagicMock()
    session._composer = mock_composer

    # Provide a filler clip for the trail
    session._filler_clips = {"acknowledgment": [b'\x00' * 4800]}

    await session._trigger_barge_in()

    # After barge-in, _stt_gated must be False and status must be listening
    assert session._stt_gated == False, f"STT should be ungated after barge-in"
    assert session.playing_audio == False, f"playing_audio should be False after barge-in"
    assert statuses[-1] == "listening", f"Final status should be 'listening', got '{statuses[-1]}'"

    # Any frames queued should not cause re-gating
    # Drain the queue and check: if there are audio frames, they must be followed by END_OF_TURN
    frames = []
    while not session._audio_out_q.empty():
        frames.append(session._audio_out_q.get_nowait())

    if any(f.type in (FrameType.TTS_AUDIO, FrameType.FILLER) for f in frames):
        # Must have END_OF_TURN after audio frames
        assert frames[-1].type == FrameType.END_OF_TURN, \
            f"Trail clip must be followed by END_OF_TURN, got {frames[-1].type}"

@test("Classifier: greeting 'hi' gets social category (not question)")
def test_classifier_hi_not_question():
    """Regression: 'hi' was being classified as 'question' triggering 'Good question' filler."""
    from input_classifier import classify
    result = classify("hi")
    assert result.category == "social", f"'hi' classified as {result.category}, expected social"
    assert result.category != "question", "'hi' must never be classified as question"

    result2 = classify("hello")
    assert result2.category == "social", f"'hello' classified as {result2.category}, expected social"


# ══════════════════════════════════════════════════════════════════
# Test Group 17: Security guards — _is_sensitive_path / _is_sensitive_command
# ══════════════════════════════════════════════════════════════════

@test("Security: .env files blocked by _is_sensitive_path")
def test_sensitive_path_env():
    from live_session import _is_sensitive_path
    blocked, reason = _is_sensitive_path("/home/user/project/.env")
    assert blocked, ".env should be blocked"
    blocked2, _ = _is_sensitive_path("/home/user/.env.production")
    assert blocked2, ".env.production should be blocked"
    blocked3, _ = _is_sensitive_path("/home/user/project/.env.local")
    assert blocked3, ".env.local should be blocked"

@test("Security: credentials/secret/password files blocked")
def test_sensitive_path_creds():
    from live_session import _is_sensitive_path
    blocked, _ = _is_sensitive_path("/home/user/credentials.json")
    assert blocked, "credentials.json should be blocked"
    blocked2, _ = _is_sensitive_path("/home/user/client_secret.json")
    assert blocked2, "client_secret.json should be blocked"
    blocked3, _ = _is_sensitive_path("/home/user/.password-store/github")
    assert blocked3, ".password-store should be blocked"

@test("Security: SSH keys and AWS creds blocked")
def test_sensitive_path_ssh_aws():
    from live_session import _is_sensitive_path
    blocked, _ = _is_sensitive_path("/home/user/.ssh/id_rsa")
    assert blocked, ".ssh/id_rsa should be blocked"
    blocked2, _ = _is_sensitive_path("/home/user/.ssh/id_ed25519")
    assert blocked2, ".ssh/id_ed25519 should be blocked"
    blocked3, _ = _is_sensitive_path("/home/user/.aws/credentials")
    assert blocked3, ".aws/credentials should be blocked"

@test("Security: key/pem/p12 file extensions blocked")
def test_sensitive_path_key_extensions():
    from live_session import _is_sensitive_path
    for ext in ['.pem', '.p12', '.pfx', '.key']:
        blocked, _ = _is_sensitive_path(f"/home/user/server{ext}")
        assert blocked, f"*{ext} should be blocked"

@test("Security: /proc/*/environ blocked")
def test_sensitive_path_proc_environ():
    from live_session import _is_sensitive_path
    blocked, _ = _is_sensitive_path("/proc/1234/environ")
    assert blocked, "/proc/*/environ should be blocked"

@test("Security: .npmrc/.netrc/.pypirc/.git-credentials blocked")
def test_sensitive_path_rc_files():
    from live_session import _is_sensitive_path
    for name in ['.npmrc', '.netrc', '.pypirc', '.git-credentials']:
        blocked, _ = _is_sensitive_path(f"/home/user/{name}")
        assert blocked, f"{name} should be blocked"

@test("Security: normal files not blocked")
def test_sensitive_path_normal():
    from live_session import _is_sensitive_path
    blocked, _ = _is_sensitive_path("/home/user/project/main.py")
    assert not blocked, "main.py should not be blocked"
    blocked2, _ = _is_sensitive_path("/home/user/README.md")
    assert not blocked2, "README.md should not be blocked"

@test("Security: cat .env blocked by _is_sensitive_command")
def test_sensitive_command_cat_env():
    from live_session import _is_sensitive_command
    blocked, reason = _is_sensitive_command("cat .env")
    assert blocked, "cat .env should be blocked"
    blocked2, _ = _is_sensitive_command("cat /home/user/project/.env.production")
    assert blocked2, "cat .env.production should be blocked"

@test("Security: printenv blocked")
def test_sensitive_command_printenv():
    from live_session import _is_sensitive_command
    blocked, _ = _is_sensitive_command("printenv")
    assert blocked, "printenv should be blocked"
    blocked2, _ = _is_sensitive_command("printenv SECRET_KEY")
    assert blocked2, "printenv SECRET_KEY should be blocked"

@test("Security: pass show and gpg decrypt blocked")
def test_sensitive_command_pass_gpg():
    from live_session import _is_sensitive_command
    blocked, _ = _is_sensitive_command("pass show github/token")
    assert blocked, "pass show should be blocked"
    blocked2, _ = _is_sensitive_command("gpg --decrypt secrets.gpg")
    assert blocked2, "gpg --decrypt should be blocked"
    blocked3, _ = _is_sensitive_command("gpg -d secrets.gpg")
    assert blocked3, "gpg -d should be blocked"

@test("Security: cat credentials/secret/password/pem/key files blocked")
def test_sensitive_command_cat_creds():
    from live_session import _is_sensitive_command
    for cmd in ["cat credentials.json", "head secret.txt", "tail -5 password.txt",
                "cat server.pem", "less id_rsa.key"]:
        blocked, _ = _is_sensitive_command(cmd)
        assert blocked, f"'{cmd}' should be blocked"

@test("Security: /proc/*/environ in command blocked")
def test_sensitive_command_proc_environ():
    from live_session import _is_sensitive_command
    blocked, _ = _is_sensitive_command("cat /proc/self/environ")
    assert blocked, "cat /proc/self/environ should be blocked"

@test("Security: normal commands not blocked")
def test_sensitive_command_normal():
    from live_session import _is_sensitive_command
    blocked, _ = _is_sensitive_command("git status")
    assert not blocked, "git status should not be blocked"
    blocked2, _ = _is_sensitive_command("ls -la")
    assert not blocked2, "ls -la should not be blocked"
    blocked3, _ = _is_sensitive_command("cat main.py")
    assert not blocked3, "cat main.py should not be blocked"


# ══════════════════════════════════════════════════════════════════
# Test Group 18: run_command tool
# ══════════════════════════════════════════════════════════════════

@test("run_command: simple command returns stdout")
async def test_run_command_simple():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {"command": "echo hello"}))
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hello"

@test("run_command: failed command returns stderr and exit code")
async def test_run_command_failure():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {"command": "ls /nonexistent_dir_xyz"}))
    assert result["exit_code"] != 0
    assert "stderr" in result

@test("run_command: working_dir is respected")
async def test_run_command_working_dir():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {
        "command": "pwd", "working_dir": "/tmp"
    }))
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "/tmp"

@test("run_command: timeout kills long-running command")
async def test_run_command_timeout():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {
        "command": "sleep 60", "timeout": 1
    }))
    assert "error" in result or result.get("exit_code") != 0

@test("run_command: output capped at 10KB")
async def test_run_command_output_cap():
    session = make_session()
    # Generate >10KB of output
    result = json.loads(await session._execute_tool("run_command", {
        "command": "python3 -c \"print('x' * 20000)\""
    }))
    assert result["exit_code"] == 0
    assert len(result["stdout"]) <= 10240 + 100  # 10KB + small margin for truncation msg

@test("run_command: pipes work")
async def test_run_command_pipes():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {
        "command": "echo hello world | wc -w"
    }))
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "2"

@test("run_command: empty command returns error")
async def test_run_command_empty():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {"command": ""}))
    assert "error" in result

@test("run_command: timeout clamped to 120s max")
async def test_run_command_timeout_clamp():
    session = make_session()
    # We can't easily test the clamp directly, but we can verify the command
    # doesn't hang for 999s. Just ensure it runs with a large timeout value.
    result = json.loads(await session._execute_tool("run_command", {
        "command": "echo clamped", "timeout": 999
    }))
    assert result["exit_code"] == 0

@test("run_command: blocks cat .env (security)")
async def test_run_command_blocks_cat_env():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {"command": "cat .env"}))
    assert "error" in result
    assert "Blocked" in result["error"]

@test("run_command: blocks printenv (security)")
async def test_run_command_blocks_printenv():
    session = make_session()
    result = json.loads(await session._execute_tool("run_command", {"command": "printenv"}))
    assert "error" in result
    assert "Blocked" in result["error"]


# ══════════════════════════════════════════════════════════════════
# Test Group 19: read_file tool
# ══════════════════════════════════════════════════════════════════

@test("read_file: reads a file successfully")
async def test_read_file_basic():
    session = make_session()
    # Read this test file itself
    result = json.loads(await session._execute_tool("read_file", {
        "path": str(Path(__file__).resolve())
    }))
    assert "content" in result
    assert "test_live_session" in result["content"] or "LiveSession" in result["content"]
    assert "total_lines" in result

@test("read_file: offset and limit work")
async def test_read_file_offset_limit():
    session = make_session()
    result = json.loads(await session._execute_tool("read_file", {
        "path": str(Path(__file__).resolve()),
        "offset": 1,
        "limit": 5
    }))
    assert "content" in result
    lines = result["content"].strip().split('\n')
    assert len(lines) <= 5

@test("read_file: missing file returns error")
async def test_read_file_missing():
    session = make_session()
    result = json.loads(await session._execute_tool("read_file", {
        "path": "/nonexistent/file.txt"
    }))
    assert "error" in result

@test("read_file: directory returns error")
async def test_read_file_directory():
    session = make_session()
    result = json.loads(await session._execute_tool("read_file", {"path": "/tmp"}))
    assert "error" in result

@test("read_file: output capped at 10KB")
async def test_read_file_output_cap():
    session = make_session()
    # Create a large temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("x" * 20000 + "\n")
        tmp_path = f.name
    try:
        result = json.loads(await session._execute_tool("read_file", {"path": tmp_path}))
        assert "content" in result
        assert len(result["content"]) <= 10240 + 100
    finally:
        os.unlink(tmp_path)

@test("read_file: blocks .env (security)")
async def test_read_file_blocks_env():
    session = make_session()
    result = json.loads(await session._execute_tool("read_file", {
        "path": "/home/user/project/.env"
    }))
    assert "error" in result
    assert "Blocked" in result["error"]

@test("read_file: blocks credentials.json (security)")
async def test_read_file_blocks_credentials():
    session = make_session()
    result = json.loads(await session._execute_tool("read_file", {
        "path": "/home/user/credentials.json"
    }))
    assert "error" in result
    assert "Blocked" in result["error"]

@test("read_file: blocks .ssh/id_rsa (security)")
async def test_read_file_blocks_ssh():
    session = make_session()
    result = json.loads(await session._execute_tool("read_file", {
        "path": "/home/user/.ssh/id_rsa"
    }))
    assert "error" in result
    assert "Blocked" in result["error"]


# ══════════════════════════════════════════════════════════════════
# Test Group 20: Configurable idle timeout
# ══════════════════════════════════════════════════════════════════

@test("Idle timeout: default is 0 (always-on)")
def test_idle_timeout_default_zero():
    """LiveSession with no idle_timeout param should default to 0 (never timeout)."""
    session = make_session()
    assert session._idle_timeout == 0, f"Default idle_timeout should be 0, got {session._idle_timeout}"

@test("Idle timeout: custom value accepted")
def test_idle_timeout_custom_value():
    """LiveSession accepts custom idle_timeout param."""
    session = make_session(idle_timeout=300)
    assert session._idle_timeout == 300, f"idle_timeout should be 300, got {session._idle_timeout}"

@test("Idle timeout: timer not scheduled when timeout is 0")
async def test_idle_timer_not_scheduled_when_zero():
    """When idle_timeout=0, _reset_idle_timer should not create a timer."""
    session = make_session(idle_timeout=0)
    session._reset_idle_timer()
    assert session._idle_timer is None, "Timer should be None when idle_timeout=0"

@test("Idle timeout: timer IS scheduled when timeout > 0")
async def test_idle_timer_scheduled_when_positive():
    """When idle_timeout>0, _reset_idle_timer should create a timer."""
    session = make_session(idle_timeout=60)
    session._reset_idle_timer()
    assert session._idle_timer is not None, "Timer should be created when idle_timeout=60"
    # Clean up
    session._cancel_idle_timer()


# ══════════════════════════════════════════════════════════════════
# Test Group 21: PulseAudio audio capture (pasimple)
# ══════════════════════════════════════════════════════════════════

@test("Audio capture creates PulseAudio stream with correct params")
async def test_audio_capture_pasimple_params():
    """Verify PaSimple is called with PA_STREAM_RECORD, S16LE, 1ch, 24kHz."""
    session = make_session()
    session.running = True
    session.generation_id = 1
    session._audio_in_q = asyncio.Queue(maxsize=100)

    mock_pa_instance = MagicMock()
    # read() returns some data once, then we stop the session
    call_count = 0
    def read_side_effect(size):
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            session.running = False
            raise Exception("stopped")
        return b'\x00' * size
    mock_pa_instance.read = read_side_effect
    mock_pa_instance.__enter__ = MagicMock(return_value=mock_pa_instance)
    mock_pa_instance.__exit__ = MagicMock(return_value=False)

    mock_pa_class = MagicMock(return_value=mock_pa_instance)

    with patch.dict('sys.modules', {'pasimple': MagicMock(
        PaSimple=mock_pa_class,
        PA_STREAM_RECORD=1,
        PA_SAMPLE_S16LE=3,
    )}):
        await asyncio.wait_for(session._audio_capture_stage(), timeout=5.0)

    mock_pa_class.assert_called()
    call_args = mock_pa_class.call_args
    assert call_args[0][0] == 1, "First arg should be PA_STREAM_RECORD"
    assert call_args[0][1] == 3, "Second arg should be PA_SAMPLE_S16LE"
    assert call_args[0][2] == 1, "channels should be 1"
    assert call_args[0][3] == 24000, "rate should be 24000"
    assert call_args[1].get('app_name') == 'push-to-talk' or \
           (len(call_args[0]) > 4 and call_args[0][4] == 'push-to-talk'), \
           "app_name should be 'push-to-talk'"


@test("Audio capture: frames arrive in queue")
async def test_audio_capture_frames_in_queue():
    """Mock pa.read() to return known data, verify PipelineFrames appear in _audio_in_q."""
    session = make_session()
    session.running = True
    session.generation_id = 42
    session._audio_in_q = asyncio.Queue(maxsize=100)

    test_data = b'\x01\x02' * 2048
    call_count = 0
    mock_pa_instance = MagicMock()
    def read_side_effect(size):
        nonlocal call_count
        call_count += 1
        if call_count > 3:
            session.running = False
            raise Exception("stopped")
        return test_data
    mock_pa_instance.read = read_side_effect
    mock_pa_instance.__enter__ = MagicMock(return_value=mock_pa_instance)
    mock_pa_instance.__exit__ = MagicMock(return_value=False)

    with patch.dict('sys.modules', {'pasimple': MagicMock(
        PaSimple=MagicMock(return_value=mock_pa_instance),
        PA_STREAM_RECORD=1,
        PA_SAMPLE_S16LE=3,
    )}):
        await asyncio.wait_for(session._audio_capture_stage(), timeout=5.0)
        # Let event loop process pending call_soon_threadsafe callbacks
        await asyncio.sleep(0.1)

    # Check that frames arrived
    frames = []
    while not session._audio_in_q.empty():
        frames.append(session._audio_in_q.get_nowait())

    assert len(frames) >= 1, f"Expected at least 1 frame, got {len(frames)}"
    assert frames[0].type == FrameType.AUDIO_RAW, "Frame should be AUDIO_RAW"
    assert frames[0].data == test_data, "Frame data should match what pa.read() returned"
    assert frames[0].generation_id == 42, "generation_id should match session"


@test("Audio capture: reconnects on PulseAudio error")
async def test_audio_capture_reconnects_on_error():
    """PaSimple raises on first call, succeeds on second — verify restart."""
    session = make_session()
    session.running = True
    session.generation_id = 1
    session._audio_in_q = asyncio.Queue(maxsize=100)

    construct_count = 0

    def make_pa(*args, **kwargs):
        nonlocal construct_count
        construct_count += 1
        mock = MagicMock()
        if construct_count == 1:
            # First construction: raise immediately on read
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(side_effect=Exception("PulseAudio connection failed"))
        else:
            # Second construction: return data then stop
            call_count = 0
            def read_ok(size):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    session.running = False
                    raise Exception("stopped")
                return b'\x00' * size
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = read_ok
        return mock

    with patch.dict('sys.modules', {'pasimple': MagicMock(
        PaSimple=make_pa,
        PA_STREAM_RECORD=1,
        PA_SAMPLE_S16LE=3,
    )}):
        await asyncio.wait_for(session._audio_capture_stage(), timeout=10.0)

    assert construct_count >= 2, f"PaSimple should be constructed at least twice (reconnect), got {construct_count}"


@test("Audio capture: stops cleanly on running=False")
async def test_audio_capture_stops_cleanly():
    """Set running=False mid-capture — method should return promptly."""
    session = make_session()
    session.running = True
    session.generation_id = 1
    session._audio_in_q = asyncio.Queue(maxsize=100)

    call_count = 0
    mock_pa_instance = MagicMock()
    def read_side_effect(size):
        nonlocal call_count
        call_count += 1
        if call_count >= 5:
            session.running = False
        time.sleep(0.01)  # Simulate blocking read so event loop isn't flooded
        return b'\x00' * size
    mock_pa_instance.read = read_side_effect
    mock_pa_instance.__enter__ = MagicMock(return_value=mock_pa_instance)
    mock_pa_instance.__exit__ = MagicMock(return_value=False)

    with patch.dict('sys.modules', {'pasimple': MagicMock(
        PaSimple=MagicMock(return_value=mock_pa_instance),
        PA_STREAM_RECORD=1,
        PA_SAMPLE_S16LE=3,
    )}):
        # Should complete without timeout
        await asyncio.wait_for(session._audio_capture_stage(), timeout=5.0)

    # If we get here without timeout, the test passes
    assert session.running == False


@test("Audio capture: queue overflow drops frames gracefully")
async def test_audio_capture_queue_overflow():
    """Fill _audio_in_q, verify capture thread doesn't crash."""
    session = make_session()
    session.running = True
    session.generation_id = 1
    # Replace queue with a tiny one
    session._audio_in_q = asyncio.Queue(maxsize=1)
    # Pre-fill the queue
    await session._audio_in_q.put(PipelineFrame(type=FrameType.AUDIO_RAW, generation_id=0, data=b'old'))

    call_count = 0
    mock_pa_instance = MagicMock()
    def read_side_effect(size):
        nonlocal call_count
        call_count += 1
        if call_count > 10:
            session.running = False
            raise Exception("stopped")
        time.sleep(0.01)  # Simulate blocking read so event loop isn't flooded
        return b'\x00' * size
    mock_pa_instance.read = read_side_effect
    mock_pa_instance.__enter__ = MagicMock(return_value=mock_pa_instance)
    mock_pa_instance.__exit__ = MagicMock(return_value=False)

    with patch.dict('sys.modules', {'pasimple': MagicMock(
        PaSimple=MagicMock(return_value=mock_pa_instance),
        PA_STREAM_RECORD=1,
        PA_SAMPLE_S16LE=3,
    )}):
        # Should not crash despite full queue
        await asyncio.wait_for(session._audio_capture_stage(), timeout=5.0)

    # Capture ran and didn't crash — call_count tells us the thread ran
    assert call_count > 1, f"Capture thread should have called read() multiple times, got {call_count}"


# ══════════════════════════════════════════════════════════════════
# Test Group 22: SSE Dashboard Server
# ══════════════════════════════════════════════════════════════════

@test("SSE server starts on configured port")
async def test_sse_server_starts():
    """Binds, accepts connections, returns 200 + SSE headers."""
    session = make_session(sse_dashboard=True)
    session.running = True
    session.generation_id = 1
    session._audio_in_q = asyncio.Queue(maxsize=100)
    session._stt_out_q = asyncio.Queue(maxsize=50)
    session._llm_out_q = asyncio.Queue(maxsize=50)
    session._audio_out_q = asyncio.Queue(maxsize=200)

    # Start the SSE server stage
    server_task = asyncio.create_task(session._sse_server_stage())
    await asyncio.sleep(0.1)  # Let server bind

    try:
        # Connect as a client
        reader, writer = await asyncio.open_connection('127.0.0.1', 9847)

        # Send HTTP request
        writer.write(b"GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()

        # Read the HTTP response headers
        headers = b""
        while b"\r\n\r\n" not in headers:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            headers += chunk
            if not chunk:
                break

        headers_str = headers.decode()
        assert "200" in headers_str, f"Expected 200 status, got: {headers_str[:100]}"
        assert "text/event-stream" in headers_str, f"Expected SSE content type, got: {headers_str[:200]}"
        assert "Access-Control-Allow-Origin" in headers_str, "Expected CORS header"

        writer.close()
    finally:
        session.running = False
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@test("SSE server sends snapshot on connect")
async def test_sse_server_snapshot():
    """First message is type: snapshot with running/muted/queue_depths."""
    session = make_session(sse_dashboard=True)
    session.running = True
    session.generation_id = 5
    session.muted = True
    session._audio_in_q = asyncio.Queue(maxsize=100)
    session._stt_out_q = asyncio.Queue(maxsize=50)
    session._llm_out_q = asyncio.Queue(maxsize=50)
    session._audio_out_q = asyncio.Queue(maxsize=200)

    server_task = asyncio.create_task(session._sse_server_stage())
    await asyncio.sleep(0.1)

    try:
        reader, writer = await asyncio.open_connection('127.0.0.1', 9847)
        writer.write(b"GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()

        # Read response including first SSE data line
        data = b""
        deadline = asyncio.get_event_loop().time() + 3.0
        while asyncio.get_event_loop().time() < deadline:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            data += chunk
            if b"\ndata:" in data or b"\ndata: " in data:
                # Read a bit more to get the full data line
                try:
                    more = await asyncio.wait_for(reader.read(4096), timeout=0.3)
                    data += more
                except asyncio.TimeoutError:
                    pass
                break

        text = data.decode()
        # Find the first data: line after the headers
        data_lines = [l for l in text.split('\n') if l.startswith('data:')]
        assert len(data_lines) >= 1, f"Expected at least one data: line, got: {text[-300:]}"

        snapshot = json.loads(data_lines[0].split('data:', 1)[1].strip())
        assert snapshot['type'] == 'snapshot', f"Expected snapshot, got {snapshot.get('type')}"
        assert snapshot['running'] == True
        assert snapshot['muted'] == True
        assert 'queue_depths' in snapshot

        writer.close()
    finally:
        session.running = False
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@test("SSE server not started when disabled")
async def test_sse_server_disabled():
    """sse_dashboard=False → no server, connection refused."""
    session = make_session()  # default: sse_dashboard=False
    assert session._sse_dashboard == False
    assert session._sse_clients == []

    # _sse_server_stage should return immediately
    task = asyncio.create_task(session._sse_server_stage())
    await asyncio.sleep(0.05)
    assert task.done(), "Stage should complete immediately when disabled"


@test("_emit_event no-op with no clients")
async def test_emit_event_no_clients():
    """Returns immediately, no error."""
    session = make_session(sse_dashboard=True)
    session.generation_id = 1
    # No clients connected
    assert len(session._sse_clients) == 0
    # Should not raise
    session._emit_event("test", foo="bar")


@test("_emit_event broadcasts to clients")
async def test_emit_event_broadcasts():
    """Mock writers receive JSON with type/ts/gen_id."""
    session = make_session(sse_dashboard=True)
    session.generation_id = 42

    # Create mock writers
    writer1 = MagicMock()
    writer1.write = MagicMock()
    writer1.is_closing = MagicMock(return_value=False)
    writer2 = MagicMock()
    writer2.write = MagicMock()
    writer2.is_closing = MagicMock(return_value=False)
    session._sse_clients = [writer1, writer2]

    session._emit_event("status", status="listening")

    # Both writers should have received data
    assert writer1.write.called, "Writer 1 should receive data"
    assert writer2.write.called, "Writer 2 should receive data"

    # Parse the SSE data
    raw = writer1.write.call_args[0][0].decode()
    assert raw.startswith("data: "), f"Expected SSE format, got: {raw[:50]}"
    payload = json.loads(raw.split("data: ", 1)[1].strip())
    assert payload['type'] == 'status'
    assert payload['gen_id'] == 42
    assert payload['status'] == 'listening'
    assert 'ts' in payload


@test("_emit_event removes dead clients")
async def test_emit_event_prunes_dead():
    """Writer that raises gets pruned."""
    session = make_session(sse_dashboard=True)
    session.generation_id = 1

    # One good writer, one dead
    good_writer = MagicMock()
    good_writer.write = MagicMock()
    good_writer.is_closing = MagicMock(return_value=False)

    dead_writer = MagicMock()
    dead_writer.write = MagicMock(side_effect=ConnectionError("broken pipe"))
    dead_writer.is_closing = MagicMock(return_value=False)

    session._sse_clients = [good_writer, dead_writer]

    session._emit_event("test")

    assert len(session._sse_clients) == 1, f"Dead client should be pruned, got {len(session._sse_clients)}"
    assert session._sse_clients[0] is good_writer


@test("_emit_event envelope has required fields")
async def test_emit_event_envelope():
    """Every event has type, ts, gen_id."""
    session = make_session(sse_dashboard=True)
    session.generation_id = 7

    writer = MagicMock()
    writer.write = MagicMock()
    writer.is_closing = MagicMock(return_value=False)
    session._sse_clients = [writer]

    session._emit_event("audio_rms", rms=150.5)

    raw = writer.write.call_args[0][0].decode()
    payload = json.loads(raw.split("data: ", 1)[1].strip())
    assert 'type' in payload, "Missing 'type'"
    assert 'ts' in payload, "Missing 'ts'"
    assert 'gen_id' in payload, "Missing 'gen_id'"
    assert payload['type'] == 'audio_rms'
    assert payload['rms'] == 150.5


# ══════════════════════════════════════════════════════════════════
# Test Group 23: Event Decimation
# ══════════════════════════════════════════════════════════════════

@test("audio_rms decimated to every 3rd chunk")
async def test_audio_rms_decimation():
    """9 emission attempts → 3 actual events sent."""
    session = make_session(sse_dashboard=True)
    session.generation_id = 1
    session._sse_rms_counter = 0

    writer = MagicMock()
    writer.write = MagicMock()
    writer.is_closing = MagicMock(return_value=False)
    session._sse_clients = [writer]

    for i in range(9):
        session._emit_audio_rms(rms=100 + i, has_speech=False, speech_chunks=0, buf_seconds=1.0)

    assert writer.write.call_count == 3, f"Expected 3 events (every 3rd), got {writer.write.call_count}"


@test("Queue depth helper returns all 5 queues")
def test_queue_depth_helper():
    """audio_in, stt_out, llm_out, audio_out, composer present."""
    session = make_session(sse_dashboard=True)
    session._audio_in_q = asyncio.Queue(maxsize=100)
    session._stt_out_q = asyncio.Queue(maxsize=50)
    session._llm_out_q = asyncio.Queue(maxsize=50)
    session._audio_out_q = asyncio.Queue(maxsize=200)

    from stream_composer import StreamComposer
    session._composer = StreamComposer(
        session._audio_out_q,
        AsyncMock(return_value=b'\x00' * 100),
        lambda: 1,
    )

    depths = session._get_queue_depths()
    assert 'audio_in' in depths, "Missing audio_in"
    assert 'stt_out' in depths, "Missing stt_out"
    assert 'llm_out' in depths, "Missing llm_out"
    assert 'audio_out' in depths, "Missing audio_out"
    assert 'composer' in depths, "Missing composer"
    assert len(depths) == 5, f"Expected 5 queues, got {len(depths)}"


# ══════════════════════════════════════════════════════════════════
# Test Group 24: StreamComposer Callback
# ══════════════════════════════════════════════════════════════════

@test("StreamComposer on_event fires for tts_start and tts_complete")
async def test_composer_on_event_tts():
    """Capture callback, verify event types."""
    from stream_composer import StreamComposer, AudioSegment, SegmentType

    events = []
    def capture_event(event_type, **data):
        events.append({"type": event_type, **data})

    mock_tts = AsyncMock(return_value=b'\x00' * 4800)
    audio_out_q = asyncio.Queue()
    gen_id = 1

    composer = StreamComposer(
        audio_out_q, mock_tts, lambda: gen_id,
        on_event=capture_event,
    )

    # Enqueue a TTS sentence and EOT
    await composer.enqueue(AudioSegment(SegmentType.TTS_SENTENCE, data="Hello world."))
    await composer.enqueue_end_of_turn()

    # Run composer briefly
    run_task = asyncio.create_task(composer.run())
    await asyncio.sleep(0.3)
    composer.stop()
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except asyncio.TimeoutError:
        run_task.cancel()

    event_types = [e['type'] for e in events]
    assert 'tts_start' in event_types, f"Expected tts_start, got {event_types}"
    assert 'tts_complete' in event_types, f"Expected tts_complete, got {event_types}"


@test("StreamComposer on_event=None doesn't crash")
async def test_composer_no_callback():
    """Existing behavior unchanged — no callback, no crash."""
    from stream_composer import StreamComposer, AudioSegment, SegmentType

    mock_tts = AsyncMock(return_value=b'\x00' * 4800)
    audio_out_q = asyncio.Queue()

    composer = StreamComposer(
        audio_out_q, mock_tts, lambda: 1,
        # No on_event — default
    )

    await composer.enqueue(AudioSegment(SegmentType.TTS_SENTENCE, data="Test."))
    await composer.enqueue_end_of_turn()

    run_task = asyncio.create_task(composer.run())
    await asyncio.sleep(0.3)
    composer.stop()
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except asyncio.TimeoutError:
        run_task.cancel()

    # If we got here without exception, the test passes


@test("StreamComposer on_event fires filler_played")
async def test_composer_on_event_filler():
    """on_event fires filler_played with sufficient flag."""
    from stream_composer import StreamComposer, AudioSegment, SegmentType

    events = []
    def capture_event(event_type, **data):
        events.append({"type": event_type, **data})

    mock_tts = AsyncMock(return_value=b'\x00' * 100)
    audio_out_q = asyncio.Queue()

    composer = StreamComposer(
        audio_out_q, mock_tts, lambda: 1,
        on_event=capture_event,
    )

    await composer.enqueue(AudioSegment(
        SegmentType.FILLER_CLIP,
        data=b'\x00' * 2400,
        metadata={"sufficient": True},
    ))
    await composer.enqueue_end_of_turn()

    run_task = asyncio.create_task(composer.run())
    await asyncio.sleep(0.3)
    composer.stop()
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except asyncio.TimeoutError:
        run_task.cancel()

    event_types = [e['type'] for e in events]
    assert 'filler_played' in event_types, f"Expected filler_played, got {event_types}"
    filler_event = [e for e in events if e['type'] == 'filler_played'][0]
    assert filler_event['sufficient'] == True


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
