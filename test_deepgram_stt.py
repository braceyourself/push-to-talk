#!/usr/bin/env python3
"""Tests for DeepgramSTT -- streaming STT with mock WebSocket.

Tests the DeepgramSTT class behavior using mock objects for the Deepgram SDK.
No real WebSocket connections or API keys are used.

Run: python3 test_deepgram_stt.py
"""

import asyncio
import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent))

from transcript_buffer import TranscriptBuffer, TranscriptSegment, is_hallucination

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


# ── Helper: Build mock Deepgram result events ────────────────────

def make_result_event(transcript, is_final=False, speech_final=False):
    """Create a mock ListenV1ResultsEvent with the given transcript fields."""
    from deepgram.extensions.types.sockets.listen_v1_results_event import (
        ListenV1ResultsEvent,
    )

    event = ListenV1ResultsEvent(
        type="Results",
        channel_index=[0],
        duration=1.0,
        start=0.0,
        is_final=is_final,
        speech_final=speech_final,
        channel={
            "alternatives": [
                {
                    "transcript": transcript,
                    "confidence": 0.99,
                    "words": [],
                }
            ]
        },
        metadata={
            "request_id": "test-req-id",
            "model_info": {"name": "nova-3", "version": "2024-01-01", "arch": "nova"},
            "model_uuid": "test-uuid",
        },
    )
    return event


def make_empty_result_event(is_final=True, speech_final=False):
    """Create a result event with empty transcript text."""
    return make_result_event("", is_final=is_final, speech_final=speech_final)


# ── Helper: Create a DeepgramSTT instance with mocked internals ──

def make_stt(transcript_q=None, on_segment=None, on_stats=None,
             on_unavailable=None):
    """Create a DeepgramSTT with mocked audio capture and connection."""
    from deepgram_stt import DeepgramSTT

    tb = TranscriptBuffer()
    stt = DeepgramSTT(
        api_key="test-key",
        transcript_buffer=tb,
        transcript_q=transcript_q,
        aec_device_name=None,
        on_segment=on_segment,
        on_stats=on_stats,
        on_unavailable=on_unavailable,
    )
    return stt, tb


# ══════════════════════════════════════════════════════════════════
# Test Group 1: Transcript Accumulation
# ══════════════════════════════════════════════════════════════════

@test("Transcript accumulation: speech_final flushes accumulated finals")
def test_transcript_accumulation_speech_final():
    """Simulate on_message callbacks with is_final segments followed by
    speech_final. Verify accumulated text is joined and emitted."""
    stt, tb = make_stt()

    # Simulate three is_final=True (not speech_final) events
    evt1 = make_result_event("Hello there", is_final=True, speech_final=False)
    evt2 = make_result_event("how are you", is_final=True, speech_final=False)
    # Then speech_final=True
    evt3 = make_result_event("today", is_final=True, speech_final=True)

    stt._on_message(evt1)
    stt._on_message(evt2)
    stt._on_message(evt3)

    # Should have emitted exactly one segment with joined text
    assert len(tb) == 1, f"Expected 1 segment, got {len(tb)}"
    segments = tb.get_since(0)
    assert segments[0].text == "Hello there how are you today", \
        f"Unexpected text: {segments[0].text}"


@test("Interim results (is_final=False) are not emitted")
def test_interim_results_not_emitted():
    """Interim results should NOT produce TranscriptSegments."""
    stt, tb = make_stt()

    # Send interim results (is_final=False)
    evt1 = make_result_event("Hel", is_final=False, speech_final=False)
    evt2 = make_result_event("Hello", is_final=False, speech_final=False)
    evt3 = make_result_event("Hello there", is_final=False, speech_final=False)

    stt._on_message(evt1)
    stt._on_message(evt2)
    stt._on_message(evt3)

    # Nothing should be emitted to TranscriptBuffer
    assert len(tb) == 0, f"Expected 0 segments, got {len(tb)}"
    # Accumulator should be empty (interims don't accumulate)
    assert len(stt._accumulated_finals) == 0, \
        "Interim results should not accumulate"


@test("Speech final flushes all accumulated is_final segments")
def test_speech_final_flushes_accumulated():
    """Send 3 is_final=True then 1 speech_final=True. Verify joined text."""
    stt, tb = make_stt()

    stt._on_message(make_result_event("I want to", is_final=True, speech_final=False))
    stt._on_message(make_result_event("check the weather", is_final=True, speech_final=False))
    stt._on_message(make_result_event("in Seattle", is_final=True, speech_final=True))

    assert len(tb) == 1
    seg = tb.get_since(0)[0]
    assert seg.text == "I want to check the weather in Seattle"


# ══════════════════════════════════════════════════════════════════
# Test Group 2: Hallucination Filter
# ══════════════════════════════════════════════════════════════════

@test("Hallucination phrases are rejected and not appended to buffer")
def test_hallucination_rejected():
    """Known hallucination phrases should be filtered."""
    stt, tb = make_stt()

    # Send hallucination phrases as speech_final
    stt._on_message(make_result_event("thank you", is_final=True, speech_final=True))
    stt._on_message(make_result_event("thanks for watching", is_final=True, speech_final=True))

    assert len(tb) == 0, f"Hallucinations should not reach buffer, got {len(tb)}"
    assert stt._hallucination_count == 2, \
        f"Expected 2 hallucination count, got {stt._hallucination_count}"


# ══════════════════════════════════════════════════════════════════
# Test Group 3: Playback Suppression
# ══════════════════════════════════════════════════════════════════

@test("Playback suppression gates transcripts during and after TTS")
def test_playback_suppression():
    """Transcripts suppressed during playback and cooldown period."""
    stt, tb = make_stt()

    # During playback: suppressed
    stt.set_playing_audio(True)
    stt._on_message(make_result_event("AI echo text", is_final=True, speech_final=True))
    assert len(tb) == 0, "Should suppress during playback"

    # Immediately after playback ends: still in cooldown
    stt.set_playing_audio(False)
    stt._on_message(make_result_event("Residual echo", is_final=True, speech_final=True))
    assert len(tb) == 0, "Should suppress during cooldown"

    # After cooldown expires
    stt._playback_end_time = time.time() - 1.0  # Force cooldown expired
    stt._on_message(make_result_event("Real speech after cooldown", is_final=True, speech_final=True))
    assert len(tb) == 1, "Should emit after cooldown"
    assert tb.get_since(0)[0].text == "Real speech after cooldown"


# ══════════════════════════════════════════════════════════════════
# Test Group 4: KeepAlive
# ══════════════════════════════════════════════════════════════════

@test("KeepAlive sent when no audio for KEEPALIVE_INTERVAL")
async def test_keepalive_sent_on_silence():
    """Verify KeepAlive is called on the connection when idle."""
    from deepgram_stt import DeepgramSTT, KEEPALIVE_INTERVAL
    from deepgram.extensions.types.sockets.listen_v1_control_message import (
        ListenV1ControlMessage,
    )

    stt, tb = make_stt()

    # Mock the connection object
    mock_conn = MagicMock()
    stt._dg_connection = mock_conn
    stt._connected = True
    stt._running = True

    # Set last audio sent time to be beyond the keepalive interval
    stt._last_audio_sent_time = time.time() - KEEPALIVE_INTERVAL - 1.0

    # Call the keepalive check method
    stt._maybe_send_keepalive()

    # Verify send_control was called with a KeepAlive message
    mock_conn.send_control.assert_called_once()
    ka_msg = mock_conn.send_control.call_args[0][0]
    assert ka_msg.type == "KeepAlive", f"Expected KeepAlive, got {ka_msg.type}"


# ══════════════════════════════════════════════════════════════════
# Test Group 5: Reconnection
# ══════════════════════════════════════════════════════════════════

@test("Reconnection triggered on send error")
def test_reconnection_on_error():
    """When send_media raises, connection state transitions to disconnected."""
    stt, tb = make_stt()

    mock_conn = MagicMock()
    mock_conn.send_media.side_effect = Exception("WebSocket closed")
    stt._dg_connection = mock_conn
    stt._connected = True

    # Attempt to send audio -- should handle error
    stt._try_send_audio(b'\x00' * 4096)

    assert stt._connected == False, "Should be disconnected after send error"


# ══════════════════════════════════════════════════════════════════
# Test Group 6: Transcript Queue Output
# ══════════════════════════════════════════════════════════════════

@test("Completed transcript put on transcript_q when provided")
def test_transcript_queue_output():
    """When transcript_q is provided, segments are put on the queue."""
    q = asyncio.Queue()
    stt, tb = make_stt(transcript_q=q)

    stt._on_message(make_result_event("Hello world", is_final=True, speech_final=True))

    assert not q.empty(), "Transcript should be on the queue"
    segment = q.get_nowait()
    assert segment.text == "Hello world"
    assert len(tb) == 1


# ══════════════════════════════════════════════════════════════════
# Test Group 7: Stats
# ══════════════════════════════════════════════════════════════════

@test("Stats property returns expected keys")
def test_stats_property():
    """Verify stats dict contains all required keys."""
    stt, tb = make_stt()

    stats = stt.stats
    required_keys = {
        'segment_count', 'hallucination_count', 'avg_latency_ms',
        'buffer_depth', 'connected', 'reconnect_attempts',
    }
    missing = required_keys - set(stats.keys())
    assert not missing, f"Missing stats keys: {missing}"

    assert stats['segment_count'] == 0
    assert stats['connected'] == False
    assert stats['reconnect_attempts'] == 0


# ══════════════════════════════════════════════════════════════════
# Test Group 8: Empty Transcript
# ══════════════════════════════════════════════════════════════════

@test("Empty transcript text is ignored")
def test_empty_transcript_ignored():
    """is_final=True with empty text should not emit anything."""
    stt, tb = make_stt()

    stt._on_message(make_empty_result_event(is_final=True, speech_final=False))
    stt._on_message(make_empty_result_event(is_final=True, speech_final=True))

    assert len(tb) == 0, "Empty transcripts should not produce segments"
    assert len(stt._accumulated_finals) == 0


# ══════════════════════════════════════════════════════════════════
# Test Group 9: on_unavailable Callback
# ══════════════════════════════════════════════════════════════════

@test("on_unavailable called after max reconnect attempts exhausted")
async def test_on_unavailable_called_after_max_reconnects():
    """After MAX_RECONNECT_ATTEMPTS failures, on_unavailable fires."""
    from deepgram_stt import DeepgramSTT, MAX_RECONNECT_ATTEMPTS

    unavailable_called = []
    stt, tb = make_stt(on_unavailable=lambda: unavailable_called.append(True))

    # Simulate exhausted reconnects
    stt._reconnect_attempts = MAX_RECONNECT_ATTEMPTS

    # Call the method that checks and triggers the callback
    stt._check_reconnect_exhausted()

    assert len(unavailable_called) == 1, \
        f"Expected on_unavailable called once, got {len(unavailable_called)}"


# ══════════════════════════════════════════════════════════════════
# Test Group 10: on_segment Callback
# ══════════════════════════════════════════════════════════════════

@test("on_segment callback fires for valid transcripts")
def test_on_segment_callback():
    """Verify on_segment is called with TranscriptSegment for valid text."""
    segments_received = []
    stt, tb = make_stt(on_segment=lambda seg: segments_received.append(seg))

    stt._on_message(make_result_event("Test speech", is_final=True, speech_final=True))

    assert len(segments_received) == 1
    assert segments_received[0].text == "Test speech"
    assert segments_received[0].source == "user"


@test("on_segment callback does NOT fire for hallucinations")
def test_on_segment_not_for_hallucinations():
    """Hallucinations should not trigger on_segment callback."""
    segments_received = []
    stt, tb = make_stt(on_segment=lambda seg: segments_received.append(seg))

    stt._on_message(make_result_event("thank you", is_final=True, speech_final=True))

    assert len(segments_received) == 0, "Hallucinations should not trigger on_segment"


# ══════════════════════════════════════════════════════════════════
# Test Group 11: Echo Suppression
# ══════════════════════════════════════════════════════════════════

@test("Echo exact match is filtered and not emitted")
def test_echo_exact_match_filtered():
    """Set recent AI speech, send identical text as speech_final.
    Verify it is filtered (not emitted to TranscriptBuffer)."""
    stt, tb = make_stt()

    # Set recent AI speech
    stt.set_recent_ai_speech(["Hello, how can I help you?"])

    # Send a speech_final matching the AI speech exactly
    stt._on_message(make_result_event(
        "Hello, how can I help you?", is_final=True, speech_final=True
    ))

    assert len(tb) == 0, f"Echo should be filtered, but got {len(tb)} segments"


@test("Echo fuzzy match is filtered (partial match above threshold)")
def test_echo_fuzzy_match_filtered():
    """Set recent AI speech, send partial/similar text.
    Verify it is filtered with fuzzy matching (>= 70% similarity)."""
    stt, tb = make_stt()

    stt.set_recent_ai_speech(["I'd be happy to help with that"])

    # Partial match -- missing last word but still very similar
    stt._on_message(make_result_event(
        "I'd be happy to help with", is_final=True, speech_final=True
    ))

    assert len(tb) == 0, f"Fuzzy echo should be filtered, but got {len(tb)} segments"


@test("Echo case-insensitive matching filters different casing/punctuation")
def test_echo_case_insensitive():
    """Set recent AI speech with specific casing. Send text with different
    casing and slight punctuation differences. Verify filtered."""
    stt, tb = make_stt()

    stt.set_recent_ai_speech(["The answer is forty-two"])

    # Different case, different hyphenation
    stt._on_message(make_result_event(
        "the answer is forty two", is_final=True, speech_final=True
    ))

    assert len(tb) == 0, f"Case-insensitive echo should be filtered, got {len(tb)} segments"


@test("Non-echo transcript passes through (not filtered)")
def test_non_echo_not_filtered():
    """Set recent AI speech about weather. Send unrelated transcript.
    Verify it passes through (NOT filtered)."""
    stt, tb = make_stt()

    stt.set_recent_ai_speech(["The weather is nice today"])

    stt._on_message(make_result_event(
        "Can you check my calendar?", is_final=True, speech_final=True
    ))

    assert len(tb) == 1, f"Non-echo should pass through, got {len(tb)} segments"
    assert tb.get_since(0)[0].text == "Can you check my calendar?"


@test("Echo timing window expires -- old AI speech no longer filtered")
def test_echo_timing_window():
    """Set recent AI speech, then advance time beyond the echo window.
    Matching text should pass through because the window expired."""
    from deepgram_stt import ECHO_WINDOW_SECONDS

    stt, tb = make_stt()

    stt.set_recent_ai_speech(["Hello, how can I help you?"])

    # Manually expire the echo fingerprints by backdating their timestamps
    stt._echo_fingerprints = [
        (text, timestamp - ECHO_WINDOW_SECONDS - 1.0)
        for text, timestamp in stt._echo_fingerprints
    ]

    # Now send matching text -- should pass through (window expired)
    stt._on_message(make_result_event(
        "Hello, how can I help you?", is_final=True, speech_final=True
    ))

    assert len(tb) == 1, f"Expired echo window should not filter, got {len(tb)} segments"


@test("Echo during playback is suppressed by both playback gating and echo filter")
def test_echo_during_playback_suppressed():
    """Set playing_audio=True AND set recent AI speech.
    Verify transcript is suppressed (belt and suspenders)."""
    stt, tb = make_stt()

    stt.set_recent_ai_speech(["The current time is three o'clock"])
    stt.set_playing_audio(True)

    stt._on_message(make_result_event(
        "The current time is three o'clock", is_final=True, speech_final=True
    ))

    assert len(tb) == 0, "Should be suppressed by playback gating AND echo filter"


@test("set_recent_ai_speech updates and replaces echo buffer")
def test_set_recent_ai_speech_updates_buffer():
    """Call set_recent_ai_speech with initial sentences, verify buffer updated.
    Call again with new sentences, verify old ones are replaced."""
    stt, tb = make_stt()

    # Set initial sentences
    stt.set_recent_ai_speech(["sentence one", "sentence two"])
    assert len(stt._echo_fingerprints) == 2, \
        f"Expected 2 fingerprints, got {len(stt._echo_fingerprints)}"

    # Verify the texts are normalized (lowercased)
    texts = [fp[0] for fp in stt._echo_fingerprints]
    assert "sentence one" in texts
    assert "sentence two" in texts

    # Replace with new sentences
    stt.set_recent_ai_speech(["new sentence alpha", "new sentence beta", "new sentence gamma"])
    assert len(stt._echo_fingerprints) == 3, \
        f"Expected 3 fingerprints after update, got {len(stt._echo_fingerprints)}"

    # Old sentences should be gone
    texts = [fp[0] for fp in stt._echo_fingerprints]
    assert "sentence one" not in texts, "Old fingerprints should be replaced"
    assert "new sentence alpha" in texts


# ══════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DeepgramSTT Unit Tests")
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
