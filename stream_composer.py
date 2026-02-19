"""
StreamComposer: Unified audio sentence queue with natural cadence and barge-in awareness.

Sits between content producers (filler manager, LLM response reader) and the existing
playback stage. Accepts filler clips, TTS text sentences, silence, and non-speech segments
through a single queue, handling per-sentence TTS generation, inter-segment pauses,
pre-buffering, and barge-in drain.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Awaitable, Callable

from pipeline_frames import PipelineFrame, FrameType

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    FILLER_CLIP = auto()    # Pre-recorded filler PCM (acknowledgment clips)
    TTS_SENTENCE = auto()   # Text to synthesise via TTS callback
    SILENCE = auto()        # Explicit silence for a given duration
    NON_SPEECH = auto()     # Non-speech audio (thinking sounds, etc.)


@dataclass
class AudioSegment:
    """A single segment to play through the composer."""
    type: SegmentType
    data: bytes | str = b""       # PCM bytes for clips, text string for TTS
    duration: float = 0.0         # Duration in seconds (for SILENCE segments)
    metadata: dict = field(default_factory=dict)


# Sentinel used internally to mark end-of-turn in the queue.
_EOT_SENTINEL = None

# Sentinel indicating no lookahead segment is buffered.
_NO_LOOKAHEAD = object()


class StreamComposer:
    """Asyncio-based audio composer managing a unified segment queue.

    Public API
    ----------
    enqueue(segment)        -- add a segment to the queue
    enqueue_end_of_turn()   -- signal end of current turn
    pause() -> list         -- barge-in: stop output, return unplayed segments
    resume()                -- clear paused state
    reset()                 -- clear all state for a new turn
    run()                   -- main loop (long-running coroutine)
    """

    def __init__(
        self,
        audio_out_q: asyncio.Queue,
        tts_fn: Callable[[str], Awaitable[bytes | None]],
        get_generation_id: Callable[[], int],
        sample_rate: int = 24000,
    ):
        self._audio_out_q = audio_out_q
        self._tts_fn = tts_fn
        self._get_gen_id = get_generation_id
        self._sample_rate = sample_rate

        # Internal segment queue (AudioSegment | None sentinel)
        self._segment_q: asyncio.Queue[AudioSegment | None] = asyncio.Queue()

        # Barge-in / state
        self._paused = False
        self._held_segments: list[AudioSegment] = []
        self._current_gen_id: int = 0

        # Pre-buffering
        self._prefetch_task: asyncio.Task | None = None
        self._prefetch_text: str | None = None
        self._prefetch_result: bytes | None = None

        # Lookahead buffer for peek-ahead without reordering
        self._lookahead: AudioSegment | None | object = _NO_LOOKAHEAD

        # ── Cadence parameters (tunable) ──
        self.inter_sentence_pause: float = 0.15   # 150ms between sentences
        self.post_clip_pause: float = 0.25         # 250ms after filler clip
        self.thinking_pause: float = 0.4           # 400ms after non-speech

    # ── Public API ────────────────────────────────────────────────

    async def enqueue(self, segment: AudioSegment) -> None:
        """Add a segment to the playback queue."""
        await self._segment_q.put(segment)

    async def enqueue_end_of_turn(self) -> None:
        """Signal end of the current turn (None sentinel)."""
        await self._segment_q.put(_EOT_SENTINEL)

    def pause(self) -> list[AudioSegment]:
        """Barge-in: immediately pause output and return unplayed segments.

        This is intentionally synchronous (no await) so it can respond
        instantly to barge-in events.
        """
        self._paused = True

        # Cancel any in-flight prefetch
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            self._prefetch_task = None
            self._prefetch_text = None
            self._prefetch_result = None

        # Include lookahead segment if present
        lookahead_segs: list[AudioSegment] = []
        if self._lookahead is not _NO_LOOKAHEAD and self._lookahead is not _EOT_SENTINEL:
            lookahead_segs.append(self._lookahead)  # type: ignore[arg-type]
        self._lookahead = _NO_LOOKAHEAD

        # Drain remaining queued segments
        drained: list[AudioSegment] = []
        while True:
            try:
                item = self._segment_q.get_nowait()
                if item is not None:
                    drained.append(item)
            except asyncio.QueueEmpty:
                break

        unplayed = self._held_segments + lookahead_segs + drained
        self._held_segments = []
        return unplayed

    def resume(self) -> None:
        """Clear paused state so the run loop can continue."""
        self._paused = False

    def reset(self) -> None:
        """Clear all state for a new turn.

        Drains the queue, clears held segments, cancels prefetch, clears lookahead.
        """
        self._paused = False
        self._held_segments.clear()
        self._lookahead = _NO_LOOKAHEAD

        # Cancel prefetch
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
        self._prefetch_task = None
        self._prefetch_text = None
        self._prefetch_result = None

        # Drain segment queue
        while True:
            try:
                self._segment_q.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Main loop ─────────────────────────────────────────────────

    async def _next_segment(self) -> AudioSegment | None:
        """Get the next segment, consuming lookahead buffer first."""
        if self._lookahead is not _NO_LOOKAHEAD:
            seg = self._lookahead
            self._lookahead = _NO_LOOKAHEAD
            return seg  # type: ignore[return-value]
        return await self._segment_q.get()

    async def run(self) -> None:
        """Main composer loop -- runs forever, processing segments."""
        while True:
            segment = await self._next_segment()

            # End-of-turn sentinel
            if segment is _EOT_SENTINEL:
                gen_id = self._get_gen_id()
                await self._audio_out_q.put(PipelineFrame(
                    type=FrameType.END_OF_TURN,
                    generation_id=gen_id,
                ))
                continue

            # If paused (barge-in), hold segment for later
            if self._paused:
                self._held_segments.append(segment)
                continue

            gen_id = self._get_gen_id()

            # Stale segment detection
            if self._current_gen_id != 0 and gen_id != self._current_gen_id:
                # Generation changed -- discard stale segment
                continue
            self._current_gen_id = gen_id

            # ── Pre-buffering: peek at the next segment ──
            # Use lookahead buffer to avoid reordering the queue.
            # We get the next item, inspect it for prefetch, then
            # store it in _lookahead so it's consumed next iteration.
            if self._lookahead is _NO_LOOKAHEAD:
                next_seg = self._try_get_next()
                if next_seg is not _NO_LOOKAHEAD:
                    self._lookahead = next_seg
                    if (
                        next_seg is not _EOT_SENTINEL
                        and next_seg is not None
                        and next_seg.type == SegmentType.TTS_SENTENCE
                        and isinstance(next_seg.data, str)
                    ):
                        text = next_seg.data
                        if self._prefetch_text != text:
                            if self._prefetch_task and not self._prefetch_task.done():
                                self._prefetch_task.cancel()
                            self._prefetch_text = text
                            self._prefetch_result = None
                            self._prefetch_task = asyncio.create_task(
                                self._do_prefetch(text)
                            )

            # ── Process the current segment ──
            if segment.type == SegmentType.FILLER_CLIP:
                await self._process_filler_clip(segment, gen_id)

            elif segment.type == SegmentType.TTS_SENTENCE:
                await self._process_tts_sentence(segment, gen_id)

            elif segment.type == SegmentType.SILENCE:
                await self._emit_silence(segment.duration, gen_id)

            elif segment.type == SegmentType.NON_SPEECH:
                await self._process_non_speech(segment, gen_id)

    # ── Segment processors ────────────────────────────────────────

    async def _process_filler_clip(self, segment: AudioSegment, gen_id: int) -> None:
        """Emit filler clip PCM followed by post-clip pause."""
        if isinstance(segment.data, bytes) and segment.data:
            await self._emit_pcm(segment.data, FrameType.FILLER, gen_id)
        await self._emit_silence(self.post_clip_pause, gen_id)

    async def _process_tts_sentence(self, segment: AudioSegment, gen_id: int) -> None:
        """Generate TTS for a sentence, emit audio + SENTENCE_DONE + pause."""
        text = segment.data
        if not isinstance(text, str) or not text.strip():
            return

        pcm: bytes | None = None

        # Check if pre-buffered result is available for this text
        if self._prefetch_text == text and self._prefetch_task is not None:
            if self._prefetch_task.done():
                pcm = self._prefetch_result
            else:
                # Wait for the prefetch to finish
                try:
                    await self._prefetch_task
                    pcm = self._prefetch_result
                except (asyncio.CancelledError, Exception):
                    pcm = None

            # Clear prefetch state
            self._prefetch_task = None
            self._prefetch_text = None
            self._prefetch_result = None

        # Fallback: generate TTS now
        if pcm is None:
            try:
                pcm = await self._tts_fn(text)
            except Exception as e:
                logger.error("TTS generation failed for %r: %s", text[:40], e)
                pcm = None

        if pcm:
            await self._emit_pcm(pcm, FrameType.TTS_AUDIO, gen_id)

        # Emit sentence-done marker
        if self._get_gen_id() == gen_id and not self._paused:
            await self._audio_out_q.put(PipelineFrame(
                type=FrameType.SENTENCE_DONE,
                generation_id=gen_id,
            ))

        await self._emit_silence(self.inter_sentence_pause, gen_id)

    async def _process_non_speech(self, segment: AudioSegment, gen_id: int) -> None:
        """Emit non-speech audio (thinking sounds) followed by thinking pause."""
        if isinstance(segment.data, bytes) and segment.data:
            await self._emit_pcm(segment.data, FrameType.FILLER, gen_id)
        await self._emit_silence(self.thinking_pause, gen_id)

    # ── Internal helpers ──────────────────────────────────────────

    async def _emit_pcm(self, pcm: bytes, frame_type: FrameType, gen_id: int) -> None:
        """Chunk PCM into 4096-byte pieces and write to audio_out_q.

        Checks generation_id and _paused before each chunk write;
        returns early if either condition triggers.
        """
        offset = 0
        while offset < len(pcm):
            if self._paused or self._get_gen_id() != gen_id:
                return
            chunk = pcm[offset:offset + 4096]
            offset += 4096
            await self._audio_out_q.put(PipelineFrame(
                type=frame_type,
                generation_id=gen_id,
                data=chunk,
            ))

    async def _emit_silence(self, duration: float, gen_id: int) -> None:
        """Generate and emit silence bytes for the given duration."""
        if duration <= 0:
            return
        num_bytes = int(self._sample_rate * 2 * duration)  # 16-bit PCM
        silence = b"\x00" * num_bytes
        await self._emit_pcm(silence, FrameType.TTS_AUDIO, gen_id)

    def _try_get_next(self) -> AudioSegment | None | object:
        """Non-blocking get of the next segment from the queue.

        Returns the segment/sentinel if available, or _NO_LOOKAHEAD if
        the queue is empty. The caller stores the result in _lookahead
        so it is consumed by _next_segment() on the next iteration,
        preserving FIFO order.
        """
        try:
            return self._segment_q.get_nowait()
        except asyncio.QueueEmpty:
            return _NO_LOOKAHEAD

    async def _do_prefetch(self, text: str) -> None:
        """Background task to pre-generate TTS for a sentence."""
        try:
            result = await self._tts_fn(text)
            self._prefetch_result = result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Prefetch TTS failed for %r: %s", text[:40], e)
            self._prefetch_result = None
