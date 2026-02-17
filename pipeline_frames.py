"""Typed dataclass frames that flow between pipeline stages via asyncio.Queue."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class FrameType(Enum):
    AUDIO_RAW = auto()       # Raw PCM audio from mic
    TRANSCRIPT = auto()      # Final transcript from STT
    TEXT_DELTA = auto()       # Streaming text chunk from LLM
    TOOL_CALL = auto()        # Tool invocation from LLM
    TOOL_RESULT = auto()      # Result from tool execution
    TTS_AUDIO = auto()        # PCM audio from TTS
    END_OF_TURN = auto()      # LLM finished its turn
    END_OF_UTTERANCE = auto() # STT detected end of user speech
    FILLER = auto()           # Filler audio (acknowledgment/thinking)
    BARGE_IN = auto()         # Barge-in signal (user interrupted AI)
    CONTROL = auto()          # Control signals (stop, interrupt, etc.)


@dataclass
class PipelineFrame:
    type: FrameType
    generation_id: int = 0
    data: Any = None
    metadata: dict = field(default_factory=dict)
