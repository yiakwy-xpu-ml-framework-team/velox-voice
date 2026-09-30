"""VeloxVoice public API."""

from .api import StreamingSession, TranscriptionResult, Velox
from .runtime.device import PlatformInfo, detect_platform, get_backend

__all__ = [
    "Velox",
    "StreamingSession",
    "TranscriptionResult",
    "PlatformInfo",
    "detect_platform",
    "get_backend",
]
__version__ = "0.1.0"
