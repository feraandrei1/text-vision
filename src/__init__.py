"""
TextIdentifier - Unified Text Identification System

Supports:
- License plate recognition
- ID card/passport text extraction
- PDF and document text extraction
"""

from .text_identifier import TextIdentifier
from .plates import PlateRecognizer
from .ids import IDRecognizer
from .documents import DocumentReader
from .video_stream import VideoStream

# Legacy support
from .platevision import PlateVision

__version__ = "2.0.0"
__all__ = [
    "TextIdentifier",
    "PlateRecognizer",
    "IDRecognizer",
    "DocumentReader",
    "VideoStream",
    "PlateVision",  # Legacy
]
