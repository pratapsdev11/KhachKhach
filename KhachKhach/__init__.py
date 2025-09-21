"""
Core modules for KhachKhach v2 library.
"""

__version__ = "2.0.0"

from .data_processor import *
from .detection_engine import *
from .utils import *
from .video_processor import *


__all__ = [
    'VideoProcessor',
    'DetectionEngine', 
    'DataProcessor'
]
