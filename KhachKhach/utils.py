"""
Utility functions for KhachKhach v2
"""

import os
from pathlib import Path
from typing import List

def validate_image_path(path: str) -> bool:
    """Validate if path is a valid image file"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(path).suffix.lower() in image_extensions

def create_output_structure(base_dir: str, subdirs: List[str]) -> dict:
    """Create organized output directory structure"""
    created_dirs = {}
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        created_dirs[subdir] = dir_path
    return created_dirs
