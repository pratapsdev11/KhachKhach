"""
Video Processing Module - Consolidates frame_extractor.py functionality
"""

import os
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        pass

    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        frame_interval: int = 1,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None,
        image_format: str = "jpg",
        quality: int = 95
    ) -> Dict[str, Any]:
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Handle time-based extraction
            start_frame = 0
            if start_time is not None:
                start_frame = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            end_frame = total_frames
            if end_time is not None:
                end_frame = min(int(end_time * fps), total_frames)

            extracted_count = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                if current_frame > end_frame:
                    break

                if (current_frame - start_frame) % frame_interval == 0:
                    filename = f"frame_{extracted_count:06d}.{image_format.lower()}"
                    filepath = os.path.join(output_dir, filename)
                    
                    if cv2.imwrite(filepath, frame):
                        extracted_count += 1
                        
                        if max_frames and extracted_count >= max_frames:
                            break

                processed_frames += 1

            cap.release()

            return {
                "success": True,
                "total_frames_extracted": extracted_count,
                "output_directory": output_dir,
                "video_properties": {
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": duration,
                    "width": width,
                    "height": height
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_frames_extracted": 0
            }

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": 0
            }
            
            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]

            cap.release()
            return {"success": True, **info}

        except Exception as e:
            return {"success": False, "error": str(e)}
