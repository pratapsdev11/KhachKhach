"""
KhachKhach v2 - Advanced Video Annotation Library

Updated with user-friendly coordinate normalization and clean output format!
"""

from .core.video_processor import VideoProcessor
from .core.detection_engine import DetectionEngine
from .core.data_processor import DataProcessor

__version__ = "2.0.0"
__author__ = "Divya"

def annotate_keypoints(input_path, output_dir, model_path="yolov8n-pose.pt", confidence=0.5, 
                      save_images=False, normalize_coords=True, output_format="comma_separated"):
    """
    Annotate keypoints using YOLO pose model
    
    Args:
        normalize_coords (bool): If True, coordinates will be 0-1 range. If False, pixel coordinates
        output_format (str): "comma_separated" or "space_separated" format
        
    Output Examples:
        comma_separated: 0.618000,0.312658,1.000000,0.633144,0.300665,1.000000
        space_separated: 0.618000 0.312658 1.000000 0.633144 0.300665 1.000000
    """
    engine = DetectionEngine(model_path)
    return engine.annotate_keypoints(input_path, output_dir, confidence, save_images, normalize_coords, output_format)

# Main API functions for easy access
def extract_frames(video_path, output_dir, frame_interval=1, start_time=None, end_time=None, max_frames=None):
    processor = VideoProcessor()
    return processor.extract_frames(video_path, output_dir, frame_interval, start_time, end_time, max_frames)

def detect_objects(input_path, output_dir, model_path="yolov8n.pt", confidence=0.5, save_images=False):
    engine = DetectionEngine(model_path)
    return engine.detect_objects(input_path, output_dir, confidence, save_images)



def process_annotations(input_dir, output_dir, format_type="yolo", processing_mode="extract_xyn"):
    processor = DataProcessor()
    return processor.process_annotations(input_dir, output_dir, format_type, processing_mode)

def get_video_info(video_path):
    processor = VideoProcessor()
    return processor.get_video_info(video_path)

def full_video_pipeline(video_path, output_base_dir, **kwargs):
    import os
    frames_dir = os.path.join(output_base_dir, "frames")
    objects_dir = os.path.join(output_base_dir, "object_annotations")
    keypoints_dir = os.path.join(output_base_dir, "keypoint_annotations")
    
    frame_result = extract_frames(video_path, frames_dir, **kwargs.get('extract_params', {}))
    object_result = detect_objects(frames_dir, objects_dir, **kwargs.get('object_params', {}))
    keypoint_result = annotate_keypoints(frames_dir, keypoints_dir, **kwargs.get('keypoint_params', {}))
    
    return {
        "pipeline_success": True,
        "frames": frame_result,
        "objects": object_result,
        "keypoints": keypoint_result
    }
