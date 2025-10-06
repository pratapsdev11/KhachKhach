#v2.0.1
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import logging
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics is required. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    Detection Engine for YOLO-based object detection and keypoint annotation.
    
    Features:
    - Object detection with bounding boxes
    - Keypoint detection with pose estimation
    - Bounding box + keypoint combined output
    - Space-separated format by default
    - Normalized coordinates (0-1 range)
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the detection engine with a YOLO model."""
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.is_pose_model = 'pose' in model_path.lower()
            logger.info(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_path}: {str(e)}")
            raise

    def detect_objects(
        self,
        input_path: Union[str, List[str]],
        output_dir: str,
        confidence: float = 0.5,
        save_images: bool = False
    ) -> Dict[str, Any]:
        """Detect objects in images and save bounding box annotations."""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            image_paths = self._get_image_paths(input_path)
            
            if not image_paths:
                return {"success": False, "error": "No valid images found"}
            
            results_summary = {
                "success": True,
                "total_images": len(image_paths),
                "processed_images": 0,
                "total_detections": 0,
                "failed_images": []
            }
            
            for image_path in image_paths:
                try:
                    results = self.model(image_path, conf=confidence, verbose=False)
                    
                    annotation_filename = Path(image_path).stem + ".txt"
                    annotation_path = os.path.join(output_dir, annotation_filename)
                    
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            boxes = result.boxes
                            for i in range(len(boxes)):
                                box_data = boxes.xywhn[i].cpu().numpy()
                                x_center, y_center, width, height = box_data
                                class_id = int(boxes.cls[i].cpu().numpy())
                                detection_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                                detections.append(detection_line)
                    
                    with open(annotation_path, "w") as f:
                        f.write("\n".join(detections))
                    
                    results_summary["processed_images"] += 1
                    results_summary["total_detections"] += len(detections)
                    
                    if save_images and detections:
                        annotated_results = results[0].plot()
                        cv2.imwrite(
                            os.path.join(output_dir, f"annotated_{Path(image_path).name}"),
                            annotated_results
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {str(e)}")
                    results_summary["failed_images"].append(str(image_path))
            
            return results_summary
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def annotate_keypoints(
        self,
        input_path: Union[str, List[str]],
        output_dir: str,
        confidence: float = 0.5,
        save_images: bool = False,
        normalize_coords: bool = True,
        output_format: str = "space_separated"
    ) -> Dict[str, Any]:
        """
        Annotate keypoints with bounding box information.
        
        Output format: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...
        
        Total: 56 values per person (1 class_id + 4 bbox + 51 keypoints)
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            image_paths = self._get_image_paths(input_path)
            
            if not image_paths:
                return {"success": False, "error": "No valid images found"}
            
            results_summary = {
                "success": True,
                "total_images": len(image_paths),
                "processed_images": 0,
                "total_persons": 0,
                "failed_images": []
            }
            
            sep = "," if output_format == "comma_separated" else " "
            
            for image_path in image_paths:
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Could not load image: {image_path}")
                    
                    h, w = image.shape[:2]
                    results = self.model(image, conf=confidence, verbose=False)
                    
                    annotation_filename = Path(image_path).stem + ".txt"
                    annotation_path = os.path.join(output_dir, annotation_filename)
                    
                    keypoint_lines = []
                    
                    for result in results:
                        has_boxes = hasattr(result, "boxes") and result.boxes is not None
                        has_keypoints = hasattr(result, "keypoints") and result.keypoints is not None
                        
                        if has_boxes and has_keypoints:
                            boxes = result.boxes
                            keypoints_data = result.keypoints
                            
                            num_detections = len(boxes)
                            
                            for i in range(num_detections):
                                # Bounding box (4 values)
                                box_data = boxes.xywhn[i].cpu().numpy()
                                x_center, y_center, width, height = box_data
                                class_id = int(boxes.cls[i].cpu().numpy())
                                
                                line_parts = [
                                    str(class_id),
                                    f"{x_center:.6f}",
                                    f"{y_center:.6f}",
                                    f"{width:.6f}",
                                    f"{height:.6f}"
                                ]
                                
                                # Keypoints (51 values = 17 × 3)
                                if i < len(keypoints_data):
                                    keypoint = keypoints_data[i]
                                    
                                    if keypoint.xy is not None and keypoint.conf is not None:
                                        kpts = keypoint.xy.cpu().numpy()
                                        confs = keypoint.conf.cpu().numpy()
                                        
                                        if len(kpts.shape) == 3:
                                            kpts = kpts[0]
                                        if len(confs.shape) == 2:
                                            confs = confs[0]
                                        
                                        for (x, y), v in zip(kpts, confs):
                                            if normalize_coords:
                                                x_out = x / w
                                                y_out = y / h
                                            else:
                                                x_out, y_out = x, y
                                            
                                            v_flag = 2 if v > 0.5 else 0
                                            
                                            line_parts.extend([
                                                f"{x_out:.6f}",
                                                f"{y_out:.6f}",
                                                str(v_flag)
                                            ])
                                
                                line = sep.join(line_parts)
                                keypoint_lines.append(line)
                    
                    with open(annotation_path, "w") as f:
                        f.write("\n".join(keypoint_lines))
                    
                    results_summary["processed_images"] += 1
                    results_summary["total_persons"] += len(keypoint_lines)
                    
                    if save_images and keypoint_lines:
                        annotated_img = results[0].plot()
                        cv2.imwrite(
                            os.path.join(output_dir, f"annotated_{Path(image_path).name}"),
                            annotated_img
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {str(e)}")
                    results_summary["failed_images"].append(str(image_path))
            
            return results_summary
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_image_paths(self, input_path: Union[str, List[str]]) -> List[str]:
        """Get list of valid image paths from input."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        if isinstance(input_path, list):
            for path in input_path:
                if os.path.isfile(path) and Path(path).suffix.lower() in image_extensions:
                    image_paths.append(path)
        elif isinstance(input_path, str):
            if os.path.isfile(input_path):
                if Path(input_path).suffix.lower() in image_extensions:
                    image_paths.append(input_path)
            elif os.path.isdir(input_path):
                for ext in image_extensions:
                    image_paths.extend(Path(input_path).glob(f"*{ext}"))
                    image_paths.extend(Path(input_path).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        return sorted(list(set(image_paths)))
