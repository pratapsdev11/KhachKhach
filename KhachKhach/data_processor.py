"""
Data Processing Module - Consolidates data processing functionality
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        pass

    def process_annotations(
        self, 
        input_dir: str, 
        output_dir: str, 
        format_type: str = "yolo",
        processing_mode: str = "extract_xyn"
    ) -> Dict[str, Any]:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            if processing_mode == "extract_xyn":
                return self._extract_xyn_arrays(input_dir, output_dir)
            elif processing_mode == "bounding_box":
                return self._process_bounding_boxes(input_dir, output_dir)
            elif processing_mode == "extended":
                return self._process_extended_arrays(input_dir, output_dir)
            else:
                return {"success": False, "error": f"Unknown processing mode: {processing_mode}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_xyn_arrays(self, annotations_folder: str, output_folder: str) -> Dict[str, Any]:
        """Extract XYN arrays from annotation files (from xyn_extractor.py)"""
        processed_files = 0
        
        for filename in os.listdir(annotations_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(annotations_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                try:
                    with open(file_path, "r") as file:
                        text = file.read()
                    
                    # Extract XYN content using regex
                    xyn_content = re.search(r'xyn.*?array\((.*?)\)', text, re.DOTALL)
                    
                    if xyn_content:
                        with open(output_path, "w") as output_file:
                            output_file.write(xyn_content.group(1).strip())
                        processed_files += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "processing_mode": "extract_xyn"
        }

    def _process_bounding_boxes(self, input_folder: str, output_folder: str) -> Dict[str, Any]:
        """Process files and add bounding boxes (from bounding_box_processor.py)"""
        processed_files = 0
        
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                
                try:
                    with open(file_path, "r") as file:
                        content = file.read()
                    
                    arr = np.array(eval(content))
                    arr_vector = arr.reshape(-1, 2)
                    arr_vector = arr_vector[(arr_vector[:, 0] != 0) & (arr_vector[:, 1] != 0)]
                    
                    if arr_vector.size == 0:
                        continue
                    
                    x_min, y_min = arr_vector.min(axis=0)
                    x_max, y_max = arr_vector.max(axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    
                    bounding_box = [x_center, y_center, width, height]
                    new_content = f"{bounding_box}\n{content}"
                    
                    output_path = os.path.join(output_folder, filename)
                    with open(output_path, "w") as file:
                        file.write(new_content)
                    
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "processing_mode": "bounding_box"
        }

    def _process_extended_arrays(self, input_folder: str, output_folder: str) -> Dict[str, Any]:
        """Process arrays with extended format (from extended_array_processor.py)"""
        processed_files = 0
        
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                try:
                    with open(file_path, "r") as file:
                        content = file.read()
                    
                    bounding_box_str, array_str = content.split('\n', 1)
                    bounding_box = eval(bounding_box_str)
                    arr = np.array(eval(array_str))
                    
                    arr_vector = arr.reshape(-1, 2)
                    arr_vector = arr_vector[(arr_vector[:, 0] != 0) & (arr_vector[:, 1] != 0)]
                    
                    if arr_vector.size == 0:
                        continue
                    
                    flattened_arr = arr_vector.flatten()
                    extended_arr = np.insert(flattened_arr, np.arange(2, len(flattened_arr), 2), 2.0)
                    
                    bounding_box_str_flat = " ".join(map(str, bounding_box))
                    extended_arr_str = " ".join(map(str, extended_arr))
                    
                    combined_content = f"{bounding_box_str_flat} {extended_arr_str}"
                    
                    with open(output_path, "w") as file:
                        file.write(combined_content)
                    
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "processing_mode": "extended"
        }

    def append_to_files(self, folder_path: str, text_to_append: str) -> Dict[str, Any]:
        """Append text to all txt files in folder (from file_appender.py)"""
        processed_files = 0
        
        try:
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "a") as file:
                        file.write(text_to_append)
                    processed_files += 1
                    
            return {
                "success": True,
                "processed_files": processed_files,
                "text_appended": text_to_append
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
