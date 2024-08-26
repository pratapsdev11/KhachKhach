# KhachKhach.py

# ---- bounding_box_processor.py ----
import os
import numpy as np

class BoundingBoxProcessor:
    def __init__(self, xyncontext_folder):
        self.xyncontext_folder = xyncontext_folder

    def process_files(self):
        for filename in os.listdir(self.xyncontext_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.xyncontext_folder, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                try:
                    arr = np.array(eval(content))
                except SyntaxError:
                    print(f"Error processing file {file_path}: invalid array format.")
                    continue
                arr_vector = arr.reshape(-1, 2)
                arr_vector = arr_vector[(arr_vector[:, 0] != 0) & (arr_vector[:, 1] != 0)]
                if arr_vector.size == 0:
                    print(f"File {file_path} contains only zero vectors.")
                    continue
                x_min, y_min = arr_vector.min(axis=0)
                x_max, y_max = arr_vector.max(axis=0)
                width = x_max - x_min
                height = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                bounding_box = [x_center, y_center, width, height]
                new_content = f"{bounding_box}\n" + content
                with open(file_path, "w") as file:
                    file.write(new_content)

# ---- detectannotate.py ----
from ultralytics import YOLO
from PIL import Image

class DetectionAnnotation:
    def __init__(self, output_folder: str, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def generate_annotation(self, image_path: str) -> None:
        image = Image.open(image_path)
        image_width, image_height = image.size
        results = self.model(image_path)
        annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        annotation_path = os.path.join(self.output_folder, annotation_filename)
        with open(annotation_path, "w") as f:
            for result in results[0].boxes.xyxy:
                x_min, y_min, x_max, y_max = result.tolist()
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                box_width = x_max - x_min
                box_height = y_max - y_min
                x_center_norm = x_center / image_width
                y_center_norm = y_center / image_height
                box_width_norm = box_width / image_width
                box_height_norm = box_height / image_height
                yolo_annotation = f"{x_center_norm:.6f} {y_center_norm:.6f} {box_width_norm:.6f} {box_height_norm:.6f}"
                f.write(yolo_annotation + "\n")
        print(f"YOLO annotation saved to {annotation_path}")

    def process_folder(self, image_folder: str) -> None:
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                self.generate_annotation(image_path)

# ---- extended_array_processor.py ----
class ExtendedArrayProcessor:
    def __init__(self, xyncontext_folder, newframe_folder):
        self.xyncontext_folder = xyncontext_folder
        self.newframe_folder = newframe_folder
        if not os.path.exists(self.newframe_folder):
            os.makedirs(self.newframe_folder)

    def process_files_in_xyncontext(self):
        for filename in os.listdir(self.xyncontext_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.xyncontext_folder, filename)
                new_file_path = os.path.join(self.newframe_folder, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                try:
                    bounding_box_str, array_str = content.split('\n', 1)
                    bounding_box = eval(bounding_box_str)
                    arr = np.array(eval(array_str))
                    arr_vector = arr.reshape(-1, 2)
                    arr_vector = arr_vector[(arr_vector[:, 0] != 0) & (arr_vector[:, 1] != 0)]
                    if arr_vector.size == 0:
                        print(f"File {file_path} contains only zero vectors.")
                        continue
                    flattened_arr = arr_vector.flatten()
                    extended_arr = np.insert(flattened_arr, np.arange(2, len(flattened_arr), 2), 2.0)
                    bounding_box_str_flat = ' '.join(map(str, bounding_box))
                    extended_arr_str = ' '.join(map(str, extended_arr))
                    combined_content = f"{bounding_box_str_flat} {extended_arr_str}"
                    with open(new_file_path, "w") as new_file:
                        new_file.write(combined_content)
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

# ---- file_appender.py ----
class FileAppender:
    def __init__(self, folder_path, text_to_append):
        self.folder_path = folder_path
        self.text_to_append = text_to_append

    def append_to_files(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.folder_path, file_name)
                with open(file_path, "a") as file:
                    file.write(self.text_to_append)

# ---- frame_extractor.py ----
import cv2

class FrameExtractor:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(self.output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()
        print(f"Extracted {frame_count} frames.")

# ---- keypoint_annotator.py ----
class KeypointAnnotator:
    def __init__(self, model_path, image_dir_path, annotate_dir_path):
        self.model = YOLO(model_path)
        self.image_dir_path = image_dir_path
        self.annotate_dir_path = annotate_dir_path
        os.makedirs(self.annotate_dir_path, exist_ok=True)

    def annotate_images(self):
        for filename in os.listdir(self.image_dir_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(self.image_dir_path, filename)
                image = cv2.imread(image_path)
                results = self.model(image)
                if results and hasattr(results[0], 'keypoints'):
                    keypoints = results[0].keypoints.cpu().numpy()[0]
                    annotate_file_path = os.path.join(self.annotate_dir_path, f'{os.path.splitext(filename)[0]}.txt')
                    with open(annotate_file_path, 'w') as f:
                        if len(keypoints) > 0:
                            f.write("Keypoints:\n")
                            for keypoint in keypoints:
                                f.write(f"{keypoint}\n")
                        else:
                            f.write("No keypoints found")
                else:
                    print(f"No keypoints found for {filename}")

# ---- xyn_extractor.py ----
import re

class XYNExtractor:
    def __init__(self, annotations_folder, xyncontext_folder):
        self.annotations_folder = annotations_folder
        self.xyncontext_folder = xyncontext_folder
        if not os.path.exists(self.xyncontext_folder):
            os.makedirs(self.xyncontext_folder)

    def extract_xyn_array(self):
        for filename in os.listdir(self.annotations_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.annotations_folder, filename)
                new_file_path = os.path.join(self.xyncontext_folder, filename)
                with open(file_path, "r") as file:
                    text = file.read()
                xyn_content = re.search(r'xyn: array\((.*?)\)', text, re.DOTALL)
                if xyn_content:
                    xyn_content = xyn_content.group(1).strip()
                    xyn_content = xyn_content.replace(", dtype=float32", "")
                    with open(new_file_path, "w") as new_file:
                        new_file.write(xyn_content)
