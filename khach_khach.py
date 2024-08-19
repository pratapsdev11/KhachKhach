import os
import cv2
import re
from ultralytics import YOLO
import numpy as np
class FrameExtractor:
    def __init__(self, video_path, output_folder):
        """
        Initialize the FrameExtractor with video path and output folder.
        
        :param video_path: Path to the input video file.
        :param output_folder: Path to the folder where frames will be saved.
        """
        self.video_path = video_path
        self.output_folder = output_folder

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames(self):
        """
        Extract frames from the video and save them as JPEG files.
        """
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_count = 0

        while True:
            # Read a frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame as JPEG file
            frame_filename = os.path.join(self.output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1

        # Release the video capture object
        cap.release()
        print(f"Extracted {frame_count} frames.")

class KeypointAnnotator:
    def __init__(self, model_path, image_dir_path, annotate_dir_path):
        """
        Initialize the KeypointAnnotator with model path, image directory path, and annotation directory path.
        
        :param model_path: Path to the YOLOv8 model.
        :param image_dir_path: Path to the folder containing images to process.
        :param annotate_dir_path: Path to the folder where annotations will be saved.
        """
        self.model = YOLO(model_path)  # Load the YOLOv8 model
        self.image_dir_path = image_dir_path
        self.annotate_dir_path = annotate_dir_path

        # Create directory for annotations if it doesn't exist
        os.makedirs(self.annotate_dir_path, exist_ok=True)

    def annotate_images(self):
        """
        Annotate images with keypoints using the YOLOv8 model.
        """
        for filename in os.listdir(self.image_dir_path):
            if filename.endswith('.jpg'):
                # Path to the image
                image_path = os.path.join(self.image_dir_path, filename)

                # Load the image
                image = cv2.imread(image_path)

                # Perform inference
                results = self.model(image)

                # Extract keypoints from the results
                if results and hasattr(results[0], 'keypoints'):
                    keypoints = results[0].keypoints.cpu().numpy()[0]  # Extracting keypoints for the first detected person

                    # File path for the annotation file
                    annotate_file_path = os.path.join(self.annotate_dir_path, f'{os.path.splitext(filename)[0]}.txt')

                    # Save keypoints to a text file
                    with open(annotate_file_path, 'w') as f:
                        if len(keypoints) > 0:
                            f.write("Keypoints:\n")
                            for keypoint in keypoints:
                                f.write(f"{keypoint}\n")
                        else:
                            f.write("No keypoints found")
                else:
                    print(f"No keypoints found for {filename}")

class XYNExtractor:
    def __init__(self, annotations_folder, xyncontext_folder):
        """
        Initialize the XYNExtractor with the annotations folder and the xyncontext folder.
        
        :param annotations_folder: Path to the folder containing the annotation text files.
        :param xyncontext_folder: Path to the folder where the extracted xyn arrays will be saved.
        """
        self.annotations_folder = annotations_folder
        self.xyncontext_folder = xyncontext_folder

        # Ensure the xyncontext folder exists
        if not os.path.exists(self.xyncontext_folder):
            os.makedirs(self.xyncontext_folder)

    def extract_xyn_array(self):
        """
        Extract the content within the xyn array from all text files in the annotations folder
        and write it to new text files in the xyncontext folder.
        """
        for filename in os.listdir(self.annotations_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.annotations_folder, filename)
                new_file_path = os.path.join(self.xyncontext_folder, filename)

                # Read the content of the text file
                with open(file_path, "r") as file:
                    text = file.read()

                # Extract content within the xyn array
                xyn_content = re.search(r'xyn: array\((.*?)\)', text, re.DOTALL)
                if xyn_content:
                    xyn_content = xyn_content.group(1).strip()
                    xyn_content = xyn_content.replace(", dtype=float32", "")

                    # Write the extracted content to the new text file
                    with open(new_file_path, "w") as new_file:
                        new_file.write(xyn_content)
class BoundingBoxProcessor:
    def __init__(self, xyncontext_folder):
        """
        Initialize the BoundingBoxProcessor with the xyncontext folder.
        
        :param xyncontext_folder: Path to the folder containing the text files.
        """
        self.xyncontext_folder = xyncontext_folder

    def process_files(self):
        """
        Processes each file in the xyncontext folder by extracting the numpy array, 
        computing the bounding box, and writing the bounding box along with the 
        original content back to the file.
        """
        # List all text files in the xyncontext folder
        for filename in os.listdir(self.xyncontext_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.xyncontext_folder, filename)
                
                # Read the content of the text file
                with open(file_path, "r") as file:
                    content = file.read()
                
                # Convert the string representation of the array to a numpy array
                try:
                    arr = np.array(eval(content))
                except SyntaxError:
                    print(f"Error processing file {file_path}: invalid array format.")
                    continue
                
                # Reshape and filter the array
                arr_vector = arr.reshape(-1, 2)
                arr_vector = arr_vector[(arr_vector[:, 0] != 0) & (arr_vector[:, 1] != 0)]
                
                if arr_vector.size == 0:
                    print(f"File {file_path} contains only zero vectors.")
                    continue
                
                # Determine the bounding box coordinates
                x_min, y_min = arr_vector.min(axis=0)
                x_max, y_max = arr_vector.max(axis=0)
                
                # Calculate width and height
                width = x_max - x_min
                height = y_max - y_min
                
                # Calculate the center of the bounding box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # Bounding box in (x_center, y_center, width, height) format
                bounding_box = [x_center, y_center, width, height]
                
                # Code to append bounding box to file content
                new_content = f"{bounding_box}\n" + content
                
                # Writing the new content back to the file
                with open(file_path, "w") as file:
                    file.write(new_content)
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
# Class to extend arrays and save processed files
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
class FileAppender:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def append_to_files(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.folder_path, file_name)
                with open(file_path, "a") as file:
                    file.write(" 2.0")
