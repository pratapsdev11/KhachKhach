# khach_khach

`khach_khach` is a Python package designed for advanced video and image processing tasks. It offers tools for extracting frames from videos, annotating keypoints using YOLOv8, and performing various operations on extracted data. This package is intended for use in computer vision and data analysis applications.

## Features

### Frame Extraction

**`FrameExtractor` Class:**
- **Purpose:** Extract frames from a video file and save them as JPEG images.
- **Functionality:**
  - **Extract Frames:** Extracts frames from a video at specified intervals.
  - **Save Frames:** Saves the extracted frames to a specified directory.
  - **Usage Example:**
    ```python
    from khach_khach import FrameExtractor

    video_path = 'path/to/your/video.mp4'
    output_folder = 'path/to/output/folder'

    extractor = FrameExtractor(video_path, output_folder)
    extractor.extract_frames(interval=30)  # Extract every 30th frame
    ```

### Keypoint Annotation

**`KeypointAnnotator` Class:**
- **Purpose:** Annotate images with keypoints using the YOLOv8 model.
- **Functionality:**
  - **Load Model:** Load a pre-trained YOLOv8 model for keypoint detection.
  - **Annotate Images:** Apply the model to images to detect and annotate keypoints.
  - **Usage Example:**
    ```python
    from khach_khach import KeypointAnnotator

    image_path = 'path/to/your/image.jpg'
    model_path = 'path/to/yolov8/model.pt'

    annotator = KeypointAnnotator(model_path)
    annotator.annotate_image(image_path, output_path='path/to/annotated_image.jpg')
    ```

### Data Extraction and Processing

**`KeypointDataExtractor` Class:**
- **Purpose:** Extract keypoint data from annotated images and perform data processing.
- **Functionality:**
  - **Extract Keypoints:** Retrieve keypoint data from images annotated with keypoints.
  - **Process Data:** Compute bounding boxes and manipulate keypoint arrays for further analysis.
  - **Usage Example:**
    ```python
    from khach_khach import KeypointDataExtractor

    annotated_image_path = 'path/to/annotated_image.jpg'

    data_extractor = KeypointDataExtractor()
    data = data_extractor.extract_keypoints(annotated_image_path)
    ```

### File Operations

**`FileAppender` Class:**
- **Purpose:** Append specified text to all `.txt` files in a given directory.
- **Functionality:**
  - **Append Text:** Appends user-defined text to each text file in the specified folder.
  - **Usage Example:**
    ```python
    from khach_khach import FileAppender

    folder_path = 'path/to/your/folder'
    text_to_append = " Additional text"

    appender = FileAppender(folder_path)
    appender.append_to_files()
    ```

## Installation

To install `khach_khach`, ensure that you have the following dependencies: `opencv-python`, `numpy`, and `ultralytics`. Install them using pip:

```bash
pip install opencv-python numpy ultralytics
pip install khach-khach2.0
