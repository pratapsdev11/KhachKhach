# khach_khach

`khach_khach` is a Python package designed for processing video frames, annotating keypoints using YOLOv8, and performing subsequent operations on extracted data. It includes functionality to extract frames from videos, annotate keypoints on images, extract and process keypoint data, and more.

## Features

- **Frame Extraction:** Extract frames from a video and save them as JPEG files.
- **Keypoint Annotation:** Annotate images with keypoints using YOLOv8.
- **Data Extraction and Processing:** Extract keypoint data, compute bounding boxes, and extend arrays for further analysis.
- **File Operations:** Append data to files as needed.

## Installation

To install `khach_khach`, you need to have `opencv-python`, `numpy`, and `ultralytics` packages installed. You can install these dependencies using pip:

```bash
pip install opencv-python numpy ultralytics
