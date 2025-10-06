# KhachKhach ✂️

**KhachKhach** is a Python library for advanced video and image processing, with a focus on frame extraction, keypoint annotation, and object detection using YOLO models. It is designed to make computer vision workflows easy, flexible, and highly customizable.

## Unique Features

- **Flexible YOLO Model Support:** Use any Ultralytics YOLO model (including pose models) for detection and keypoint annotation.
- **Frame Extraction:** Extract frames from videos at custom intervals, time ranges, or frame counts.
- **Keypoint Annotation:** Annotate single images or entire folders with keypoints, supporting normalized coordinates and multiple output formats.
- **Bounding Box Processing:** Detect and annotate bounding boxes, with options for saving annotated images and exporting results.
- **Batch Processing:** Process entire folders of images or videos in one go.
- **Custom Output:** Choose between normalized (0-1) or absolute coordinates, and select output format (space-separated, comma-separated, etc).
- **Easy Integration:** Simple API for use in scripts, research, or production pipelines.
- **Utilities:** Includes tools for appending text to files, extracting XYN arrays, and more.

## Installation

Install the required dependencies and KhachKhach via pip:

```bash
pip install opencv-python numpy pillow ultralytics
pip install KhachKhach
```

## Quick Start

### 1. Extract Frames from a Video

```python
import khachkhach as kk

video_path = "your_video.mp4"
frames_dir = "frames"

video_processor = kk.VideoProcessor()
video_processor.extract_frames(video_path, frames_dir)
```

### 2. Detect Objects or Annotate Keypoints

```python
import khachkhach as kk

engine = kk.DetectionEngine("yolo11n-pose.pt")
engine.detect_objects(input_path="frames", output_dir="objectin")
# or for keypoints:
engine.annotate_keypoints(input_path="frames", output_dir="objectin", normalize_coords=True)
```

### 3. Full Pipeline Example

```python
import os
import khachkhach as kk

video_path = "your_video.mp4"
frames_dir = "frames"
output_dir = "objectin"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

video_processor = kk.VideoProcessor()
video_processor.extract_frames(video_path, frames_dir)

engine = kk.DetectionEngine("yolo11n-pose.pt")
engine.annotate_keypoints(input_path=frames_dir, output_dir=output_dir)
```

## Advanced Usage

- **Normalized Coordinates:** Get keypoints in 0-1 range for ML workflows.
- **Custom Output Format:** Choose between space-separated or comma-separated output.
- **Batch Processing:** Process all images in a folder with a single call.
- **Integration:** Use in your own scripts or extend with custom logic.

## Example Scripts

See the `test` folder for ready-to-run examples:
- `test_detect.py`: Object detection on images.
- `test_video.py`: Full pipeline from video to keypoints.
- `test.py`: Advanced keypoint extraction with normalization.

## Requirements

- Python 3.7+
- opencv-python
- numpy
- pillow
- ultralytics

## License

See `LICENSE` for details.


