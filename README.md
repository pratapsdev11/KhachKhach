# KhachKhach✂️

KhachKhach is a Python library for processing video frames, annotating keypoints, and more. It provides various utilities for handling and analyzing images and videos, making it easier to integrate computer vision tasks into your workflows.

## Versions

### Version 0.6 and 0.6.1

Version 0.6 offers users the flexibility to customize their workflows with different Ultralytics YOLO models.
version 0.6.1 offers no customization and DetectionAnnotation module is smooth to use in this version as prev. version only offers 
single image annotaion at a time.
#### Features:
- **Frame Extraction:** Extract frames from video files and save them as JPEG files.
- **Keypoint Annotation:** Annotate images with keypoints using YOLOv8 models.
- **Bounding Box Processing:** Process text files with bounding box data and append computed bounding boxes to file content.
- **Extended Array Processing:** Process text files with bounding box and array data, adding extended array information.
- **File Appending:** Append text to all text files in a specified folder.
- **XYN Extraction:** Extract and save XYN arrays from annotation files.

#### for further details and how to use :
  check out the test folder.
#### Installation

To install KhachKhach version 0.6 and other requiered packages , use pip:   
```bash
pip install opencv-python numpy ultralytics
pip install KhachKhach==0.6
pip install KhachKhach


