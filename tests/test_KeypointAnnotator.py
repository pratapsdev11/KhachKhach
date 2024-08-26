from KhachKhach import KeypointAnnotator
# Initialize the KeypointAnnotator with the YOLO model path, image directory, and annotation directory
annotator = KeypointAnnotator(model_path="yolov8n-pose.pt", image_dir_path="frames", annotate_dir_path="annotations")

# Annotate the images with keypoints and save the annotations
annotator.annotate_images()
