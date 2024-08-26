from KhachKhach import KeypointAnnotator
from KhachKhach import BoundingBoxProcessor, DetectionAnnotation
detector = DetectionAnnotation(output_folder="output_folder")
# Process all images in the specified folder
detector.process_folder(image_folder="floder")
#keypoint annotaion 
annotator = KeypointAnnotator(image_dir_path="frames", annotate_dir_path="annotations")
annotator.annotate_images()