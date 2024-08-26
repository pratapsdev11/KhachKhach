import os
import cv2
from ultralytics import YOLO
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
