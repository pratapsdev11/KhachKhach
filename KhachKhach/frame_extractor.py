import os
import cv2
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


