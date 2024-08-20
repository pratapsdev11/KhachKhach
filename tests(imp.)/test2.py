import khach_khach
from khach_khach import FrameExtractor
video_path = 'male-Barbell-barbell-front-squat-olympic-front.mp4'
output_folder = 'frames29'
extractor = FrameExtractor(video_path, output_folder)
extractor.extract_frames()