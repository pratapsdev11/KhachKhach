from khach_khach import KeypointAnnotator

model_path = 'Khach_Khach\yolov8n-pose.pt'
image_dir_path = 'frames29'
annotate_dir_path = 'annotations'

annotator = KeypointAnnotator(model_path, image_dir_path, annotate_dir_path)
annotator.annotate_images()
