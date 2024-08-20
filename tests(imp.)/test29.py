from khach_khach import BoundingBoxProcessor, ExtendedArrayProcessor

xyncontext_folder = 'xyncontext'
newframe_folder = 'newframes29'

bbox_processor = BoundingBoxProcessor(xyncontext_folder)
bbox_processor.process_files()

array_processor = ExtendedArrayProcessor(xyncontext_folder, newframe_folder)
array_processor.process_files_in_xyncontext()
