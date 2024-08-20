from khach_khach import XYNExtractor

annotations_folder = 'annotations'
xyncontext_folder = 'xyncontext'

extractor = XYNExtractor(annotations_folder, xyncontext_folder)
extractor.extract_xyn_array()
