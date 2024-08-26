from KhachKhach import XYNExtractor
# Initialize the XYNExtractor with the annotations folder and output folder
xyn_extractor = XYNExtractor(annotations_folder="annotations", xyncontext_folder="xyncontext_folder")

# Extract the xyn arrays from the annotation files and save them to the xyncontext folder
xyn_extractor.extract_xyn_array()
