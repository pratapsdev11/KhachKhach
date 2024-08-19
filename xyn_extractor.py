import os
import re
class XYNExtractor:
    def __init__(self, annotations_folder, xyncontext_folder):
        """
        Initialize the XYNExtractor with the annotations folder and the xyncontext folder.
        
        :param annotations_folder: Path to the folder containing the annotation text files.
        :param xyncontext_folder: Path to the folder where the extracted xyn arrays will be saved.
        """
        self.annotations_folder = annotations_folder
        self.xyncontext_folder = xyncontext_folder

        # Ensure the xyncontext folder exists
        if not os.path.exists(self.xyncontext_folder):
            os.makedirs(self.xyncontext_folder)

    def extract_xyn_array(self):
        """
        Extract the content within the xyn array from all text files in the annotations folder
        and write it to new text files in the xyncontext folder.
        """
        for filename in os.listdir(self.annotations_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.annotations_folder, filename)
                new_file_path = os.path.join(self.xyncontext_folder, filename)

                # Read the content of the text file
                with open(file_path, "r") as file:
                    text = file.read()

                # Extract content within the xyn array
                xyn_content = re.search(r'xyn: array\((.*?)\)', text, re.DOTALL)
                if xyn_content:
                    xyn_content = xyn_content.group(1).strip()
                    xyn_content = xyn_content.replace(", dtype=float32", "")

                    # Write the extracted content to the new text file
                    with open(new_file_path, "w") as new_file:
                        new_file.write(xyn_content)