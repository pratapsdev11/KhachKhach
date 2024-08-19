import os
import numpy as np
class ExtendedArrayProcessor:
    def __init__(self, xyncontext_folder, newframe_folder):
        self.xyncontext_folder = xyncontext_folder
        self.newframe_folder = newframe_folder
        if not os.path.exists(self.newframe_folder):
            os.makedirs(self.newframe_folder)

    def process_files_in_xyncontext(self):
        for filename in os.listdir(self.xyncontext_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.xyncontext_folder, filename)
                new_file_path = os.path.join(self.newframe_folder, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                try:
                    bounding_box_str, array_str = content.split('\n', 1)
                    bounding_box = eval(bounding_box_str)
                    arr = np.array(eval(array_str))
                    arr_vector = arr.reshape(-1, 2)
                    arr_vector = arr_vector[(arr_vector[:, 0] != 0) & (arr_vector[:, 1] != 0)]
                    if arr_vector.size == 0:
                        print(f"File {file_path} contains only zero vectors.")
                        continue
                    flattened_arr = arr_vector.flatten()
                    extended_arr = np.insert(flattened_arr, np.arange(2, len(flattened_arr), 2), 2.0)
                    bounding_box_str_flat = ' '.join(map(str, bounding_box))
                    extended_arr_str = ' '.join(map(str, extended_arr))
                    combined_content = f"{bounding_box_str_flat} {extended_arr_str}"
                    with open(new_file_path, "w") as new_file:
                        new_file.write(combined_content)
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue