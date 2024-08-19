import os
class FileAppender:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def append_to_files(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.folder_path, file_name)
                with open(file_path, "a") as file:
                    file.write(" 2.0")
