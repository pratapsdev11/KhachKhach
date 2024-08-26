'''In this example, we will append the text " 2.0" to all text files in the folder "newframe_folder". 
This is simply because there was a bug which was not capturing the constant 2.0. Remember to give sapce before 2.0'''
from KhachKhach import FileAppender
appender = FileAppender(folder_path="newframe_folder", text_to_append=" 2.0")

# Append the specified text to all text files in the folder
appender.append_to_files()