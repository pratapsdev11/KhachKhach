'''i used this because i required 52 numeric value but one 2.0 numeric value (which is a constant) 
was missing at the end of the file dont know why ....thats why fileappender is used to append 2.0 at the end of the file
if the reader find the reason please share with me
'''
from khach_khach import FileAppender

folder_path = 'newframes2'
text_to_append = " 2.0"

appender = FileAppender(folder_path)
appender.append_to_files()
