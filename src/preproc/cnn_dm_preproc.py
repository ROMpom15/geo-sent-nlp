# Preprocess and Clean data

# Import
import os
import re
import pandas as pd
#https://stackoverflow.com/questions/63999261/simple-regex-pattern-to-match-any-py-files-in-specific-directory

# File System
# https://www.geeksforgeeks.org/python/get-current-directory-python/
# https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
# 
file = 'cnn_dm_preproc.py'
print(os.getcwd()) 
cnews_dir = os.path.abspath("README.md")
script_dir = os.path.dirname(os.path.abspath(file)) # gemini assistance
print(script_dir)
# Iterate through files, load parquet files
# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory


for file in os.listdir(cnews_dir):
    filename = os.fsdecode(file)
    if filename.endswith(".parquet") or filename.endswith(".py"): 
            print (filename)



