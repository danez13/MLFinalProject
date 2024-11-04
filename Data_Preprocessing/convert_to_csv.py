"""
- the following script turns the data text files into CSV Files
- all dataset categories (RUL001-RUL004, train001-train004, test001 - test 004) are done at the same time
"""

import os
import pandas as pd

path = "CMAPSSData"
new_path = "Processed_Data"
file_list = os.listdir(path)[2:]

true_RUL_files = file_list[:4]
test_files = file_list[4:8]
train_files = file_list[8:12]

# process RUL text files into CSV
for file_name in true_RUL_files:
    df = pd.read_csv(path+"/"+file_name,names=["RUL"])

    new_file_name = file_name.replace(".txt",".csv")
    df.to_csv(new_path+"/"+new_file_name,index=False)

# column names for the next two file categories
column_names = ["ID", "cycle", "op_setting_1", "op_setting_2", "op_setting_3", "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]

# process test files into CSV
for file_name in test_files:
    df = pd.read_csv(path+"/"+file_name,sep=" ", names=column_names,index_col=False)

    new_file_name = file_name.replace(".txt",".csv")
    df.to_csv(new_path+"/"+new_file_name,index=False)

# process train files into CSV
for file_name in train_files:
    df = pd.read_csv(path+"/"+file_name,sep=" ", names=column_names,index_col=False,)

    new_file_name = file_name.replace(".txt",".csv")
    df.to_csv(new_path+"/"+new_file_name,index=False)

""" 
- output for the next two files categories states improper length between column_names and the text file data
    - but the text file data carries two extra "columns"
        - an empty column 
            - i.e.""
        - and a new line column
            - i.e. "\n"
""" 
"""
column names have been reduced for convenience
- Full column names:
    - unit number, time, in cycles, operation setting 1, operation setting 2, operational setting 3, sensor measurement 1, sensor measurement 2, sensor measurement 3, sensor measurement 4, sensor measurement 5, sensor measurement 6, sensor measurement 7, sensor measurement 8, sensor measurement 9, sensor measurement 10, sensor measurement 11, sensor measurement 12, sensor measurement 13, sensor measurement 14, sensor measurement 15, sensor measurement 16, sensor measurement 17, sensor measurement 18, sensor measurement 19, sensor measurement 20, sensor measurement 21
"""