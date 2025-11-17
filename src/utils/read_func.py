import pandas as pd
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__)) # gemini assistance
proj_root_dir = os.path.abspath(os.path.join(script_dir,'../../'))
cnewsum_clean_chinese_data_dir = os.path.join(proj_root_dir,'data','raw','CNewSum')
cnewsum_trans_data_dir = os.path.join(proj_root_dir,'data','clean','CNewSum','trans')
cnn_clean_data_dir = os.path.join(proj_root_dir,'data','raw','cnn_dailymail')

def load_chinese_cnewsum_data(type='train',directory = cnewsum_clean_chinese_data_dir):
    """
    Loads data from all CSV files of a specific type in a directory
    types: ['dev','test','train']
    input: directory obj
    output: list of tuples [(article,id),...]
    """
    all_files = glob.glob(os.path.join(directory, f"{type}*.csv"))
    # print(all_files)
    
    data_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        for index, row in df.iterrows(): #iterate through every row
            data_list.append((row[0],row[1]))
            # print(data_list)
            
    return data_list

def load_trans_cnewsum_data(type='train', directory = cnewsum_trans_data_dir):
    """
    Loads data from all CSV files of a specific type in a directory
    types: ['dev','test','train']
    input: directory obj
    output: list of tuples [(article,id),...]
    """
    all_files = glob.glob(os.path.join(directory, f"{type}*.csv"))
    # print(all_files)
    
    data_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        for index, row in df.iterrows(): #iterate through every row
            data_list.append((row[0],row[1]))
            # print(data_list)
            
    return data_list

def load_cnn_data(type='train', directory=cnn_clean_data_dir):
    """
    Loads data from all CSV files of a specific type in a directory
    types: ['validation','test','train']
    input: directory obj
    output: list of tuples [(article,id),...]
    """
    all_files = glob.glob(os.path.join(directory, f"{type}*.csv"))
    # print(all_files)
    
    data_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        for index, row in df.iterrows(): 
            data_list.append((row.iloc[1],row.iloc[2]))
            # print(data_list)
            
    return data_list

if __name__ == "__main__":
    dl = load_cnn_data('test')
    ids = [id for art,id in dl]
    print(f"cnn: \n \
          count of articles: {len(dl)} \n \
          first 10 ids: {ids[0:9]}")
    
    dl = load_chinese_cnewsum_data('test')
    ids = [id for art,id in dl]
    print(f"Chinese NewSum: \n \
          count of articles: {len(dl)} \n \
          first 10 ids: {ids[0:9]}")
