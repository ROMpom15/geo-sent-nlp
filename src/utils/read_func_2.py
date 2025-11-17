# Import
import pandas as pd
import os
import glob
import json

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__)) # gemini assistance for os
proj_root_dir = os.path.abspath(os.path.join(script_dir,'../../'))
cnewsum_data_dir = os.path.join(proj_root_dir,'data','raw','CNewSum') # Use one dir for CNewSum
cnn_clean_data_dir = os.path.join(proj_root_dir,'data','raw','cnn_dailymail')

# --- Helper Function for CNewSum (JSONL) ---
def load_cnewsum_data(type='train', directory=cnewsum_data_dir):
    """
    Loads data from all JSONL files of a specific type in a directory.
    types: ['dev', 'test', 'train'] (default type is train)
    input: directory path
    output: list of tuples [(article, id), ...]
    """
    # Look for files ending in .jsonl
    all_files = glob.glob(os.path.join(directory, f"{type}*.jsonl")) # gemini query, how to find all files of one type 
    
    # get all data in the same list
    data_list = []
    for filename in all_files: # https://rowzero.com/blog/open-jsonl-file-format
        with open(filename, 'r', encoding='utf-8') as f: # gemini query on working with jsonl files.
            for line in f:
                try:
                    # Load each line as a JSON object
                    record = json.loads(line.strip()) # attempted pandas ref: https://medium.com/@whyamit101/understanding-jsonl-format-11fd86897f1a
                    # JSON keys: 'article' and 'id'
                    article = record.get('article')
                    doc_id = record.get('id')
                    
                    if article is not None and doc_id is not None:
                        data_list.append((article, doc_id))
                except json.JSONDecodeError as e: # gemini generated this code when debugging. 
                    print(f"Error decoding JSON in file {filename}: {e}")
                except AttributeError as e:
                    print(f"Missing 'article' or 'id' key in a record in file {filename}: {e}")
                    
    return data_list

# --- Function for CNN/Daily Mail (Parquet) ---
def load_cnn_data(type='train', directory=cnn_clean_data_dir):
    """
    Loads data from all Parquet files of a specific type in a directory.
    Assumes Parquet columns are 'article' (or 'story') and 'id'.
    types: ['validation', 'test', 'train']
    input: directory path
    output: list of tuples [(article, id), ...]
    """
    # Look for files ending in .parquet
    all_files = glob.glob(os.path.join(directory, f"{type}*.parquet"))
    
    data_list = []
    for filename in all_files:
        try:
            # Read Parquet file
            df = pd.read_parquet(filename)
            
            # CNN/Daily Mail often uses 'story' for the article and 'id'
            # We check for 'article' first, then 'story'
            article_col = 'article' if 'article' in df.columns else 'story'
            id_col = 'id'
            
            if article_col in df.columns and id_col in df.columns:
                # Select the required columns and convert to a list of tuples
                # .to_numpy() is generally faster than iterrows()
                data_list.extend(df[[article_col, id_col]].to_numpy().tolist())
            else:
                print(f"Skipping file {filename}: Missing '{article_col}' or '{id_col}' columns.")
                
        except Exception as e:
            print(f"Error reading Parquet file {filename}: {e}")
            
    return data_list

if __name__ == "__main__":
    # --- Example Usage for CNN/Daily Mail (Parquet) ---
    print("🚀 Testing CNN/Daily Mail (Parquet) Loader...")
    # NOTE: This will only work if you have 'test*.parquet' files in the cnn_dailymail directory.
    dl_cnn = load_cnn_data('test') 
    
    if dl_cnn:
        ids_cnn = [doc_id for art, doc_id in dl_cnn]
        print(f"cnn_dailymail: \n \
          count of articles: {len(dl_cnn)} \n \
          first 10 ids: {ids_cnn[0:10]}")
    else:
        print("cnn_dailymail: No data loaded (check if Parquet files exist).")
    
    print("-" * 30)
    
    # --- Example Usage for CNewSum (JSONL) ---
    print("📚 Testing CNewSum (JSONL) Loader...")
    # NOTE: This will only work if you have 'test*.jsonl' files in the CNewSum directory.
    dl_cnewsum = load_cnewsum_data('test')
    
    if dl_cnewsum:
        ids_cnewsum = [doc_id for art, doc_id in dl_cnewsum]
        print(f"CNewSum: \n \
          count of articles: {len(dl_cnewsum)} \n \
          first 10 ids: {ids_cnewsum[0:10]}")
    else:
        print("CNewSum: No data loaded (check if JSONL files exist).")