# Import
import pandas as pd
import os
import glob
import json
import csv

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__)) # gemini assistance for os
proj_root_dir = os.path.abspath(os.path.join(script_dir,'../../'))
cnewsum_eng_data_dir = os.path.join(proj_root_dir,'data','clean','CNewSum') # Use TWO func
cnewsum_zho_data_dir = os.path.join(proj_root_dir,'data','raw','CNewSum','final')
cnn_clean_data_dir = os.path.join(proj_root_dir,'data','raw','cnn_dailymail')


#  CNewSum (JSONL) Chinese
def load_zho_cnewsum_data(type='train', directory=cnewsum_zho_data_dir):
    """
    Loads data from all zho JSONL files of a specific type in a directory.
    types: ['dev', 'test', 'train'] (default type is train)
    input: directory path
    output: list of tuples [(article, id), ...]
        ALERT: Articles are a list of sentances: i.e. ['澎湃新闻 记者 。',... '长三角 政商 字号 。']
    """
    # Look for files ending in .jsonl
    print(f"type: {type}")
    all_files = glob.glob(os.path.join(directory, f"{type}*.jsonl")) # gemini query, how to find all files of one type 
    
    # get all data in the same list
    data_list = []
    for filename in all_files: # https://rowzero.com/blog/open-jsonl-file-format
        print(f"filename: {filename}")
        with open(filename, 'r', encoding='utf-8') as f: # gemini query on working with jsonl files.
            for line in f:
                try:
                    # Load each line as a JSON object
                    record = json.loads(line.strip()) # attempted pandas ref: https://medium.com/@whyamit101/understanding-jsonl-format-11fd86897f1a
                    # JSON keys: 'article' and 'id'
                    article = record.get('article')
                    doc_id = record.get('id')
                    
                    if article is not None and doc_id is not None:
                        data_list.append((article, f'{filename}_{doc_id}')) # get a unique id for every article
                except json.JSONDecodeError as e: # gemini generated this code when debugging. 
                    print(f"Error decoding JSON in file {filename}: {e}")
                except AttributeError as e:
                    print(f"Missing 'article' or 'id' key in a record in file {filename}: {e}")
                    
    return data_list

#  CNewSum (JSONL) English ## Under construction
def load_eng_cnewsum_data(type='train', directory=cnewsum_eng_data_dir):
    """
    Loads data from all JSONL files of a specific type in a directory.
    types: ['dev', 'test', 'train'] (default type is train)
    input: directory path
    output: list of tuples [(article, id), ...]
    """
    # Look for files ending in .jsonl
    print(f"type: {type}")
    all_files = glob.glob(os.path.join(directory, f"{type}*.jsonl")) # gemini query, how to find all files of one type 
    
    # get all data in the same list
    data_list = []
    for filename in all_files: # https://rowzero.com/blog/open-jsonl-file-format
        print(f"filename: {filename}")
        with open(filename, 'r', encoding='utf-8') as f: # gemini query on working with jsonl files.
            for line in f:
                try:
                    # Load each line as a JSON object
                    record = json.loads(line.strip()) # attempted pandas ref: https://medium.com/@whyamit101/understanding-jsonl-format-11fd86897f1a
                    # JSON keys: 'article' and 'id'
                    article = record.get('article')
                    doc_id = record.get('id')
                    
                    if article is not None and doc_id is not None:
                        data_list.append((article, f'{filename}_{doc_id}'))
                except json.JSONDecodeError as e: # gemini generated this code when debugging. 
                    print(f"Error decoding JSON in file {filename}: {e}")
                except AttributeError as e:
                    print(f"Missing 'article' or 'id' key in a record in file {filename}: {e}")
                    
    return data_list

# CNN (Parquet)
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
            df = pd.read_parquet(filename) # use pd to read in parquet
            article_col = 'article'
            id_col = 'id'
            
            if article_col in df.columns and id_col in df.columns: # gemini query for error generation
                # Select the required columns and convert to a list of tuples
                # .to_numpy() is generally faster than iterrows()
                df['full_id'] = (f'{filename}_{df[id_col]}')
                data_list.extend(map(tuple, df[[article_col, 'full_id']].to_numpy())) # ('article','filename_id') CNN id is text, gemini recommended map
                #print(data_list)
            else:
                print(f"Skipping file {filename}: Missing '{article_col}' or '{id_col}' columns.") # gemini error code generation.
                
        except Exception as e:
            print(f"Error reading Parquet file {filename}: {e}")
            
    return data_list

# Setup translation dir struct
def setup_dir(output_dir, filename="translations.csv"):
    """
    Creates the directory and initializes the CSV with headers.
    Returns the full path to the CSV file. Generated with Gemini.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, filename)
    
    # 'w' mode overwrites existing file to start fresh
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Article_ID', 'Sentence_Index', 'Original_Chinese', 'English_Translation'])
        
    return csv_path

# Flatten Chinese sentences
def flatten_sent(chinese_sentence_list):
    """"
    input: a list of chinese news articles tuples.
        each tuple is the article,id combination.
        The Chinese news article text is a list of sentances.
        This function flattens the sentences. Gemini was used to generate and
        troubleshoot much of this function.

        convert [(["sent1", "sent2"], id1), ...] 
        into -> [(id1, 0, "sent1"), (id1, 1, "sent2"), ...]
    """
    print("Preparing sentences...")
    flat_sentence_list = []
    for article_sentences, art_id in chinese_sentence_list:
        for idx, sentence in enumerate(article_sentences): # https://www.geeksforgeeks.org/python/enumerate-in-python/
            if sentence.strip(): # Skip empty strings
                flat_sentence_list.append({
                    "id": art_id,
                    "idx": idx,
                    "text": sentence
                })

    total_sentences = len(flat_sentence_list)
    print(f"Found {total_sentences} total sentences to translate.")
    return flat_sentence_list

if __name__ == "__main__":
    # CNN
    print("cnn/dailymail")
    dl_cnn = load_cnn_data('test') 

    ids_cnn = [doc_id for art, doc_id in dl_cnn]
    print(f"cnn_dailymail: \n \
        count of articles: {len(dl_cnn)} \n \
        arts: {dl_cnn[0:2]} \n \
        first 10 ids: {ids_cnn[0:10]}") # verify unique ids
    
    print("-" * 30) # gemini suggestion
# zho CNewSum
    print("Chinese cnewsum")
    dl_cns = load_zho_cnewsum_data('test') 

    ids_cns = [doc_id for art, doc_id in dl_cns]
    print(f"cnewsum: \n \
        count of articles: {len(dl_cns)} \n \
                arts: {dl_cns[0:2]} \n \
        first 10 ids: {ids_cns[0:10]}")
    
    print("-" * 30)
# eng CNewSum
    # print("English cnewsum")
    # dl_cns = load_zho_cnewsum_data('test') 

    # ids_cns = [doc_id for art, doc_id in dl_cns]
    # print(f"cnewsum: \n \
    #     count of articles: {len(dl_cns)} \n \
    #             arts: {dl_cns[0:2]} \n \
    #     first 10 ids: {ids_cns[0:10]}")