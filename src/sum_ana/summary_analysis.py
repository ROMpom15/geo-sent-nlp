# Summary Analysis
# this file summarizes topic clusters into a 1-2 sentence overview
# the t5 code comes from https://medium.com/@ivavrtaric/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771 and adapted to the use of the NLP project. 


# pip install torch transformers
# pip install pip-system-certs

# import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead
import os
from datasets import load_dataset
import pandas as pd

# Load the model
model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name, return_dict=True)

# Load the US data
file_path = os.path.join(os.path.expanduser("~"), "train-00000-of-00003.parquet") # absolute path to the dataset
us_data_load = load_dataset("parquet", data_files={'train': file_path})
us_data = us_data_load["train"]

# Load the Chinese data
chinese_csv = "translations.csv" # name of CSV file
df = pd.read_csv(chinese_csv) # open and read the file
df = df.sort_values(by=['Article_ID', 'Sentence_Index']) # sort by index number
chinese_data = df.groupby('Article_ID').agg(Full_English=('English_Translation', ' '.join)).reset_index() # group articles together by index


# Initialize variables for user input loop
keyword = ""
# key_text = ""

# Loop until user types "quit"
while (keyword != "quit"):
    key_text = "" # reset key_text for next keyword search (keyword is already reset)

    keyword = input("Enter a keyword to summarize articles about (or type 'quit' to exit): ")
    if keyword == "quit":
        break
    

    ''' American Summary '''
    print (f"Summarizing US articles with \"{keyword}\"")
    for i in range(len(us_data)):
        if keyword in us_data[i]['article']: # append based on if of keyword in article text

            # access the specific column by its name ['article'] & add to key_text
            key_text += us_data[i]['article'] + "\n"

    if not key_text:
        print(f"No US articles with \"{keyword}\"")
        key_text = "N/A"
        us_generated_summary = "N/A"
    else:
        # T5 requires the prefix "summarize: " 
        input_text = "Summarize: " + key_text

        # Encode inputs
        inputs = tokenizer.encode(
                        input_text,             # actual text to be summarized, begins with "Summarize: "
                        return_tensors='pt',    # encoded sequence will be returned as a PyTorch tensor object
                        max_length=512,         # max length is 512 tokens
                        truncation=True)        # anything longer will be removed

        # Generate Output (1-2 sentences)
        outputs = model.generate(
                        inputs, 
                        min_length=30,          # minimum length
                        max_length=500,         # maximum length
                        num_beams=5,            # keep track of top 5 possible sentences at a time
                        no_repeat_ngram_size=3, # N-grams of size three cannot be repeated -> no duplicate 3 word phrases in output
                        early_stopping=True)    # stop whenever num_beams begin generating <STOP> tokens
                    

        # Decode
        us_generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)


    ''' Chinese Summary '''
    print (f"Summarizing Chinese articles with \"{keyword}\"")
    key_text = "" # reset key_text for Chinese summary generation

    for i in range(len(chinese_data)):
        curr = chinese_data.iloc[i]['Full_English']

        if keyword in str(curr): # append based on if of keyword in article text

            # access the specific column by its name ['article'] & add to key_text
            key_text += str(curr) + "\n"
            
    if not key_text:
        print(f"No Chinese articles with \"{keyword}\"")
        key_text = "N/A"
        chinese_generated_summary = "N/A"
    else:
        # T5 requires the prefix "summarize: " 
        input_text = "Summarize: " + key_text

        # Encode inputs
        inputs = tokenizer.encode(
                        input_text,             # actual text to be summarized, begins with "Summarize: "
                        return_tensors='pt',    # encoded sequence will be returned as a PyTorch tensor object
                        max_length=512,         # max length is 512 tokens
                        truncation=True)        # anything longer will be removed

        # Generate Output (1-2 sentences)
        outputs = model.generate(
                        inputs, 
                        min_length=30,          # minimum length
                        max_length=500,         # maximum length
                        num_beams=5,            # keep track of top 5 possible sentences at a time
                        no_repeat_ngram_size=3, # N-grams of size three cannot be repeated -> no duplicate 3 word phrases in output
                        early_stopping=True)     # stop whenever num_beams begin generating <STOP> tokens
        

        # Decode
        chinese_generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print American and Chinese summaries
    print(f"Summary of \"{keyword}\" in American news: {us_generated_summary}")
    print(f"Summary of \"{keyword}\" in Chinese news: {chinese_generated_summary}")

