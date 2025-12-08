# Summary Analysis
# this file summarizes topic clusters into a 1-2 sentence overview
# 95704

# the t5 code comes from https://medium.com/@ivavrtaric/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771 and adapted to the use of 
# the NLP project. 


# pip install torch transformers
# pip install pip-system-certs
# may have to pip install the system certs to get 

# import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead
import os
from datasets import load_dataset

# load the model
model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name, return_dict=True)

# load the US data
file_path = os.path.join(os.path.expanduser("~"), "train-00000-of-00003.parquet") # absolute path to the dataset
us_data_load = load_dataset("parquet", data_files={'train': file_path})
us_data = us_data_load["train"]

# initialize variables for user input loop
keyword = ""
# key_text = ""

# loop until user types "quit"
while (keyword != "quit"):
    key_text = "" # reset key_text for next keyword search (keyword is already reset)

    keyword = input("Enter a keyword to summarize articles about (or type 'quit' to exit): ")
    if keyword == "quit":
        break
    
    for i in range(len(us_data)):
        if keyword in us_data[i]['article']: # append based on if of keyword in article text

            # access the specific column by its name ['article'] & add to key_text
            key_text += us_data[i]['article'] + "\n"
            

    # --- FEEDING IT INTO T5 ---
    # T5 requires the prefix "summarize: " 
    input_text = "summarize: " + key_text

    # Encode inputs
    inputs = tokenizer.encode(
                    input_text,             # actual text to be summarized, begins with "Summarize: "
                    return_tensors='pt',    # encoded sequence will be returned as a PyTorch tensor object
                    max_length=512,         # max length is 512 tokens
                    truncation=True)        # anything longer will be removed

    # Generate Output (using your preferred 1-2 sentence settings)
    outputs = model.generate(
        inputs, 
        min_length=30,              # minimum length
        max_length=500,             # maximum length
        num_beams=5,                # keep track of top 5 possible sentences at a time
        no_repeat_ngram_size=3,     # N-grams of size three cannot be repeated -> no duplicate 3 word phrases in output
        early_stopping=True         # stop whenever num_beams begin generating <STOP> tokens
    )

    # Decode and print
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Summary of \"{keyword}\" in the news: {generated_summary}")
