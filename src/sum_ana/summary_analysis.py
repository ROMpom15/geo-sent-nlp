# Summary Analysis
# this file summarizes topic clusters into a 1-2 sentence overview

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
# print(us_data.column_names)
# print(us_data[0])


for i in range(len(us_data)):
    
    # --- ACCESSING THE HIGHLIGHTS COLUMN ---
    # We access the specific column by its name ['highlights']
    highlight_text = us_data[i]['highlights']
    
    # print(f"\n--- Processing Record {i} ---")
    # print(f"Original Highlight Text: {highlight_text[:100]}...") # Print first 100 chars

    # --- FEEDING IT INTO T5 ---
    # T5 requires the prefix "summarize: " for this task
    input_text = "summarize: " + highlight_text

    # Encode inputs
    inputs = tokenizer.encode(
        input_text, 
        return_tensors='pt', 
        max_length=512, 
        truncation=True
    )

    # Generate Output (using your preferred 1-2 sentence settings)
    outputs = model.generate(
        inputs, 
        min_length=15, 
        max_length=500, # 50 
        num_beams=5, 
        no_repeat_ngram_size=3, 
        early_stopping=True
    )

    # Decode and print
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"T5 Summary: {generated_summary}")







































# new_text = "summarize: " +  us_data

# # inputs=tokenizer.encode("summarize: " + topic, return_tensors='pt', max_length=512, truncation=True)
# inputs = tokenizer.encode(
#                     new_text,               # actual text to be summarized, begins with "Summarize: "
#                     return_tensors='pt',    # encoded sequence will be returned as a PyTorch tensor object
#                     max_length=512,         # max length is 512 tokens
#                     truncation=True)        # anything longer will be removed

# # output is a tensor
# output = model.generate(inputs, 
#                         min_length=15,          # minimum length
#                         max_length=50,          # maximum length
#                         num_beams=5,            # keep track of top 5 possible sentences at a time
#                         no_repeat_ngram_size=3, # N-grams of size three cannot be repeated -> no duplicate 3 word phrases in output
#                         early_stopping=True)    # stop whenever num_beams begin generating <STOP> tokens

# summary = tokenizer.decode(output[0], skip_special_tokens=True)

# print (summary)

# query = ""

# # loop for user queries
# while (query != "quit"):
#     query = input("What topic would you like summarized (Formatted as \"Country:Topic\"?)" )

#     new_text = "summarize in 2 sentences: "




#     inputs = tokenizer.encode(
#                         new_text,               # actual text to be summarized, begins with "Summarize: "
#                         return_tensors='pt',    # encoded sequence will be returned as a PyTorch tensor object
#                         max_length=512,         # max length is 512 tokens
#                         truncation=True)        # anything longer will be removed

#     # output is a tensor
#     output = model.generate(inputs, 
#                             min_length=15,          # minimum length
#                             max_length=50,          # maximum length
#                             num_beams=5,            # keep track of top 5 possible sentences at a time
#                             no_repeat_ngram_size=3, # N-grams of size three cannot be repeated -> no duplicate 3 word phrases in output
#                             early_stopping=True)    # stop whenever num_beams begin generating <STOP> tokens

#     summary = tokenizer.decode(output[0], skip_special_tokens=True)

#     print (summary)