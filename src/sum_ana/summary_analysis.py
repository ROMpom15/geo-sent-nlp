# Summary Analysis
# this file summarizes topic clusters into a 1-2 sentence overview
# compares summaries to analyze tone, emphasis, and framing
# - use lexicon list from lab 4 to identify sentimental words
# - use word vectors to find objective synonyms

# !pip install torch transformers

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead


# load the model
model_name = 'T5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name, return_dict=True)

sequence = ""

inputs=tokenizer.encode("sumarize: " + sequence,return_tensors='pt', max_length=512, truncation=True)

output = model.generate(inputs, min_length=80, max_length=100)

summary = tokenizer.decode(output[0])
print (summary)