# Summary Analysis
# this file summarizes topic clusters into a 1-2 sentence overview
# compares summaries to analyze tone, emphasis, and framing
# - use lexicon list from lab 4 to identify sentimental words
# - use word vectors to find objective synonyms
import argparse
# import numpy # for synonyms
# import gensim.downloader as api
# from gensim.models.word2vec import Word2Vec # for synonyms
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# load the model
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# # load the tokenizer from disk, location of the pre-downloaded tokenizer and model 
# cachepath = "/courses/nchamber/nlp/huggingface"
# model_name = 'microsoft/deberta-base-mnli'

# #  load the tokenizer	    
# tokenizer = DebertaTokenizer.from_pretrained(model_name, cache_dir=cachepath, local_files_only=True)
# # load the DeBERTa model
# config = DebertaConfig.from_pretrained(model_name, output_hidden_states=True, cache_dir=cachepath, local_files_only=True)
# model = DebertaModel.from_pretrained(model_name, config=config, cache_dir=cachepath, local_files_only=True)


def generate_summary (text, topic):
    '''
    - Generates a 1-2 sentence summary of a topic cluster.
    - The summary should be as objective as possible.
    - The summary should also be without bias (tone, emphasis, and framing)
    - Returns the summary of queried topic
    '''
    # Step 1: generate a summary from the dataset given a topic
    # Step 2: remove subjective lexicon from summary -> make it objective
    # Step 3: ensure the summary is 1-2 sentences by summary.split() and checking if length <= 2
    # Step 4: combine/trim as necessary and return summary

    input_ids = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=512
    ).input_ids

    # actual summary generation
    output_ids = model.generate(
        input_ids,
        max_length=84,  # summary length requirement
        no_repeat_ngram_size=2,
        num_beams=4
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary

    
def map_subjective_to_objective(lexicon_list, subjective_word):
    '''
    - creates a dictionary of {subjective word : objective words (list)} mapping
    a subjective word to an objective synonym
    - This loads and runs prior to the execution of the summary generation and
    should only be called once
    - Use vectors to find synonyms
    - Returns a sorted list of objective synonyms (highest -> lowest)
    '''
    pass


# or pass a dictionary {subj : obj} and iterate over sentence, replacing via dictionary values
def substitute (summary_sentence, subjective_word, objective_word):
    '''
    - Function to switch the subjective word in the summary to its objective synonym
    - Returns the objectively modified summary
    '''
    pass



if __name__ == "__main__":
    negative_word_list = []
    positive_word_list = []
    
    # load the lists of negative and positive words
    with open ("negative-words.txt", encoding="utf-8", errors="ignore") as negative_word_file:
        negative_word_list = negative_word_file.read()
        
    with open ("positive-words.txt", encoding="utf-8", errors="ignore") as positive_word_file:
        positive_word_list = positive_word_file.read()
    
    
    
    # load data

    # 
