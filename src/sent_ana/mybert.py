########### mybert.py
from transformers import DebertaTokenizer, DebertaConfig, DebertaModel

# Location of the pre-downloaded tokenizer and model 
cachepath = "/courses/nchamber/nlp/huggingface"
model_name = 'microsoft/deberta-base-mnli'

# Now load the tokenizer	    
tokenizer = DebertaTokenizer.from_pretrained(model_name, cache_dir=cachepath, local_files_only=True)
config = DebertaConfig.from_pretrained(model_name, output_hidden_states=True, cache_dir=cachepath, local_files_only=True)
model = DebertaModel.from_pretrained(model_name, config=config, cache_dir=cachepath, local_files_only=True)

def embed_sentence(str):
    inputs = tokenizer([str], padding=True, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0] # [0] grabs the first sentence in the given list
    return(last_hidden_states[0])

if __name__ == "__main__":
    embed_sentence('Hello I am delighted')