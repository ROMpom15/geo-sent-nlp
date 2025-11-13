import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os

# Load the model from our pre-downloaded directory
ID = "/SI425/Llama-2-7b-hf/"

# Define quantization settings (4-bit) to reduce model size when loaded.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # enable 4-bit quantization
    bnb_4bit_quant_type="nf4",          # use "normal float 4" quantization
    bnb_4bit_compute_dtype=torch.float16,  # compute in half precision
)

# Create the generation pipeline.
tokenizer = AutoTokenizer.from_pretrained(ID, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ID, quantization_config=bnb_config, dtype=torch.float16, device_map="auto", local_files_only=True, trust_remote_code=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt the model.
def translate_article(art_line):
output = pipe(f'Please translate the below news article from Chinese to English. Please keep the associated id number.\
                  for example: \
                  \'["今天天气很好,我们一起去公园散步吧." , "学习编程语言对未来的职业发展非常重要。" ]\',14' \
                 , max_new_tokens=) # add '\nsystem: ' to pipe input 
op = output[0]['generated_text']
print(op)

# Revised function incorporating better control. Originally generated with Gemini, revised for local use
def translate_article_with_control(art_line_with_id):
    # art_line_with_id should be a tuple/dict containing (article_text, article_id)
    article_text, article_id = art_line_with_id
    
    prompt = f"Translate the following Chinese news article to English. Output only the English translation and the ID in the format: ID:{article_id} Translation: [English Translation]. Article: {article_text}"
    
    output = pipe(
        prompt,
        max_new_tokens=120,
        return_full_text=False,  # Essential: only returns new tokens, maximizing translation length
        do_sample=False,         # Recommended for translation; reduces randomness
        batch_size=BATCH_SIZE    # Use this only if you pass a list of prompts
    )
    
    # You would then parse output[0]['generated_text'] to extract the Translation and ID.
    return output[0]['generated_text']