# Must be run on a GPU (On a lab machine)
# https://huggingface.co/haoranxu/ALMA-13B

# Import
import torch
# from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaTokenizer
import os
import read_func as rf


script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root_dir = os.path.abspath(os.path.join(script_dir,'../../'))
CNewSum_directory = os.path.join(proj_root_dir,'data','clean','CNewSum')
# model_path = "~/.cache/huggingface/hub/models--haoranxu--ALMA-13B"
model_path = "haoranxu/ALMA-13B"

# Constants
BATCH_SIZE = 1 # Adjustable!!
MAX_OUTPUT_TOKENS = 256

# 1. Configure 4-bit quantization
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model weights
model = AutoModelForCausalLM.from_pretrained(
    "haoranxu/ALMA-13B", 
    dtype=torch.float16, 
    quantization_config=nf4_config,
    device_map="auto",
    force_download=False,      # <--- This forces a fresh download. Nuclear option
    local_files_only=True    # <--- Set True for default, talk to Conrad before changing
)
# model = PeftModel.from_pretrained(model, "haoranxu/ALMA-13B") # Don't need, already using a fine-tuned model
tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')

def test_translate(tokenizer=tokenizer,text_to_translate='我爱机器翻译'):
    # Add the source setence into the prompt template
    prompt=f"Translate this from Chinese to English:\nChinese: {text_to_translate}。\nEnglish:"
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=MAX_OUTPUT_TOKENS, truncation=True).input_ids.cuda()

    # Translation
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=MAX_OUTPUT_TOKENS, return_full_text=False, do_sample=True, temperature=0.6, top_p=0.9)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(outputs)

# Much of these below functions code came from Gemini's 2.5 flash model. All code was edited
# and understood before implemented. We decided to implement 
def batch_translate(flat_art, model, tokenizer=tokenizer):
    """
    Translates articles in batches using the ALMA 13B pipeline.

    input: flattened Chinese articles: [(id1, 0, "sent1"), (id1, 1, "sent2"), ...]
    """
    
    # Specificy prompt construction for ALMA.
    prompts = [
        f"Translate this from Chinese to English:\nChinese: {item['text']}\nEnglish:" 
        for item in flat_art
    ] # Generate a list of all the prompts from our flattened Chinese sentances

    # Tokenize
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512 # shouldn't be a problem. Each flattened sentence is usually <= 200 tokens.
    ).to(model.device)

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=5,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    # Decode
    raw_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # --- Clear memory immediately after generation ---
    del inputs
    del generated_ids
    torch.cuda.empty_cache()

    # Clean/Parse results
    clean_translations = []
    for raw, prompt in zip(raw_outputs, prompts):
        # Logic: Find where the prompt ends and take the rest
        split_marker = "English:"
        if split_marker in raw:
            clean_text = raw.split(split_marker)[-1].strip()
        else: # HOPEFULLY SHOULDN'T HAPPEN
            # Fallback: remove prompt string manually
            print('fallback')
            clean_text = raw.replace(prompt, "").strip()
        clean_translations.append(clean_text)

    return clean_translations

def mass_trans(all_articles, model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE, output_dir=CNewSum_directory):
    """
    The main function calling all helpers. Generated with Gemini.
    """
    # 1. Setup
    csv_path = rf.setup_dir(CNewSum_directory)
    
    # 2. Prepare Data
    flat_sentences = rf.flatten_sent(all_articles)
    total = len(flat_sentences)
    print(f"Starting translation for {total} sentences...")

    # 3. Loop through batches
    for i in range(0, total, batch_size):
        # Slice the batch
        batch_items = flat_sentences[i : i + batch_size]
        
        # Run Model
        translations = batch_translate(batch_items, model, tokenizer)
        
        # Save Results
        rf.save_batch(csv_path, batch_items, translations)
        
        # Progress bar
        if (i // batch_size) % 5 == 0:
            print(f"Processed {min(i + batch_size, total)}/{total}...")

    print(f"Job Complete. Results saved to: {csv_path}")
    return csv_path

if __name__ == "__main__":
    #test_translate()
    # test = rf.load_zho_cnewsum_data(type='train')[0:10]
    # fin_path = mass_trans(test)

    to_translate = []
    to_translate.extend(rf.load_zho_cnewsum_data(type='train'))
    to_translate.extend(rf.load_zho_cnewsum_data(type='dev'))
    to_translate.extend(rf.load_zho_cnewsum_data(type='test'))
    fin_path = mass_trans(to_translate[9376:])
