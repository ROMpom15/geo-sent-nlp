# Must be run on a GPU (On a lab machine)
# https://huggingface.co/haoranxu/ALMA-13B

# Import
import torch
# from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root_dir = os.path.abspath(os.path.join(script_dir,'../../'))
CNewSum_directory = os.path.join(proj_root_dir,'data','clean','CNewSum')
# model_path = "~/.cache/huggingface/hub/models--haoranxu--ALMA-13B"
model_path = "haoranxu/ALMA-13B"

# Constants
BATCH_SIZE = 16 # Adjustable
MAX_OUTPUT_TOKENS = 120

# Load base model weights
model = AutoModelForCausalLM.from_pretrained(
    "haoranxu/ALMA-13B", 
    dtype=torch.float16, 
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

# Much of these functions code came from Gemini's 2.5 flash model. All code was edited
# and understood before implemented. We decided to implement 
def batch_translate(all_articles, tokenizer=tokenizer):
    """
    Translates articles in batches using the ALMA 13B pipeline.
    """
    results = []
    
    for i in range(0, len(all_articles), BATCH_SIZE): # step through length of article list one batch ata time
        batch = all_articles[i : i + BATCH_SIZE]
        
        # Create a specific prompt for each item in the batch
        batch_prompts = []
        for article,id in batch:
            art_len = len(article)
            for sentence in article:
                prompt = (
                    f"Please translate the following Chinese news article to English. "
                    f"Output ONLY the English translation and the ID in this exact format: "
                    f"ID:{id}.{sentence_count} TRANSLATION: [English Translation]. "
                    f"Article: {sentence}"
                )
                batch_prompts.append(prompt)
                #   # Add the source setence into the prompt template
                #     prompt=f"Translate this from Chinese to English:\nChinese: {text_to_translate}。\nEnglish:"
                #     print(prompt)
                #     input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=MAX_OUTPUT_TOKENS, truncation=True).input_ids.cuda()

                #     # Translation
                #     with torch.no_grad():
                #         generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=MAX_OUTPUT_TOKENS, return_full_text=False, do_sample=True, temperature=0.6, top_p=0.9)
                #     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                #     print(outputs)
            # ⭐️ The optimized pipeline call
                input_ids = tokenizer( ###############################################################
                    batch_prompts,
                    return_tensors="pt",
                    max_new_tokens=MAX_OUTPUT_TOKENS,
                    return_full_text=False,
                    do_sample=False
                )

        # Collect results
        for item, output in zip(batch, batch_output):
            # The generated text contains the ID and Translation due to the prompt design
            generated_text = output[0]['generated_text'].strip()
            
            # Simple parsing/cleaning logic (you may need a more robust regex parser)
            try:
                # Assuming the model follows the requested format
                translation_start = generated_text.find("TRANSLATION:") + len("TRANSLATION:")
                translation = generated_text[translation_start:].strip()
            except:
                translation = generated_text # Fallback to raw text

            results.append({
                'id': item['id'],
                'original_text': item['chinese_text'],
                'translated_text': translation
            })
            
    return results

if __name__ == "__main__":
    test_translate()