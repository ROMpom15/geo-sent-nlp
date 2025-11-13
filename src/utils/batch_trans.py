# Much of this code came from Gemini's 2.5 flash model. All code was edited
# Import
import pandas as pd
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root_dir = os.path.abspath(os.path.join(script_dir,'../../'))
CNewSum_directory = os.path.join(proj_root_dir,'data','clean','CNewSum')

def load_and_prepare_data(CNewSum_directory, text_column, id_column, type='dev'):
    """Loads data from all CSV files in a directory and formats it for the pipeline."""
    all_files = glob.glob(os.path.join(CNewSum_directory, f"{type}*.csv"))
    
    data_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        for index, row in df.iterrows(): #iterate through every row
            data_list.append({
                'id': row[id_column],
                'chinese_text': row[text_column]
            })
            
    return data_list

# Example Usage
# all_articles = load_and_prepare_data('./my_csv_data_folder', 'ChineseArticle', 'ArticleID')

# Constants
BATCH_SIZE = 16 # Adjust this based on your GPU memory (start low and increase)
MAX_OUTPUT_TOKENS = 120

def batch_translate(pipe, all_articles):
    """
    Translates articles in batches using the Hugging Face pipeline.
    """
    results = []
    
    for i in range(0, len(all_articles), BATCH_SIZE):
        batch = all_articles[i : i + BATCH_SIZE]
        
        # Create a specific prompt for each item in the batch
        batch_prompts = []
        for item in batch:
            prompt = (
                f"Please translate the following Chinese news article to English. "
                f"Output ONLY the English translation and the ID in this exact format: "
                f"ID:{item['id']} TRANSLATION: [English Translation]. "
                f"Article: {item['chinese_text']}"
            )
            batch_prompts.append(prompt)

        # ⭐️ The optimized pipeline call
        batch_output = pipe(
            batch_prompts,
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