# Summary Analysis
# this file summarizes topic clusters into a 1-2 sentence overview
# compares summaries to analyze tone, emphasis, and framing
# - use lexicon list from lab 4 to identify sentimental words
# https://medium.com/@ivavrtaric/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771

# pip install torch transformers
# pip install pip-system-certs

# import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead


# load the model
model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name, return_dict=True)

original_text = """
Kobe Bryant stands as a towering figure in the pantheon of sports history, 
revered not only for his statistical dominance but for the unyielding philosophy he 
termed the "Mamba Mentality." A lifelong Los Angeles Laker, Bryant secured five NBA 
championships, 18 All-Star selections, and the 2008 MVP award over a distinctively 
explosive 20-year career. However, his influence transcended the hardwood; he evolved 
into an Oscar-winning storyteller, a dedicated father to four daughters, and a fervent 
advocate for the growth of women's basketball. Though his life was tragically cut short 
in 2020, his legacy of obsessive preparation and fearless competition continues to 
serve as the gold standard for athletes and dreamers worldwide.
"""

original_text1 = """
On January 22, 2006, Kobe Bryant etched his name into NBA history by scoring 81 points against the Toronto Raptors, the second-highest single-game total ever recorded. Over his 20-season tenure with the Los Angeles Lakers, Bryant amassed 33,643 career points, five NBA championships, and two Olympic gold medals with Team USA in 2008 and 2012. His excellence was further recognized off the court in 2018, when he became the first professional athlete to win an Academy Award for his animated short film, Dear Basketball.
"""

original_text2 = """
Kobe Bryant is often defined less by the trophies on his shelf than by the psychological intensity he brought to his craft, a mindset he famously dubbed the 'Mamba Mentality.' This philosophy centered on an obsessive focus on self-improvement, resilience in the face of failure, and a 'killer instinct' that intimidated opponents before the game even began. For millions of fans and fellow athletes, Bryant represents the ultimate symbol of hard work, proving that greatness is not just a talent, but a relentless daily habit.
"""

original_text3 = """
After hanging up his jersey, Kobe Bryant successfully pivoted from competitive sports to becoming a champion for women's athletics and a creative storyteller. Embracing his role as a 'Girl Dad,' he became a highly visible mentor for the next generation of female basketball players, including his daughter Gianna, while simultaneously building a multimedia production company. His second act was defined by a desire to inspire youth through fantasy novels and film, aiming to teach life lessons through the lens of sports.
"""


new_text = "summarize: " + original_text + original_text1 + original_text2 + original_text3

# inputs=tokenizer.encode("summarize: " + topic, return_tensors='pt', max_length=512, truncation=True)
inputs=tokenizer.encode(
                        new_text, 
                        return_tensors='pt', 
                        max_length=512, 
                        truncation=True)

# output is a tensor
output = model.generate(inputs, 
                        min_length=80, 
                        max_length=100, 
                        num_beams=5, 
                        no_repeat_ngram_size=3, 
                        early_stopping=True) 

summary = tokenizer.decode(output[0], skip_special_tokens=True)


print (summary)

# Kobe Bryant's legacy of obsessive preparation and fearless competition continues to serve as the gold standard for athletes and dreamers worldwide. Though his life was tragically cut short in 2020, his legacy of obsession and fearlessness still serves as the golden standard for athlete and dreamer worldwide. Picture courtesy of TMZ.com. All rights reserved.