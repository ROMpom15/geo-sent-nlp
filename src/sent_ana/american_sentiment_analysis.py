# This program was written with the assistance of generative AI.

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/home/mids/m263624/si425/proj/geo-sent-nlp/src/sent_ana/local_distilbert_sst2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOPICS_EN = [
    "Taiwan Strait",
    "South China Sea",
    "American military",
    "Chinese military",
    "American navy", "US navy",
    "Chinese navy", "PLA navy",
    "shipbuilding",
    "economy"
]

FILE_PATH = "/home/mids/m263624/si425/proj/geo-sent-nlp/data/clean/raw/cnn_dailymail/train-00000-of-00003.parquet"
OUTPUT_PATH = "us_sentiment_output.csv"
LOCAL_PATH = "/home/mids/m263624/si425/proj/geo-sent-nlp/src/sent_ana/local_distilbert_sst2"

# LOAD MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)

# Force loading from pytorch_model.bin instead of safetensors
model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_PATH, 
    local_files_only=True,
    dtype=torch.float32,  
    ignore_mismatched_sizes=True
)

model.to(DEVICE)
model.eval()

# SENTIMENT FUNCTION
def get_sentiment_en(text: str) -> float:
    """Returns sentiment score between 0 (negative) and 1 (positive)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[0, 1].item()  # positive class probability

# TOPIC DETECTION
def detect_topics(text: str, topics: list) -> list:
    detected = [t for t in topics if t.lower() in text.lower()]
    return detected

# LOAD DATA
data = pd.read_parquet(FILE_PATH)
print("Loaded American dataset:", data.head())

# PROCESS ARTICLES
print("Starting fast batched sentiment analysis...")
print(f"Total articles in dataset: {len(data):,}")

data = data.head(30000).copy()  # Remove this line if you want to process ALL 287k articles
print(f"Processing {len(data):,} articles (remove .head() for full dataset)\n")

records = []
batch_size = 32
texts       = data["article"].tolist()
ids         = data["id"].tolist()
highlights  = data["highlights"].tolist()

print(f"Running inference in batches of {batch_size}...")

for i in range(0, len(texts), batch_size):
    batch_texts      = texts[i:i+batch_size]
    batch_ids      = ids[i:i+batch_size]
    batch_highlights = highlights[i:i+batch_size]

    # Batched tokenization & inference
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        sentiments = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()  # positive prob

    # Topic detection
    for j, text in enumerate(batch_texts):
        topics = detect_topics(text, TOPICS_EN)
        records.append({
            "id": batch_ids[j],
            "text": text,
            "highlights": batch_highlights[j],
            "sentiment": float(sentiments[j]),
            "topics": topics
        })

    # Progress update
    if (i // batch_size + 1) % 50 == 0 or i + batch_size >= len(texts):
        print(f"   Processed {min(i + batch_size, len(texts)):,}/{len(texts):,} articles")

# Create DataFrame once at the end
df_us = pd.DataFrame(records)

print(f"\nBatch processing complete! {len(df_us):,} articles analyzed.")

# CLUSTERING
topic_vectors = np.array([[1 if t in topics else 0 for t in TOPICS_EN] for topics in df_us["topics"]])
if len(df_us) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_us["cluster"] = kmeans.fit_predict(topic_vectors)
else:
    df_us["cluster"] = 0

# SAVE OUTPUT
df_us.to_csv(OUTPUT_PATH, index=False)
print("US analysis saved to:", OUTPUT_PATH)