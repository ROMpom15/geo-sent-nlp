# This program was written with the assistance of generative AI.

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import numpy as np

file_path = "/home/mids/m263624/si425/proj/geo-sent-nlp/data/clean/raw/cnn_dailymail/train-00000-of-00003.parquet"

data = pd.read_parquet(file_path)

# Available columns: article, highlights, id
print("Loaded American dataset:", data.head())

# SENTIMENT FUNCTION 
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_en(text):
    scores = analyzer.polarity_scores(text)
    return (scores["compound"] + 1) / 2  # normalize to 0–1


# TOPICS (ENGLISH VERSION)
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

def detect_topics(text, topics):
    detected = []
    for t in topics:
        if t.lower() in text.lower():
            detected.append(t)
    return detected


# PROCESSING ARTICLES
records = []

for idx, row in data.iterrows():
    article = row["article"]
    detected_topics = detect_topics(article, TOPICS_EN)
    sentiment_score = get_sentiment_en(article)

    records.append({
        "id": row["id"],
        "text": article,
        "highlights": row["highlights"],
        "sentiment": sentiment_score,
        "topics": detected_topics
    })

df_us = pd.DataFrame(records)

# CLUSTERING 
topic_vectors = []

for topics in df_us["topics"]:
    vec = [1 if t in topics else 0 for t in TOPICS_EN]
    topic_vectors.append(vec)

topic_vectors = np.array(topic_vectors)

if len(df_us) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_us["cluster"] = kmeans.fit_predict(topic_vectors)
else:
    df_us["cluster"] = 0


# SAVE OUTPUT
output_path = "us_sentiment_output.csv"
df_us.to_csv(output_path, index=False)
print("US analysis saved to:", output_path)


