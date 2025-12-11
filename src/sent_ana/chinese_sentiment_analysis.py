# This program was written with the assistance of generative AI.

import os
import pandas as pd
from snownlp import SnowNLP
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np

# Define the topics to track
topics = [
    "Taiwan Strait", "South China Sea", "American military forces", "Chinese military forces", 
    "American navies", "Chinese navies", "shipbuilding", "economy"
]

# Load the Chinese news dataset
# Would have used the results from the translation model here
file_path = os.path.join(os.path.expanduser("~"), "train.simple.label.jsonl")
us_data_load = load_dataset("parquet", data_files={'train': file_path})
us_data = us_data_load["train"] 

# Function to perform sentiment analysis using snownlp
def get_sentiment(text):
    s = SnowNLP(text)
    return s.sentiments  # Returns a sentiment score between 0 and 1

# Function to check if any topic keywords appear in the article
def find_topics_in_text(text, topics):
    detected_topics = []
    for topic in topics:
        if topic in text:
            detected_topics.append(topic)
    return detected_topics

# Prepare a list to store sentiment results and related topics
sentiment_data = []

for i in range(len(us_data)):
    # Accessing the news article (replace 'text_column' with the correct column name from your dataset)
    article_text = us_data[i]['text_column']  # Assuming each article is in the 'text_column'

    # Get the sentiment of the article
    sentiment_score = get_sentiment(article_text)

    # Find topics mentioned in the article
    detected_topics = find_topics_in_text(article_text, topics)

    # Store the sentiment and topics
    sentiment_data.append({
        'article': article_text,
        'sentiment': sentiment_score,
        'detected_topics': detected_topics
    }) 

# Convert the sentiment data to a DataFrame for easier manipulation
df_sentiment = pd.DataFrame(sentiment_data)

# Print out some of the results
print(df_sentiment.head())

# Cluster the sentiment data based on detected topics
# First, vectorize the detected topics for clustering
topic_vectors = []
for index, row in df_sentiment.iterrows():
    topic_vector = [0] * len(topics)  # Initialize a vector of zeros
    for topic in row['detected_topics']:
        topic_index = topics.index(topic)  # Find the index of the topic in the topics list
        topic_vector[topic_index] = 1  # Mark the topic as present
    topic_vectors.append(topic_vector)

# Convert to a NumPy array for clustering
topic_vectors = np.array(topic_vectors)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for exampl
df_sentiment['cluster'] = kmeans.fit_predict(topic_vectors)

# group the articles by cluster and analyze the sentiment
for cluster in range(3): 
    print(f"\n--- Cluster {cluster} ---")
    cluster_data = df_sentiment[df_sentiment['cluster'] == cluster]
    cluster_sentiment_avg = cluster_data['sentiment'].mean()
    print(f"Average Sentiment in Cluster {cluster}: {cluster_sentiment_avg:.2f}")
    print(cluster_data[['article', 'detected_topics', 'sentiment']].head())
