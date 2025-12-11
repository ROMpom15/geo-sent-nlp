# This program was written with the assistance of generative AI.
import pandas as pd

# LOAD OUTPUT FILES
df_cn = pd.read_csv("china_sentiment_output.csv")
df_us = pd.read_csv("us_sentiment_output.csv")

print("Chinese dataset loaded:", df_cn.shape)
print("US dataset loaded:", df_us.shape)

# COMPARE SENTIMENT AVERAGES
cn_avg = df_cn["sentiment"].mean()
us_avg = df_us["sentiment"].mean()

print("\n=== AVERAGE SENTIMENT SCORES ===")
print(f"Chinese news sentiment: {cn_avg:.3f}")
print(f"American news sentiment: {us_avg:.3f}")

# COMPARISON BY TOPICS
def explode_topics(df):
    return df.explode("topics").groupby("topics")["sentiment"].mean()

print("\n=== SENTIMENT BY TOPIC (CHINA) ===")
print(explode_topics(df_cn))

print("\n=== SENTIMENT BY TOPIC (US) ===")
print(explode_topics(df_us))

# SAVE COMPARISON OUTPUT
comparison_df = pd.DataFrame({
    "Chinese_avg_sentiment": [cn_avg],
    "US_avg_sentiment": [us_avg]
})

comparison_df.to_csv("comparison_report.csv", index=False)
print("\nSaved comparison report to comparison_report.csv")
