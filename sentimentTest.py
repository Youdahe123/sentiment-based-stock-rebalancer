from transformers import pipeline
import pandas as pd

classifier = pipeline('sentiment-analysis')
df = pd.read_csv('posts.csv')
df_sample = df.head(20).copy()
df_sample["sentiment"] = df_sample['selftext'].astype(str).apply(lambda x: classifier(x[:512])[0]['label'])
print(df_sample[["selftext","sentiment"]])
df_sample.to_csv("sentiment_resut.csv")


# selftext sentiment
# 0   AAPL just entered a contract to purchase 51 of...  NEGATIVE
# 1   What kind of an upgrade was this? It just suck...  NEGATIVE
# 2                                                 NaN  POSITIVE
# 3                                                 NaN  POSITIVE
# 4   This will post every other Monday (EST) at 6AM...  NEGATIVE
# 5                                                 NaN  POSITIVE
# 6   # tl;dr\n\n - Stock picking game will last all...  POSITIVE
# 7                                           [deleted]  NEGATIVE
# 8   **Steam Winter Sale 2017** - [Day 1](https://r...  NEGATIVE
# 9   I absolutely love McDavid's Evo but I compared...  NEGATIVE
# 10                                                NaN  POSITIVE
# 11                                                NaN  POSITIVE
# 12  Looking at AAPLs fundamentals and the pile of ...  NEGATIVE
# 13                                                NaN  POSITIVE
# 14                                          [removed]  NEGATIVE
# 15                                                NaN  POSITIVE
# 16                                                NaN  POSITIVE
# 17  I’ve decided I’m most likely interested in jus...  NEGATIVE
# 18  I've been on the Robinhood platform since earl...  POSITIVE
# 19  I’ve decided I’m most likely interested in jus...  NEGATIVE