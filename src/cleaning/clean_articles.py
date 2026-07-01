import pandas as pd
import re

filepath = "data/processed/nvidia_articles_with_text.csv"
df = pd.read_csv(filepath)

short_articles = df[df["word_count"] < 500]

print("Number of articles below 500 words:")
print(len(short_articles))

print(short_articles[["source", "title", "url", "word_count"]])

df_clean = df.copy()
df_clean["full_text_clean"] = df_clean["full_text"]

def clean_article_text(text):
    text = str(text)
    # Remive url
    text = re.sub(r"http\S+|www\S+", " ", text)
    # We do not have em but JIC :-
    emoji_pattern = r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
    text = re.sub(emoji_pattern, " ", text)
    # Whitespacex MOST IMPORTANT:-
    text = " ".join(text.split())
    return text


df_clean["full_text_clean"] = df_clean["full_text"].apply(clean_article_text)


output_path = "data/processed/nvidia_articles_cleaned.csv"

df_clean.to_csv(output_path, index=False)