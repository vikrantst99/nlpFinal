from pathlib import Path
import time

import pandas as pd
import trafilatura
from ftfy import fix_text


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

INPUT_FILE = RAW_DATA_DIR / "nvidia_articles.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "nvidia_articles_with_text.csv"

MINIMUM_WORDS = 300


def extract_text_from_url(url):
    downloaded_page = trafilatura.fetch_url(url)

    if downloaded_page is None:
        return ""

    full_text = trafilatura.extract(
        downloaded_page,
        include_links=False,
        include_comments=False,
        include_tables=False,
    )

    if full_text is None:
        return ""

    full_text = fix_text(full_text)

    return full_text


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_df = pd.read_csv(INPUT_FILE)

    full_texts = []

    for i, j in new_df.iterrows():
        url = j["url"]

        print("Extracting article:")
        print(i + 1)
        print(url)

        full_text = extract_text_from_url(url)

        full_texts.append(full_text)

        time.sleep(1)

    new_df["full_text"] = full_texts

    new_df["word_count"] = new_df["full_text"].apply(lambda x: len(str(x).split()))

    print("Before filtering:")
    print(new_df.shape)

    new_df = new_df[new_df["full_text"].notna()]
    new_df = new_df[new_df["full_text"] != ""]
    new_df = new_df[new_df["word_count"] >= MINIMUM_WORDS]

    new_df = new_df.drop_duplicates(subset=["url", "full_text"])

    print("After filtering:")
    print(new_df.shape)

    new_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved full-text file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()