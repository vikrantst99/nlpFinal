from pathlib import Path
from datetime import datetime

import pandas as pd
import feedparser


RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DATA_DIR / "nvidia_articles.csv"


RSS_FEEDS = {
    "NVIDIA Newsroom": "https://nvidianews.nvidia.com/releases.xml",
    "NVIDIA Blog": "https://feeds.feedburner.com/nvidiablog",
    "NVIDIA Developer Blog": "https://developer.nvidia.com/blog/feed",

    "Google News - NVIDIA Gaming": "https://news.google.com/rss/search?q=NVIDIA%20gaming%20GPU%20GeForce%20NOW&hl=en-US&gl=US&ceid=US:en",
    "Google News - NVIDIA AI Chips": "https://news.google.com/rss/search?q=NVIDIA%20AI%20chips%20Blackwell%20GPU&hl=en-US&gl=US&ceid=US:en",
    "Google News - NVIDIA Supply Chain": "https://news.google.com/rss/search?q=NVIDIA%20supply%20chain%20TSMC%20HBM&hl=en-US&gl=US&ceid=US:en",
}


def collect_articles():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        print("Collecting from:")
        print(source_name)

        feed = feedparser.parse(feed_url)

        for i in feed.entries:
            article = {
                "source": source_name,
                "title": i.get("title", ""),
                "url": i.get("link", ""),
                "published": i.get("published", ""),
                "summary": i.get("summary", ""),
                "collected_at": datetime.now().isoformat(timespec="seconds"),
            }

            articles.append(article)

    new_df = pd.DataFrame(articles)

    print("Before cleaning:")
    print(new_df.shape)

    new_df = new_df.drop_duplicates(subset=["url"])
    new_df = new_df[new_df["url"].notna()]
    new_df = new_df[new_df["url"] != ""]

    print("After cleaning:")
    print(new_df.shape)

    new_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved article URL file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    collect_articles()