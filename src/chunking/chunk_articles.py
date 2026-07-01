import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():

    df = pd.read_csv("data/processed/nvidia_articles_cleaned.csv")

    print("Total articles loaded:", len(df))

    # PARAMETER CHANGE ALERT
    # chunk_size = how many characters go into one chunk, chunk_overlap = shared characters between consecutive chunks
    # if asked to change chunk size live, just edit the numbers below, rest of the code stays the same
    # docs: https://python.langchain.com/docs/how_to/recursive_text_splitter/
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

    chonks = []

    for i, j in df.iterrows():
        texts = j["full_text_clean"]

        if not isinstance(texts, str) or len(texts.strip()) == 0:
            continue

        full_chunks = splitter.split_text(texts)

        for chunk_id, chunk_text in enumerate(full_chunks):
            chunk_row = {
                "doc_id":           i,
                "chunk_id":         f"doc_{i}_chunk_{chunk_id}",
                "source":           j["source"],
                "title":            j["title"],
                "url":              j["url"],
                "published":        j["published"],
                "matched_keywords": j["matched_keywords"],
                "relevance_score":  j["relevance_score"],
                "chunk_text":       chunk_text,
                "char_count":       len(chunk_text)
            }
            chonks.append(chunk_row)

    print("Total chunks created:", len(chonks))

    df_chunks = pd.DataFrame(chonks)
    df_chunks.to_csv("data/chunked/nvidia_article_chunks.csv", index=False)

    print("Saved to data/chunked/nvidia_article_chunks.csv")


if __name__ == "__main__":
    main()