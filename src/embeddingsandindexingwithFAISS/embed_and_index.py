import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os


def main():

    df = pd.read_csv("data/chunked/nvidia_article_chunks.csv")

    alles = []
    for i, j in df.iterrows():
        doc = Document(page_content=j["chunk_text"],
                        metadata={"chunk_id":   j["chunk_id"],
                                  "source":     j["source"],
                                  "title":      j["title"],
                                  "URL":        j["url"],
                                  "published":  j["published"]})
        alles.append(doc)

    # docs: https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/
    model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

    vector_store = FAISS.from_documents(documents=alles, embedding=model)

    save_dir = "data/embeddings/EMBEDDINGS&FAISS"
    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)

if __name__ == "__main__":
    main()