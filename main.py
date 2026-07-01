import subprocess
import sys


PIPELINE = [
    ("Step 1 - Collecting article URLs from RSS feeds",      "src/collection/collect_articles.py"),
    ("Step 2 - Extracting full text from each article URL",  "src/collection/extract_full_text.py"),
    ("Step 3 - Cleaning article texts",                      "src/cleaning/clean_articles.py"),
    ("Step 4 - Chunking cleaned articles",                   "src/chunking/chunk_articles.py"),
    ("Step 5 - Embedding chunks and building FAISS index",   "src/embeddings/embed_and_index.py"),
    ("Step 6 - Running AI CEO Agent briefing",                "src/agent/rag_agent.py"),
]


def run_step(step_name, script_path):
    print("")
    print("=" * 60)
    print(step_name)
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        check=False
    )

    if result.returncode != 0:
        print("")
        print("ERROR: Step failed — " + step_name)
        print("Pipeline stopped.")
        sys.exit(1)

    print("")
    print("Done: " + step_name)


def main():
    print("")
    print("=" * 60)
    print("AI CEO NVIDIA STRATEGIC INTELLIGENCE PIPELINE")
    print("=" * 60)

    for step_name, script_path in PIPELINE:
        run_step(step_name, script_path)

    print("")
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("To launch the dashboard run:")
    print("streamlit run src/dashboard/dashboard.py")
    print("=" * 60)
    print("")


if __name__ == "__main__":
    main()