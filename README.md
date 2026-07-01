# AI CEO: NVIDIA Strategic Intelligence Agent

This is my final exam project. The goal was to build an AI agent that can actually act like a strategic advisor for NVIDIA — not just answer questions, but plan, search for evidence, reason over it, and give recommendations a CEO could actually use.

I picked NVIDIA because there's a ton of public news/blog content on them and it made the data collection part way more interesting than something with less news coverage.

## How the pipeline works

```
collect_articles.py + extract_full_text.py
        |
        v
clean_articles.py
        |
        v
chunk_articles.py  (RecursiveCharacterTextSplitter)
        |
        v
embed_and_index.py  (BAAI/bge-base-en-v1.5 + FAISS)
        |
        v
rag_agent.py  (hybrid retrieval + LangChain agent)
        |
        v
dashboard.py  (Streamlit)
```

`main.py` just runs all of these one after another with subprocess, and stops if anything breaks.

### Architecture, roughly

```
                Layer 1 - Data Collection
                (RSS feeds, scraping NVIDIA sources)
                          |
                Layer 2 - Cleaning
                (dedup, normalize text, keep raw text too)
                          |
                Layer 3 - Chunking
                (LangChain text splitter, 1000 chars, 250 overlap)
                          |
                Layer 4 - Embeddings + FAISS
                (BAAI/bge-base-en-v1.5, 768-dim)
                          |
        ----------------------------------------
        |             Layer 5                   |
        |                                        |
        |   FAISS retriever     BM25 retriever   |
        |        \                  /            |
        |         EnsembleRetriever               |
        |              |                          |
        |     search_nvidia_knowledge tool          |
        |     get_source_details tool                |
        |              |                          |
        |     ChatOllama (Qwen3 8B, local)            |
        |              |                          |
        |     create_tool_calling_agent                |
        |     + AgentExecutor                            |
        |     + RunnableWithMessageHistory (memory)        |
        ----------------------------------------
                          |
                Layer 6 - Dashboard
                (Streamlit, 7 sections)
```

## Tech stack

- LangChain — used as the orchestration layer across literally everything (chunking, embeddings, vector store, agent), since this was a hard requirement
- feedparser / trafilatura / ftfy — for the collection + cleaning step
- BAAI/bge-base-en-v1.5 — embedding model, 768 dims, L2 normalized
- FAISS — vector store (`IndexFlatIP` under the hood via LangChain's wrapper)
- BM25Retriever — keyword search, combined with FAISS through `EnsembleRetriever`
- Qwen3 8B via Ollama — local LLM, no external API calls anywhere
- Streamlit — dashboard

No OpenAI/Anthropic/Gemini anywhere in this, per the assignment rules.

## Why hybrid search (FAISS + BM25)

Semantic search alone kept missing exact product names. Like if you search "Blackwell B200" it sometimes pulls back something about GPUs in general instead of the actual chunk that mentions B200 specifically, because the overall meaning of that chunk drifts toward something else. BM25 fixes that since it just matches words directly, but BM25 alone is bad at understanding that "chip shortage" and "semiconductor supply constraints" mean the same thing. So I combined both with `EnsembleRetriever`, 50/50 weights.

## Why this counts as an agent and not just RAG

A basic RAG setup is just `prompt -> LLM -> answer`. That's not what this is. The agent here:

- plans what to search for before doing anything
- decides on its own which tool to call and when (search_nvidia_knowledge vs get_source_details)
- can call search multiple times if it doesn't have enough evidence yet
- has to validate that it's actually citing real sources before it gives a final answer

This whole loop (goal -> plan -> retrieve -> analyze -> decide -> recommend -> validate) is enforced through the system prompt and through how the tools are written, not just hardcoded into the response.

## Other decisions I made

- Went with FAISS over ChromaDB since the dataset isn't huge (~120 articles, under 2000 chunks) so I don't need approximate search, exact search with `IndexFlatIP` is fine and just simpler to reason about.
- Only built two tools instead of more, because I wanted them to do clearly different things — one returns the actual chunk content for the LLM to think about, the other returns just source/url/date for citations. Didn't want overlapping tools that do basically the same thing.
- Dashboard calls the agent live instead of reading from a cached file. Slower, but it means the dashboard always reflects whatever the retriever/prompt currently look like, which matters if parameters get changed.
- `max_iterations=10` on the AgentExecutor so it can't loop forever, but still has enough room to search more than once per question.

## Running it

```powershell
.venv\Scripts\Activate.ps1
python main.py
streamlit run src/dashboard/dashboard.py
```

Need Ollama running locally with qwen3:8b pulled before any of the agent/dashboard stuff works:

```powershell
ollama serve
ollama list
```

## Folder structure

```
nlpFinal/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunked/
│   └── embeddings/
├── src/
│   ├── collection/
│   ├── cleaning/
│   ├── chunking/
│   ├── embeddings/
│   ├── agent/
│   └── dashboard/
└── main.py
```