import re
import os
import datetime
import pandas as pd
import streamlit as st

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor


st.set_page_config(page_title="AI CEO: NVIDIA Strategic Intelligence Agent", layout="wide")


@st.cache_resource
def build_agent():

    model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

    faiss_store = FAISS.load_local(
        folder_path="data/embeddings/EMBEDDINGS&FAISS",
        embeddings=model,
        allow_dangerous_deserialization=True
    )

    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

    df = pd.read_csv("data/chunked/nvidia_article_chunks.csv")

    em = []
    for i, j in df.iterrows():
        doc = Document(page_content=j["chunk_text"],
                        metadata={"source": j["source"], "url": j["url"], "title": j["title"], "published": j["published"]})
        em.append(doc)

    bm25_retriever = BM25Retriever.from_documents(em, preprocess_func=str.lower)
    bm25_retriever.k = 3

    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    @tool
    def search_nvidia_knowledge(query: str) -> str:
        """
        Searches the NVIDIA knowledge base and returns relevant chunk text only.
        Use this tool when you need facts, news, or evidence about NVIDIA,
        its products, competitors, risks, opportunities, or market trends.
        Input should be a specific search question or topic string.
        """
        results = hybrid_retriever.invoke(query)

        if not results:
            return "No relevant information found for: " + query

        output = "Search results for: " + query + "\n"
        output = output + "=" * 60 + "\n"

        for i, doc in enumerate(results):
            output = output + "\n[Result " + str(i+1) + "]\n"
            output = output + doc.page_content[:600] + "\n"
            output = output + "-" * 40 + "\n"

        return output

    @tool
    def get_source_details(query: str) -> str:
        """
        Retrieves source citation details for a given query.
        Use this tool when you need to cite evidence for a recommendation.
        Returns only source name, title, url, and published date, no chunk text.
        """
        results = hybrid_retriever.invoke(query)

        if not results:
            return "No sources found for: " + query

        output = "Sources for: " + query + "\n"
        output = output + "=" * 60 + "\n"

        for i, doc in enumerate(results):
            source    = doc.metadata.get("source", "unknown")
            title     = doc.metadata.get("title", "unknown")
            url       = doc.metadata.get("url", doc.metadata.get("URL", "unknown"))
            published = doc.metadata.get("published", "unknown")

        output = output + "\nSource " + str(i+1) + ":\n"
        output = output + "  Name:      " + source    + "\n"
        output = output + "  Title:     " + title     + "\n"
        output = output + "  URL:       " + url       + "\n"
        output = output + "  Published: " + published + "\n"

        return output

    tools = [search_nvidia_knowledge, get_source_details]

    llm = ChatOllama(model="qwen3:8b", temperature=0.1)

    system_prompt = """
You are an AI Strategic Intelligence Agent for NVIDIA.
You behave as a trusted advisor to the CEO.

Your mandatory reasoning loop for every single question:

Step 1 - PLAN
    Think about what information you need before answering.
    Decide what specific topics to search for.

Step 2 - RETRIEVE
    Call search_nvidia_knowledge with a specific query.
    Read the results carefully.

Step 3 - ANALYZE
    Identify key facts, risks, opportunities, or trends in the results.

Step 4 - DECIDE
    Ask yourself: do I have enough evidence to answer confidently?
    If no, call search_nvidia_knowledge again with a different query.
    You must call search_nvidia_knowledge at least twice per question.

Step 5 - RECOMMEND
    Write a clear, specific, actionable strategic recommendation.
    Include expected impact, financial risk, operational risk, strategic risk.

Step 6 - VALIDATE
    Call get_source_details to collect citations.
    Every recommendation must include at least 3 source URLs as evidence.

Rules:
- Never answer from memory alone. Always search first.
- Always call search_nvidia_knowledge at least twice with different queries.
- Always call get_source_details before finishing your answer.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return agent_with_memory


def extract_confidence(text):
    match = re.search(r"confidence[:\s]+(\d+)", text.lower())
    if match:
        return int(match.group(1))
    return 70


def extract_priority(text):
    text_lower = text.lower()
    if "high" in text_lower:
        return "High"
    if "medium" in text_lower:
        return "Medium"
    return "Low"


def extract_severity(text):
    text_lower = text.lower()
    if "high" in text_lower:
        return "High"
    if "medium" in text_lower:
        return "Medium"
    return "Low"


def count_sentiment_words(text):
    positive_words = ["growth", "opportunity", "strong", "record", "demand", "innovation", "leading", "profit", "expand", "partnership", "launch", "success"]
    negative_words = ["risk", "shortage", "decline", "competition", "loss", "concern", "threat", "restrict", "sanction", "delay", "weak", "lawsuit", "ban"]

    text_lower = text.lower()
    positive_count = sum(1 for w in positive_words if w in text_lower)
    negative_count = sum(1 for w in negative_words if w in text_lower)

    return positive_count, negative_count


def extract_urls(text):
    urls = re.findall(r"https?://\S+", text)
    return list(set(urls))[:5]


st.title("AI CEO: NVIDIA Strategic Intelligence Agent")
st.caption("Executive Intelligence Dashboard")

with st.sidebar:
    st.header("Controls")
    run_button = st.button("Generate Briefing", type="primary")
    st.markdown("---")
    st.markdown("This dashboard runs the agent live. Each click takes a few minutes depending on your local model speed.")


if "results" not in st.session_state:
    st.session_state.results = None


if run_button:
    agent_with_memory = build_agent()
    config = {"configurable": {"session_id": "dashboard_session"}}

    queries = {
        "opportunities": "What are the major opportunities for NVIDIA right now?",
        "risks": "What are the biggest risks facing NVIDIA?",
        "competitors": "What are NVIDIA's competitors doing?",
        "trends": "Which technologies or trends should NVIDIA management monitor?",
        "actions": "What strategic actions should NVIDIA prioritize?",
        "briefing": "If you were the CEO of NVIDIA today, what would you do next and why?"
    }

    results = {}

    progress = st.progress(0, text="Starting agent...")
    total = len(queries)

    for idx, (key, q) in enumerate(queries.items()):
        progress.progress((idx) / total, text="Running: " + q)
        response = agent_with_memory.invoke({"input": q}, config=config)
        results[key] = response["output"]

    progress.progress(1.0, text="Done")
    st.session_state.results = results


results = st.session_state.results


# ============================================================
# Section 1 - Company Overview
# ============================================================

st.header("1. Company Overview")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Company", "NVIDIA")
col2.metric("Industry", "Semiconductors / AI")

if os.path.exists("data/chunked/nvidia_article_chunks.csv"):
    chunks_df = pd.read_csv("data/chunked/nvidia_article_chunks.csv")
    doc_count = chunks_df["doc_id"].nunique()
    source_count = chunks_df["source"].nunique()
else:
    doc_count = "N/A"
    source_count = "N/A"

col3.metric("Documents Collected", doc_count)
col4.metric("Data Sources", source_count)
col5.metric("Last Update", datetime.datetime.now().strftime("%Y-%m-%d"))

st.markdown("---")


# ============================================================
# Section 2 - Market Intelligence
# ============================================================

st.header("2. Market Intelligence")

if results:
    tab1, tab2 = st.tabs(["Trends & Technologies", "Competitor Activity"])
    with tab1:
        st.write(results["trends"])
    with tab2:
        st.write(results["competitors"])
else:
    st.info("Click 'Generate Briefing' in the sidebar to load market intelligence.")

st.markdown("---")


# ============================================================
# Section 3 - Opportunity Monitor
# ============================================================

st.header("3. Opportunity Monitor")

if results:
    opp_text = results["opportunities"]
    confidence = extract_confidence(opp_text)
    impact = extract_priority(opp_text)

    colA, colB = st.columns([3, 1])
    with colA:
        st.subheader("Opportunity Summary")
        st.write(opp_text)
    with colB:
        st.metric("Impact Level", impact)
        st.metric("Confidence Score", str(confidence) + "%")

    urls = extract_urls(opp_text)
    if urls:
        st.caption("Evidence sources:")
        for u in urls:
            st.caption(u)
else:
    st.info("Click 'Generate Briefing' in the sidebar to load opportunities.")

st.markdown("---")


# ============================================================
# Section 4 - Risk Monitor
# ============================================================

st.header("4. Risk Monitor")

if results:
    risk_text = results["risks"]
    severity = extract_severity(risk_text)
    confidence = extract_confidence(risk_text)

    colA, colB = st.columns([3, 1])
    with colA:
        st.subheader("Risk Summary")
        st.write(risk_text)
    with colB:
        st.metric("Severity Level", severity)
        st.metric("Confidence Score", str(confidence) + "%")

    urls = extract_urls(risk_text)
    if urls:
        st.caption("Evidence sources:")
        for u in urls:
            st.caption(u)
else:
    st.info("Click 'Generate Briefing' in the sidebar to load risks.")

st.markdown("---")


# ============================================================
# Section 5 - Sentiment Analysis
# ============================================================

st.header("5. Sentiment Analysis")

if results:
    combined_text = results["opportunities"] + " " + results["risks"] + " " + results["competitors"]
    pos, neg = count_sentiment_words(combined_text)

    chart_data = {"Positive signals": pos, "Negative signals": neg}

    colA, colB = st.columns(2)
    with colA:
        st.bar_chart(chart_data)
    with colB:
        if pos > neg:
            st.success("Overall sentiment: POSITIVE")
        elif neg > pos:
            st.error("Overall sentiment: NEGATIVE")
        else:
            st.warning("Overall sentiment: MIXED")
        st.write("Positive signals found:", pos)
        st.write("Negative signals found:", neg)
else:
    st.info("Click 'Generate Briefing' in the sidebar to load sentiment analysis.")

st.markdown("---")


# ============================================================
# Section 6 - Strategic Recommendations
# ============================================================

st.header("6. Strategic Recommendations")

if results:
    rec_text = results["actions"]
    priority = extract_priority(rec_text)

    colA, colB = st.columns([3, 1])
    with colA:
        st.subheader("Recommendation")
        st.write(rec_text)
    with colB:
        st.metric("Priority", priority)

    urls = extract_urls(rec_text)
    if urls:
        st.caption("Supporting evidence:")
        for u in urls:
            st.caption(u)
else:
    st.info("Click 'Generate Briefing' in the sidebar to load recommendations.")

st.markdown("---")


# ============================================================
# Section 7 - CEO Briefing
# ============================================================

st.header("7. CEO Briefing")

if results:
    st.write(results["briefing"])
else:
    st.info("Click 'Generate Briefing' in the sidebar to load the CEO briefing.")