import pandas as pd
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


def main():

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

    # weights control how much each retriever contributes, must add up to 1.0
    # first number is BM25, second is FAISS
    # docs: https://python.langchain.com/docs/how_to/ensemble_retriever/
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
            output = output + "\nSource " + str(i+1) + ":\n"
            output = output + "  Name:      " + doc.metadata["source"]    + "\n"
            output = output + "  Title:     " + doc.metadata["title"]     + "\n"
            output = output + "  URL:       " + doc.metadata["url"]       + "\n"
            output = output + "  Published: " + doc.metadata["published"] + "\n"

        return output

    tools = [search_nvidia_knowledge, get_source_details]

    # temperature controls randomness
    # lower temperature = more focused answers, higher = more creative
    # docs: https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama
    # I use langchain_community.chat_models, older path, same idea)
    # https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html
    llm = ChatOllama(model="qwen3:8b", temperature=0.1) #:-
    # ChatOllama — a class from LangChain that knows how to talk to a locally running Ollama server.
    # llm is just a variable which I can rename anytim i ant.


    # docs for ChatPromptTemplate + MessagesPlaceholder:
    # https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
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
# ChatPromptTemplate — a LangChain class that builds a structured conversation template, made of multiple message "slots."
# .from_messages([...]) — a method that accepts a list of message definitions and builds the template from them. Each item in the list is a tuple or a placeholder object.         
        ("system", system_prompt), #A tuple. 
        # First item "system" is the role label. 
        # Second item is the actual text 
        # our long instructions string. 
        # This becomes the system message.
        MessagesPlaceholder(variable_name="chat_history"), # MessagesPlaceholder 
        #a special LangChain object that reserves an empty 
        # slot in the prompt. It does not contain text itself 
        # it is a placeholder that gets filled in later, 
        # dynamically, when the prompt actually runs. 
        # variable_name="chat_history" means: when this 
        # prompt runs, look for a value called chat_history 
        # and insert all of its messages right here. 
        # This is how memory gets injected into the conversation.
        ("human", "{input}"), # Another tuple. 
        # Role is "human". 
        # The text "{input}" is a template variable 
        # curly braces mean "fill this in later." 
        # When the prompt actually runs, whatever 
        # string you pass as input gets substituted here.
        MessagesPlaceholder(variable_name="agent_scratchpad") #Another reserved slot. 
        # This one is filled automatically by LangChain during 
        # the agent's reasoning loop it holds the record of 
        # which tools were called and what they returned, 
        # so the LLM can see its own past actions within the same question.
    ])
    # AFTER THE AFOREMENTIONED STEPS:- 
    # prompt variable holds :-
    # the complete four-part template: 
    # system instructions, history slot, current question, scratchpad slot

    # docs: https://reference.langchain.com/python/langchain-classic/agents/tool_calling_agent/base/create_tool_calling_agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt) #create_tool_calling_agent — a LangChain function that wires three things together: the LLM, the list of tools, and the prompt template.

    # max_iterations caps how many tool calls the agent can make before stopping
    # raise this if the agent needs more reasoning steps, lower it to force faster answers
    # docs: https://callsphere.ai/blog/building-langchain-agents-tools-agentexecutor-react-loop
    # (covers max_iterations, max_execution_time, handle_parsing_errors)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
    # AgentExecutor 
    # a LangChain class that actually runs the agent in a loop. 
    # The agent object from the line above only knows how to 
    # make one decision; AgentExecutor is what repeats that 
    # decision-making process until the LLM is done.

    # docs for memory pattern (store dict + get_session_history + RunnableWithMessageHistory):
    # https://reference.langchain.com/python/langchain-core/runnables/history/RunnableWithMessageHistory
    # https://medium.com/@jkSurampudi5/message-history-in-langchain-8b6c10e89e60
    store = {} # History 

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

    config = {"configurable": {"session_id": "nvidia_ceo_session"}}

    queries = [
        "What are the major opportunities for NVIDIA right now?",
        "What are the biggest risks facing NVIDIA?",
        "What are NVIDIA's competitors doing?",
        "Which technologies or trends should NVIDIA management monitor?",
        "What strategic actions should NVIDIA prioritize?",
        "If you were the CEO of NVIDIA today, what would you do next and why?"
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print("QUERY: " + q)
        print("=" * 70)

        response = agent_with_memory.invoke({"input": q}, config=config)

        print("\n--- FINAL ANSWER ---")
        print(response["output"])
        print("\n")


if __name__ == "__main__":
    main()