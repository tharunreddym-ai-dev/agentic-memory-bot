from langchain.tools import tool
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()


@tool
def web_search_tool(query: str) -> str:
    """This Tool is used to Search Information From the Web
    Use This When you need
     -Current Events
     -Recent Information
     -General Information from Web
     -Live Information
     -And Anything which is recent and which is outdated from your knowledge"""

    try:
        if not query or not query.strip():
            raise Exception("Query Cannot Empty")
        return search.run(query)
    except Exception as e:
        return f"TOOL_ERROR---{e}"


def build_retriever(vectorstore, chunks) -> EnsembleRetriever:
    """Builds (or rebuilds) the hybrid semantic + keyword retriever from the
    current vectorstore and chunk set.

    NOT an agent tool. A ReAct agent's LLM can only ever produce a *string*
    as tool input -- it has no way to pass a live vectorstore/chunk-list
    object into a tool call. This is a plain internal helper used by
    `search_long_term_memory` below, which is the actual thing the agent
    calls.
    """
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        c=60,
    )


def make_long_term_memory_tool(memory_store: dict):
    """Factory that returns a `search_long_term_memory` tool bound to
    `memory_store`, a dict holding the CURRENT `"vectorstore"` and
    `"chunks"` for this chat session.

    A factory (instead of one module-level @tool) is used because the
    vectorstore doesn't exist until the first memory compaction happens,
    and it keeps growing after that. The closure reads `memory_store` fresh
    on every call, so the tool always sees the latest archived memory
    instead of a snapshot frozen at import time.
    """

    @tool
    def search_long_term_memory(query: str) -> str:
        """Search the user's earlier conversation history that has already
        been summarized and archived into long-term memory.

        Use this tool ONLY when:
        - the user refers to a previous conversation or session
        - the user says something like "remember when..." or "what did I say about..."
        - the user asks about an earlier project, decision, or preference
        - the answer is clearly not present in the recent conversation you were given

        Do NOT use this tool for:
        - general knowledge questions
        - anything already answerable from the recent conversation
        - current events or live information (use the web search tool for that instead)

        Input should be a short, specific natural-language query describing
        what you're looking for (e.g. "what vector database the user chose"),
        not the user's raw message verbatim.
        """
        vectorstore = memory_store.get("vectorstore")
        chunks = memory_store.get("chunks")

        if vectorstore is None or not chunks:
            return "No long-term memory has been archived yet for this conversation."

        try:
            retriever = build_retriever(vectorstore, chunks)
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant long-term memory found for that query."
            return "\n\n---\n\n".join(d.page_content for d in docs)
        except Exception as e:
            return f"TOOL_ERROR---{e}"

    return search_long_term_memory
