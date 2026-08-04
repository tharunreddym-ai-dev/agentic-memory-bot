from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore

from agent.embeddings import NvidiaEmbeddings
from agent.memory import chunk_documents, summarize_for_long_term, update_history
from agent.prompt import REACT_CHAT_PROMPT
from agent.tools import make_long_term_memory_tool, web_search_tool
from config import (
    COMPACTION_THRESHOLD,
    GROQ_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    chat_model,
)


class ChatSession:
    """One conversational agent session.

    Short-term memory: the last COMPACTION_THRESHOLD (user, ai) turns are
    kept in `recent_messages` and injected into the prompt's chat_history
    on every call -- the agent always sees these for free.

    Long-term memory: once `recent_messages` hits the threshold, the whole
    batch is summarized by the LLM and embedded into a local on-disk Qdrant
    store. The agent can only reach that memory by deciding to call the
    `search_long_term_memory` tool -- it is never force-fed into the prompt.
    """

    def __init__(self):
        self.llm = ChatGroq(model=chat_model, api_key=GROQ_API_KEY)
        self.memory_store = {"vectorstore": None, "chunks": []}

        tools = [web_search_tool, make_long_term_memory_tool(self.memory_store)]

        agent = create_react_agent(llm=self.llm, tools=tools, prompt=REACT_CHAT_PROMPT)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_execution_time=60,
            max_iterations=6,
            early_stopping_method="generate",
            handle_parsing_errors=True,
        )

        self.recent_messages: list[tuple[str, str]] = []
        self.last_compaction_note: str | None = None

    def send(self, user_message: str) -> str:
        """Send one user message through the agent and return its reply."""
        if not user_message or not user_message.strip():
            return ""

        try:
            result = self.executor.invoke(
                {
                    "input": user_message,
                    "chat_history": self._format_recent(),
                }
            )
            output = result["output"]
        except Exception as e:
            output = f"[Error] Agent failed to produce a response: {e}"

        self.recent_messages.append((user_message, output))
        self._maybe_compact()
        return output

    def _format_recent(self) -> str:
        if not self.recent_messages:
            return "(no earlier messages yet)"
        return "\n\n".join(f"User: {u}\nAI: {a}" for u, a in self.recent_messages)

    def _maybe_compact(self) -> None:
        """Summarize + archive the oldest batch once the short-term window
        fills up. Failures here are non-fatal -- the chat keeps working,
        it just temporarily loses this batch's long-term memory."""
        if len(self.recent_messages) < COMPACTION_THRESHOLD:
            return

        batch = self.recent_messages
        self.recent_messages = []

        try:
            update_history(batch)

            summary = summarize_for_long_term(self.llm, batch)
            new_chunks = chunk_documents([Document(page_content=summary)])
            self.memory_store["chunks"].extend(new_chunks)

            if self.memory_store["vectorstore"] is None:
                self.memory_store["vectorstore"] = QdrantVectorStore.from_documents(
                    documents=self.memory_store["chunks"],
                    embedding=NvidiaEmbeddings(),
                    collection_name=QDRANT_COLLECTION,
                    path=QDRANT_PATH,
                )
            else:
                self.memory_store["vectorstore"].add_documents(new_chunks)

            self.last_compaction_note = f"Archived {len(batch)} messages into long-term memory."
        except Exception as e:
            self.last_compaction_note = f"Memory compaction failed, continuing without update: {e}"
