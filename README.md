# SM Agent

A conversational AI agent (ReAct architecture, LangChain) with two kinds of
memory — short-term memory that's always visible to the model, and
long-term memory that the agent has to *decide* to go look up — plus live
web search. Ships with a Streamlit chat UI.

Originally prototyped as an n8n workflow (Telegram trigger → Postgres
short-term memory → Pinecone long-term memory), then rebuilt as a
standalone Python project.

## How the memory works

```
User message
     │
     ▼
 ReAct Agent (Groq LLM) ── decides per-turn whether it needs a tool ──┐
     │                                                                 │
     │                                          ┌──────────────────┐  │
     │                                          │ web_search_tool  │◄─┤
     │                                          └──────────────────┘  │
     │                                          ┌──────────────────────────┐
     │                                          │ search_long_term_memory  │◄─┘
     │                                          └──────────────────────────┘
     ▼
  Output
```

- **Short-term memory** — the last `COMPACTION_THRESHOLD` (user, AI) turns
  are kept in memory and injected straight into the prompt's
  `chat_history` on every call. The agent always has these for free, no
  tool call needed.

- **Long-term memory** — once the short-term window fills up, that whole
  batch is:
  1. summarized by the LLM into a dense "episodic memory" (decisions,
     preferences, code, open questions — greetings and filler dropped),
  2. chunked,
  3. embedded (NVIDIA embeddings) and stored in a local, on-disk **Qdrant**
     collection.

  The agent can *only* reach this by explicitly calling the
  `search_long_term_memory` tool — it's never force-injected into the
  prompt. The system prompt tells the model to reach for it when the user
  refers to an earlier session, says "remember...", or asks about a past
  decision/project — and to leave it alone for general knowledge or
  anything already in the recent conversation.

- **Web search** — a separate `web_search_tool` (DuckDuckGo) for current
  events / live information, picked independently by the same
  agent-decides-per-turn logic.

Retrieval over long-term memory uses a hybrid retriever: Qdrant semantic
search (MMR) combined with BM25 keyword search via LangChain's
`EnsembleRetriever`.

## Project structure

```
.
├── app.py              # Streamlit UI (chat + memory-status sidebar)
├── main.py             # Terminal chat loop, for local testing
├── core.py             # ChatSession — agent/executor wiring, shared by app.py & main.py
├── config.py            # Env var loading + validation
├── agent/
│   ├── prompt.py        # Local ReAct prompt (system rules + chat_history)
│   ├── tools.py          # web_search_tool, search_long_term_memory
│   ├── memory.py         # Archiving, LLM summarization, chunking
│   └── embeddings.py     # NVIDIA embeddings wrapper
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in your keys
```

Required in `.env`:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (chat LLM) |
| `NVIDIA_API_KEY_EMBEDDED` | NVIDIA NIM API key (embeddings) |
| `chat_model` | Groq model name, e.g. `llama-3.3-70b-versatile` |
| `embedding_model` | Embedding model name |
| `embedding_model_url` | NVIDIA embedding endpoint URL |

Optional (all have working defaults — see `.env.example`):
`QDRANT_PATH`, `QDRANT_COLLECTION`, `COMPACTION_THRESHOLD`.

## Run it

**Streamlit UI:**
```bash
streamlit run app.py
```

**Terminal:**
```bash
python main.py
```

## Notes / limitations

- Built for a **single local user** — session state (short-term memory,
  the agent executor) lives in one Streamlit session, not per-visitor.
- Qdrant runs in **local on-disk mode** (`QDRANT_PATH`, gitignored under
  `data/`) — no server to run, but only one process can hold the DB open
  at a time (don't run `app.py` and `main.py` against the same path
  simultaneously).
- Memory compaction happens automatically once the short-term window
  fills up (`COMPACTION_THRESHOLD` turns); the sidebar surfaces a status
  note when it runs.
