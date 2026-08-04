from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

long_term_archive: list[Document] = []

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an episodic memory creator. Summarize the following conversation.\n\n"
        "Keep:\n"
        "- technical discussions\n"
        "- architecture / project decisions\n"
        "- user goals and preferences\n"
        "- mistakes made and conclusions\n"
        "- unresolved questions\n"
        "- any code that was generated, preserved exactly as written\n\n"
        "Remove:\n"
        "- greetings, thanks, filler, repetitive replies\n\n"
        "The summary must be understandable without reading the original conversation. "
        "Return ONLY the summary.",
    ),
    ("human", "{conversation}"),
])


def update_history(messages: list[tuple[str, str]]) -> None:
    """Append raw (user, ai) turns to the in-memory long-term archive log.

    This keeps the original, un-summarized exchanges around (e.g. for
    debugging or future re-summarization) separately from what actually
    gets embedded into the vector store.
    """
    for user_msg, ai_msg in messages:
        long_term_archive.append(Document(page_content=f"User: {user_msg}\nAI: {ai_msg}"))


def get_content() -> list[Document]:
    """Returns all compiled archival records."""
    return long_term_archive


def summarize_for_long_term(llm, messages: list[tuple[str, str]]) -> str:
    """Condenses a batch of raw turns into a single episodic-memory summary
    using the chat LLM, before it gets chunked/embedded.

    Storing summaries instead of raw chat text (greetings, filler, dead
    ends) keeps the vector store meaning-dense, which is what actually
    helps retrieval quality later.
    """
    conversation = "\n\n".join(f"User: {u}\nAI: {a}" for u, a in messages)
    chain = _SUMMARY_PROMPT | llm
    result = chain.invoke({"conversation": conversation})
    return getattr(result, "content", str(result))


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Splits documents into smaller chunks before embedding.

    Long summaries get split into multiple appropriately-sized chunks so a
    single embedding vector doesn't have to represent an entire summary.
    Short summaries that are already under chunk_size pass through as a
    single chunk.
    """
    return text_splitter.split_documents(docs)
