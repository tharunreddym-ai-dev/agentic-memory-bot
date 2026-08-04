import streamlit as st

from config import COMPACTION_THRESHOLD, chat_model
from core import ChatSession

st.set_page_config(page_title="SM Agent", page_icon="🧠", layout="wide")

if "session" not in st.session_state:
    st.session_state.session = ChatSession()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

session: ChatSession = st.session_state.session

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("🧠 Memory")
    st.caption(f"Model: `{chat_model}`")

    st.subheader("Short-term")
    recent_count = len(session.recent_messages)
    st.write(f"{recent_count} / {COMPACTION_THRESHOLD} turns in the active window")
    st.progress(min(recent_count / COMPACTION_THRESHOLD, 1.0))

    st.subheader("Long-term")
    has_long_term = session.memory_store["vectorstore"] is not None
    if has_long_term:
        st.write(f"Archived — {len(session.memory_store['chunks'])} chunk(s) stored")
        st.caption("Reachable only via the search_long_term_memory tool.")
    else:
        st.write("Nothing archived yet")

    if session.last_compaction_note:
        st.info(session.last_compaction_note)

    st.divider()
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.session = ChatSession()
        st.session_state.chat_display = []
        st.rerun()

# ------------------------------------------------------------------ main --
st.title("SM Agent")
st.caption(
    "A ReAct agent with short-term memory (always in context) and "
    "tool-based long-term memory (recalled only when the agent decides it's needed), "
    "plus live web search."
)

for role, text in st.session_state.chat_display:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_display.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            output = session.send(user_input)
        st.markdown(output)

    st.session_state.chat_display.append(("assistant", output))
    st.rerun()
