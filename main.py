"""Terminal chat loop for local testing.

For the Streamlit UI, run `streamlit run app.py` instead -- both share the
same ChatSession from core.py.
"""

from core import ChatSession


def run_cli() -> None:
    session = ChatSession()
    print("SM Agent (CLI) -- type /q to exit\n")

    while True:
        user_input = input("You (/q for exit): ")
        if user_input == "/q":
            print("Chat Exited")
            break
        if not user_input.strip():
            continue

        output = session.send(user_input)
        print(f"AI: {output}\n")

        if session.last_compaction_note:
            print(f"[memory] {session.last_compaction_note}\n")
            session.last_compaction_note = None


if __name__ == "__main__":
    run_cli()
