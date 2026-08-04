from dotenv import load_dotenv
import os

load_dotenv()

required_keys = [
    "GROQ_API_KEY",
    "NVIDIA_API_KEY_EMBEDDED",
    "chat_model",
    "embedding_model",
    "embedding_model_url",
]
missing = [k for k in required_keys if not os.getenv(k)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {missing}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY_EMBEDDED = os.getenv("NVIDIA_API_KEY_EMBEDDED")
chat_model = os.getenv("chat_model")
embedding_model = os.getenv("embedding_model")
embedding_model_url = os.getenv("embedding_model_url")


QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant_db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "SM-bot")

COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "20"))
