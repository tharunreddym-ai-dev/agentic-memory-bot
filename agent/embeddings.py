import time
import requests
from langchain_core.embeddings import Embeddings
from config import embedding_model_url, NVIDIA_API_KEY_EMBEDDED

emb_api = NVIDIA_API_KEY_EMBEDDED
emb_url = embedding_model_url


def get_embedding(texts: list, input_type: str, max_tries: int = 3) -> list:
    headers = {
        "Authorization": f"Bearer {emb_api}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "nvidia/nv-embedqa-e5-v5",
        "input": texts,
        "encoding_format": "float",
        "input_type": input_type,
    }

    for attempt in range(max_tries):
        try:
            response = requests.post(url=emb_url, headers=headers, json=body, timeout=30)
            if response.status_code != 200:
                raise Exception(f"API Error {response.text} with code {response.status_code}")
            data = response.json()
            if "data" not in data:
                raise Exception(f"Unexpected response {data}")
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            if attempt == max_tries - 1:
                raise e
            wait = 2 ** attempt
            print(f"Attempt {attempt + 1} failed, retrying in {wait}s... ({e})")
            # NOTE: the original version computed `wait` but never actually
            # slept, so retries fired back-to-back with no backoff.
            time.sleep(wait)


class NvidiaEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return get_embedding(texts, input_type="passage")

    def embed_query(self, text):
        return get_embedding(texts=[text], input_type="query")[0]
