# rag/retriever.py

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "kb_index.faiss"
CHUNKS_PATH = BASE_DIR / "data" / "kb_chunks.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Knowledge base index not found. Run `python ingest.py` first."
            )
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, k: int = 4) -> list[dict]:
        query_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(np.array(query_vec), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "distance": float(dist)})
        return results
