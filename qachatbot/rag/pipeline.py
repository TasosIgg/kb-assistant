# rag/pipeline.py

import time

from rag.retriever import Retriever
from rag.generator import Generator

# Distance threshold above which retrieved context is considered irrelevant
# (IndexFlatL2 on normalized MiniLM embeddings; tuned empirically on the KB).
MAX_RELEVANT_DISTANCE = 1.2


class RagPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def ask(self, question: str, history: list[dict] | None = None, k: int = 4) -> dict:
        start = time.perf_counter()

        results = self.retriever.search(question, k=k)
        relevant = [r for r in results if r["distance"] <= MAX_RELEVANT_DISTANCE]

        if not relevant:
            answer = "I don't have that information in the knowledge base."
            sources = []
        else:
            context_blocks = [r["text"] for r in relevant]
            answer = self.generator.generate(question, context_blocks, history=history)
            seen = set()
            sources = []
            for r in relevant:
                if r["url"] not in seen:
                    seen.add(r["url"])
                    sources.append({"title": r["title"], "url": r["url"]})

        latency_ms = int((time.perf_counter() - start) * 1000)
        return {"answer": answer, "sources": sources, "latency_ms": latency_ms}
