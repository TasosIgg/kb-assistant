# rag/pipeline.py

import re
import threading
import time

from rag.retriever import Retriever
from rag.generator import Generator
from rag.logging_config import setup_logging, log_request

# Distance threshold above which retrieved context is considered irrelevant
# (IndexFlatL2 on normalized MiniLM embeddings; tuned empirically on the KB).
MAX_RELEVANT_DISTANCE = 1.2

# Best-effort heuristic for obvious prompt-injection attempts. Not exhaustive —
# defense-in-depth alongside the system prompt's instruction-hierarchy language,
# not a complete solution.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |the )?(previous|prior|above) instructions",
        r"disregard (the |your )?system prompt",
        r"you are now",
        r"forget (all |the )?(previous|prior) (instructions|rules)",
        r"reveal (your |the )?system prompt",
    ]
]

logger = setup_logging()


def looks_like_injection(question: str) -> bool:
    return any(p.search(question) for p in INJECTION_PATTERNS)


class RagPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        # llama-cpp-python's Llama object isn't safe for concurrent generate calls;
        # this serializes access so concurrent requests queue correctly rather than
        # racing against the same model instance. Gives correctness, not throughput —
        # true parallel serving would need multiple model replicas or a GPU-backed
        # inference server, which is out of scope for this free CPU-only deploy.
        self._generation_lock = threading.Lock()

    def ask(self, question: str, history: list[dict] | None = None, k: int = 3) -> dict:
        start = time.perf_counter()

        if looks_like_injection(question):
            answer = (
                "I can't follow instructions embedded in a question — I can only "
                "answer questions about the knowledge base itself."
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(logger, question=question, answered=False,
                        latency_ms=latency_ms, top_source=None, flagged=True)
            return {"answer": answer, "sources": [], "latency_ms": latency_ms}

        results = self.retriever.search(question, k=k)
        relevant = [r for r in results if r["distance"] <= MAX_RELEVANT_DISTANCE]

        if not relevant:
            answer = "I don't have that information in the knowledge base."
            sources = []
        else:
            context_blocks = [r["text"] for r in relevant]
            with self._generation_lock:
                answer = self.generator.generate(question, context_blocks, history=history)
            seen = set()
            sources = []
            for r in relevant:
                if r["url"] not in seen:
                    seen.add(r["url"])
                    sources.append({"title": r["title"], "url": r["url"]})

        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(logger, question=question, answered=bool(relevant),
                    latency_ms=latency_ms, top_source=sources[0]["url"] if sources else None)
        return {"answer": answer, "sources": sources, "latency_ms": latency_ms}
