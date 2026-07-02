# rag/generator.py

from pathlib import Path

from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "You are an internal Employee Self-Service Assistant for a company's knowledge base. "
    "Answer the user's question using ONLY the information in the provided context. "
    "If the context does not contain the answer, say clearly that you don't have that "
    "information in the knowledge base instead of guessing. Be concise and cite which "
    "section(s) you used."
)


class Generator:
    def __init__(self, n_ctx: int = 4096):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run scripts/download_model.py first."
            )
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=n_ctx,
            n_threads=max(1, __import__("os").cpu_count() - 1),
            verbose=False,
        )

    def generate(self, question: str, context_blocks: list[str], history: list[dict] | None = None) -> str:
        context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or [])[-4:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        })

        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )
        return result["choices"][0]["message"]["content"].strip()
