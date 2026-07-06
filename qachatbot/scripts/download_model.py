# scripts/download_model.py
#
# Downloads the quantized local LLM used for generation. Run once before
# starting the API, or during the Docker image build.

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(MODELS_DIR))
    print(f"Model ready at {path}")


if __name__ == "__main__":
    main()
