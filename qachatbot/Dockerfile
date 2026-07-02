FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

# Bake the quantized model into the image so the container is self-contained
# (needed for a single-container deploy target like Hugging Face Spaces).
RUN python scripts/download_model.py

EXPOSE 8000 8501 7860

CMD ["./start.sh"]
