#!/bin/sh
# Runs both services in one container, for single-port deploy targets
# (e.g. Hugging Face Spaces) where only one exposed port is available.
set -e

uvicorn api:app --host 0.0.0.0 --port 8000 &

exec streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-7860}" --server.headless true
