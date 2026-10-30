#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Per usare OPENAI_API_KEY, puoi esportarla nel terminale prima di avviare:
# export OPENAI_API_KEY="..."
python -m uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
sleep 3
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
