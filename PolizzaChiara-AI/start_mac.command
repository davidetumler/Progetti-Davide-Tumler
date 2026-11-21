#!/bin/bash
cd "$(dirname "$0")"
if [ ! -f ".venv/bin/python" ]; then
  echo "Prima crea l'ambiente virtuale seguendo README.md"
  exit 1
fi
".venv/bin/python" -m streamlit run app.py
