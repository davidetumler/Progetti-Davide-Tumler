@echo off
cd /d "%~dp0"
if not exist .venv (py -3.11 -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
start "Backend FastAPI" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
