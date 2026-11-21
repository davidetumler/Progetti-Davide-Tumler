@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Devi creare prima l'ambiente virtuale seguendo README.md
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
