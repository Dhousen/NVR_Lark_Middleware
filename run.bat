@echo off
REM Shortcut untuk menjalankan middleware di Windows.
REM Pastikan sudah menjalankan setup (lihat README.md) sebelum pakai ini:
REM   1. python -m venv venv
REM   2. venv\Scripts\activate
REM   3. pip install -r requirements.txt
REM   4. copy .env.example .env  (lalu isi nilainya)
REM   5. python tools\download_reference_photos.py

if not exist venv\Scripts\python.exe (
    echo [ERROR] Virtual environment belum dibuat.
    echo Jalankan dulu: python -m venv venv
    echo Lalu: venv\Scripts\activate  dan  pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist .env (
    echo [ERROR] File .env belum ada.
    echo Salin .env.example menjadi .env lalu isi nilainya terlebih dahulu.
    pause
    exit /b 1
)

venv\Scripts\python.exe main.py
pause
