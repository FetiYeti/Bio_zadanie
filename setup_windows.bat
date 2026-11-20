@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Instalacia detektora voice changera ===
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg nie je nainstalovany - instalujem cez winget...
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  echo.
  echo !!! ffmpeg bol nainstalovany. ZATVORTE toto okno a spustite setup_windows.bat ZNOVA. !!!
  pause
  exit /b
)
if not exist .venv (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo === Kontrola ===
ffmpeg -hide_banner -filters 2>nul | findstr /i "rubberband"
python -c "import joblib; m=joblib.load('models/detector.joblib'); print('Model OK:', m['detector_name'])"
echo.
echo Hotovo. Ukazku spustite cez spusti_demo.bat
pause
