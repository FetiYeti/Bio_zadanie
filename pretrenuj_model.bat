@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate
REM Ak sa model nenacita (ina verzia scikit-learn), pretrenuje ho z ulozenych priznakov (~3 min)
python train_eval.py
pause
