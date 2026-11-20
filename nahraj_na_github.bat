@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Nahratie projektu na GitHub: FetiYeti/Bio_zadanie ===
where git >nul 2>nul
if errorlevel 1 (
  echo Git nie je nainstalovany - instalujem cez winget...
  winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
  echo.
  echo !!! Git bol nainstalovany. ZATVORTE toto okno a spustite nahraj_na_github.bat ZNOVA. !!!
  pause
  exit /b
)
if not exist .git git init -b main
git config user.name >nul 2>nul || git config user.name "Peter Sabol"
git config user.email >nul 2>nul || git config user.email "FetiYeti@users.noreply.github.com"
git add -A
git commit -m "Detekcia a odhalenie voice changera v reci" -m "Semestralne zadanie - Biometricke systemy bezpecnosti, TUKE FEI 2026/27." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_016hbRUhufnjH412FNQwF8Rw"
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin https://github.com/FetiYeti/Bio_zadanie.git
echo.
echo Ak sa otvori okno GitHubu, prihlaste sa (Sign in with your browser).
git push -u origin main
echo.
if errorlevel 1 (echo NAHRATIE ZLYHALO - poslite screenshot tohto okna.) else (echo Hotovo: https://github.com/FetiYeti/Bio_zadanie)
pause
