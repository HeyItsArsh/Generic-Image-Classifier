@echo off
setlocal
cd /d "%~dp0"
title Index - Image Desk
cls
echo  INDEX / IMAGE DESK
echo  -----------------
echo  Starting your saved classifier. Keep this window open.
echo  Your browser will open automatically when the model is ready.
echo.
if exist "%~dp0venv\Scripts\python.exe" (
  "%~dp0venv\Scripts\python.exe" "%~dp0app.py" --open
) else (
  py -3 "%~dp0app.py" --open
)
echo.
echo  The image desk has stopped. Close this window or press a key to dismiss it.
pause >nul
