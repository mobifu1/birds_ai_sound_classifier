@echo off
cd /d "%~dp0"
echo Starte Birds AI Sound Classifier...
python birds_ai_sound_classifier.py
if %errorlevel% neq 0 pause
exit /b 0
