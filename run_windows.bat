@echo off
cd /d "%~dp0"
pythonw meteor_composer.py
if errorlevel 1 python meteor_composer.py
