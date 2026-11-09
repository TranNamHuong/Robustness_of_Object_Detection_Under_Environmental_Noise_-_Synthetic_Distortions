@echo off
set "PROJECT_DIR=%~dp0"
if not exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    echo Khong tim thay moi truong .venv trong thu muc du an.
    echo Cai dat dependencies theo README.md truoc khi mo ung dung.
    pause
    exit /b 1
)
cd /d "%PROJECT_DIR%"
"%PROJECT_DIR%.venv\Scripts\python.exe" "%PROJECT_DIR%app.py"
if errorlevel 1 pause
