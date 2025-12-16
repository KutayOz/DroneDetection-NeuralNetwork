@echo off
REM ============================================
REM Hunter Drone GUI Launcher (Windows)
REM ============================================

cd /d "%~dp0"

REM Virtual environment kontrolu
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment aktif edildi.
) else (
    echo [UYARI] Virtual environment bulunamadi.
    echo Once setup.bat calistirin.
    pause
    exit /b 1
)

echo.
echo Hunter Drone GUI baslatiliyor...
echo.

python hunter_gui.py

if errorlevel 1 (
    echo.
    echo [HATA] GUI baslatilamadi!
    pause
)
