@echo off
REM ============================================
REM Hunter Drone - Tek Tikla Kurulum (Windows)
REM ============================================
REM Cift tiklayin, gerisi otomatik!

setlocal enabledelayedexpansion
color 0A

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo.
echo ============================================
echo    HUNTER DRONE - OTOMATIK KURULUM
echo ============================================
echo.
echo    Her sey otomatik yukleniyor...
echo    Lutfen bekleyin.
echo.

REM === PYTHON KONTROLU ===
echo [1/5] Python kontrol ediliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [HATA] Python bulunamadi!
    echo.
    echo Python 3.10+ yukleyin: https://www.python.org/downloads/
    echo Yuklerken "Add Python to PATH" secenegini isaretleyin!
    echo.
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo [HATA] Python 3.10 veya uzeri gerekli!
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do echo       Python %%i bulundu.

REM === VIRTUAL ENVIRONMENT ===
echo.
echo [2/5] Ortam hazirlaniyor...
if exist "venv" rmdir /s /q venv
python -m venv venv
if errorlevel 1 (
    echo [HATA] Virtual environment olusturulamadi!
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1

REM === BAGIMLILIKLAR ===
echo.
echo [3/5] Kutuphaneler yukleniyor (bu biraz zaman alabilir)...
pip install -e ".[dev,training]" >nul 2>&1
if errorlevel 1 (
    echo       Ilk deneme basarisiz, tekrar deneniyor...
    pip install -e .
)

REM === MODEL INDIRME ===
echo.
echo [4/5] YOLO11 modeli indiriliyor...
if not exist "models" mkdir models

python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" >nul 2>&1
if exist "yolo11m.pt" (
    move yolo11m.pt models\ >nul 2>&1
    echo       Model indirildi: models/yolo11m.pt
) else if exist "models\yolo11m.pt" (
    echo       Model zaten mevcut.
) else (
    echo       [UYARI] Model indirilemedi, GUI'den manuel indirebilirsiniz.
)

REM === KLASORLER ===
echo.
echo [5/5] Klasorler hazirlaniyor...
if not exist "database\images\train" mkdir "database\images\train"
if not exist "database\images\val" mkdir "database\images\val"
if not exist "database\labels\train" mkdir "database\labels\train"
if not exist "database\labels\val" mkdir "database\labels\val"
if not exist "output" mkdir "output"

REM === GPU KONTROLU ===
echo.
echo ============================================
python -c "import torch; cuda=torch.cuda.is_available(); print(f'       GPU (CUDA): {\"AKTIF\" if cuda else \"Pasif\"}')" 2>nul

echo.
echo ============================================
echo    KURULUM TAMAMLANDI!
echo ============================================
echo.
echo    Simdi GUI aciliyor...
echo    (Bir sonraki seferde launch_gui.bat kullanin)
echo.
echo ============================================
echo.

REM === GUI BASLAT ===
timeout /t 2 >nul
python hunter_gui.py

pause
