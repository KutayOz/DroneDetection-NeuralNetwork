@echo off
REM ============================================
REM Hunter Drone - Windows Kurulum Script
REM ============================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo    Hunter Drone - Otomatik Kurulum
echo ============================================
echo.

REM Renk ayarı
color 0A

REM Mevcut dizini kaydet
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/6] Python kontrolu yapiliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi!
    echo Python 3.10+ yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Python %PYTHON_VERSION% bulundu.

REM Python versiyon kontrolü
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo [HATA] Python 3.10 veya uzeri gerekli!
    pause
    exit /b 1
)

echo.
echo [2/6] Virtual environment olusturuluyor...
if exist "venv" (
    echo       Mevcut venv bulundu, siliniyor...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo [HATA] Virtual environment olusturulamadi!
    pause
    exit /b 1
)
echo       venv olusturuldu.

echo.
echo [3/6] Virtual environment aktif ediliyor...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [HATA] Virtual environment aktif edilemedi!
    pause
    exit /b 1
)
echo       venv aktif edildi.

echo.
echo [4/6] pip guncelleniyor...
python -m pip install --upgrade pip >nul 2>&1
echo       pip guncellendi.

echo.
echo [5/6] Bagimliliklari yukle...
echo.
echo    Hangi kurulum turunu istiyorsunuz?
echo    [1] Temel kurulum (sadece inference)
echo    [2] Gelistirici kurulumu (test + lint)
echo    [3] Egitim kurulumu (tensorboard + mlflow)
echo    [4] Tam kurulum (tum bagimlilklar)
echo.
set /p INSTALL_TYPE="Seciminiz (1-4): "

if "%INSTALL_TYPE%"=="1" (
    echo.
    echo       Temel kurulum yapiliyor...
    pip install -e .
) else if "%INSTALL_TYPE%"=="2" (
    echo.
    echo       Gelistirici kurulumu yapiliyor...
    pip install -e ".[dev]"
) else if "%INSTALL_TYPE%"=="3" (
    echo.
    echo       Egitim kurulumu yapiliyor...
    pip install -e ".[training]"
) else if "%INSTALL_TYPE%"=="4" (
    echo.
    echo       Tam kurulum yapiliyor...
    pip install -e ".[dev,training]"
) else (
    echo.
    echo       Varsayilan: Temel kurulum yapiliyor...
    pip install -e .
)

if errorlevel 1 (
    echo [HATA] Bagimliliklari yuklerken hata olustu!
    pause
    exit /b 1
)
echo       Bagimliliklari yuklendi.

echo.
echo [6/6] Kurulum dogrulaniyor...
python -c "from hunter import Pipeline, HunterConfig; print('       Import testi: BASARILI')" 2>nul
if errorlevel 1 (
    echo [UYARI] Import testi basarisiz, ancak kurulum tamamlandi.
)

REM GPU kontrolü
python -c "import torch; cuda=torch.cuda.is_available(); print(f'       CUDA: {\"Aktif\" if cuda else \"Pasif\"}')" 2>nul

echo.
echo ============================================
echo    KURULUM TAMAMLANDI!
echo ============================================
echo.
echo Sonraki adimlar:
echo   1. Database klasorune dataset'inizi kopyalayin
echo   2. Models klasorune YOLO11 agirliklarini indirin
echo   3. python scripts/run_inference.py --help
echo.
echo Virtual environment'i aktif etmek icin:
echo   venv\Scripts\activate
echo.
echo ============================================

pause
