#!/bin/bash
# ============================================
# Hunter Drone - Tek Tikla Kurulum (Mac/Linux)
# ============================================
# Terminal'de: ./setup.sh
# Her sey otomatik!

set -e

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   HUNTER DRONE - OTOMATIK KURULUM${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "   Her sey otomatik yukleniyor..."
echo "   Lutfen bekleyin."
echo ""

# === PYTHON KONTROLU ===
echo -e "${GREEN}[1/5]${NC} Python kontrol ediliyor..."

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo ""
    echo -e "${RED}[HATA] Python bulunamadi!${NC}"
    echo ""
    echo "Mac: brew install python@3.10"
    echo "Ubuntu: sudo apt install python3.10 python3.10-venv"
    echo ""
    exit 1
fi

$PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null || {
    echo -e "${RED}[HATA] Python 3.10 veya uzeri gerekli!${NC}"
    exit 1
}

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "       Python $PYTHON_VERSION bulundu."

# === VIRTUAL ENVIRONMENT ===
echo ""
echo -e "${GREEN}[2/5]${NC} Ortam hazirlaniyor..."
[ -d "venv" ] && rm -rf venv
$PYTHON_CMD -m venv venv
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1

# === BAGIMLILIKLAR ===
echo ""
echo -e "${GREEN}[3/5]${NC} Kutuphaneler yukleniyor (bu biraz zaman alabilir)..."
pip install -e ".[dev,training]" > /dev/null 2>&1 || pip install -e .

# === MODEL INDIRME ===
echo ""
echo -e "${GREEN}[4/5]${NC} YOLO11 modeli indiriliyor..."
mkdir -p models

if [ ! -f "models/yolo11m.pt" ]; then
    python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" > /dev/null 2>&1 || true
    [ -f "yolo11m.pt" ] && mv yolo11m.pt models/
fi

if [ -f "models/yolo11m.pt" ]; then
    echo "       Model hazir: models/yolo11m.pt"
else
    echo -e "${YELLOW}       [UYARI] Model indirilemedi, GUI'den indirebilirsiniz.${NC}"
fi

# === KLASORLER ===
echo ""
echo -e "${GREEN}[5/5]${NC} Klasorler hazirlaniyor..."
mkdir -p database/images/train database/images/val
mkdir -p database/labels/train database/labels/val
mkdir -p output

# === GPU KONTROLU ===
echo ""
echo -e "${BLUE}============================================${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    python -c "import torch; mps=torch.backends.mps.is_available(); print(f'       GPU (Apple Silicon): {\"AKTIF\" if mps else \"Pasif\"}')" 2>/dev/null || true
else
    python -c "import torch; cuda=torch.cuda.is_available(); print(f'       GPU (CUDA): {\"AKTIF\" if cuda else \"Pasif\"}')" 2>/dev/null || true
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}   KURULUM TAMAMLANDI!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "   Simdi GUI aciliyor..."
echo "   (Bir sonraki seferde ./launch_gui.sh kullanin)"
echo ""
echo -e "${BLUE}============================================${NC}"
echo ""

# === GUI BASLAT ===
sleep 2
python hunter_gui.py
