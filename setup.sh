#!/bin/bash
# ============================================
# Hunter Drone - Mac/Linux Kurulum Script
# ============================================

set -e

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Hunter Drone - Otomatik Kurulum${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Proje dizinine git
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Python komutunu belirle
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[HATA] Python bulunamadi!${NC}"
    echo "Python 3.10+ yukleyin: https://www.python.org/downloads/"
    exit 1
fi

echo -e "${GREEN}[1/6]${NC} Python kontrolu yapiliyor..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "       Python $PYTHON_VERSION bulundu."

# Python versiyon kontrolü
$PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[HATA] Python 3.10 veya uzeri gerekli!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}[2/6]${NC} Virtual environment olusturuluyor..."
if [ -d "venv" ]; then
    echo "       Mevcut venv bulundu, siliniyor..."
    rm -rf venv
fi
$PYTHON_CMD -m venv venv
echo "       venv olusturuldu."

echo ""
echo -e "${GREEN}[3/6]${NC} Virtual environment aktif ediliyor..."
source venv/bin/activate
echo "       venv aktif edildi."

echo ""
echo -e "${GREEN}[4/6]${NC} pip guncelleniyor..."
pip install --upgrade pip > /dev/null 2>&1
echo "       pip guncellendi."

echo ""
echo -e "${GREEN}[5/6]${NC} Bagimliliklar yukleniyor..."
echo ""
echo "   Hangi kurulum turunu istiyorsunuz?"
echo "   [1] Temel kurulum (sadece inference)"
echo "   [2] Gelistirici kurulumu (test + lint)"
echo "   [3] Egitim kurulumu (tensorboard + mlflow)"
echo "   [4] Tam kurulum (tum bagimlilklar)"
echo ""
read -p "Seciminiz (1-4): " INSTALL_TYPE

case $INSTALL_TYPE in
    1)
        echo ""
        echo "       Temel kurulum yapiliyor..."
        pip install -e .
        ;;
    2)
        echo ""
        echo "       Gelistirici kurulumu yapiliyor..."
        pip install -e ".[dev]"
        ;;
    3)
        echo ""
        echo "       Egitim kurulumu yapiliyor..."
        pip install -e ".[training]"
        ;;
    4)
        echo ""
        echo "       Tam kurulum yapiliyor..."
        pip install -e ".[dev,training]"
        ;;
    *)
        echo ""
        echo "       Varsayilan: Temel kurulum yapiliyor..."
        pip install -e .
        ;;
esac

echo "       Bagimliliklari yuklendi."

echo ""
echo -e "${GREEN}[6/6]${NC} Kurulum dogrulaniyor..."
python -c "from hunter import Pipeline, HunterConfig; print('       Import testi: BASARILI')" 2>/dev/null || echo -e "${YELLOW}       [UYARI] Import testi basarisiz${NC}"

# GPU kontrolü (macOS için MPS, Linux için CUDA)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - MPS kontrolü
    python -c "import torch; mps=torch.backends.mps.is_available(); print(f'       MPS (Apple Silicon): {\"Aktif\" if mps else \"Pasif\"}')" 2>/dev/null || true
else
    # Linux - CUDA kontrolü
    python -c "import torch; cuda=torch.cuda.is_available(); print(f'       CUDA: {\"Aktif\" if cuda else \"Pasif\"}')" 2>/dev/null || true
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}   KURULUM TAMAMLANDI!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Sonraki adimlar:"
echo "  1. Database klasorune dataset'inizi kopyalayin"
echo "  2. Models klasorune YOLO11 agirliklarini indirin"
echo "  3. python scripts/run_inference.py --help"
echo ""
echo "Virtual environment'i aktif etmek icin:"
echo "  source venv/bin/activate"
echo ""
echo -e "${BLUE}============================================${NC}"
