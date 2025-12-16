#!/bin/bash
# ============================================
# Hunter Drone GUI Launcher (Mac/Linux)
# ============================================

# Proje dizinine git
cd "$(dirname "${BASH_SOURCE[0]}")"

# Virtual environment kontrolu
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment aktif edildi."
else
    echo "[UYARI] Virtual environment bulunamadi."
    echo "Once ./setup.sh calistirin."
    exit 1
fi

echo ""
echo "Hunter Drone GUI baslatiliyor..."
echo ""

python hunter_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[HATA] GUI baslatilamadi!"
    read -p "Devam etmek icin Enter'a basin..."
fi
