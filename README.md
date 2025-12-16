# Hunter Drone - Drone Tespit Sistemi

YOLO11 + Siamese Network tabanli gercek zamanli drone tespit ve takip sistemi.

## Kurulum

**Windows:** `setup.bat` dosyasina cift tiklayin.

**Mac/Linux:** Terminal'de `./setup.sh` calistirin.

Kurulum tamamlaninca GUI otomatik acilir. Sonraki kullanimlar icin `launch_gui.bat` veya `./launch_gui.sh` kullanin.

## Ne Yapar?

- Video veya canli kamera goruntulerinde drone tespit eder
- Tespit edilen dronelari takip eder (ID atama, hareket tahmini)
- Sonuclari JSON formatinda kaydeder

## Sistem Gereksinimleri

- Python 3.10+
- GPU (NVIDIA CUDA veya Apple Silicon) onerilen, CPU'da da calisir

## Pipeline

```
Goruntu → YOLO11 Tespit → Siamese Dogrulama → Kalman Takip → Sonuc
```

---

Detayli teknik dokumantasyon icin `docs/` klasorune bakin.
