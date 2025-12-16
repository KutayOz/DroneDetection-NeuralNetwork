# Hunter Drone - Starting Guide

Bu rehber, Hunter Drone Detection & Tracking sistemini sıfırdan kurup çalıştırmanız için adım adım talimatlar içerir.

---

## İçindekiler

1. [Gereksinimler](#1-gereksinimler)
2. [Kurulum](#2-kurulum)
3. [Dataset Hazırlama](#3-dataset-hazırlama)
4. [Model Eğitimi](#4-model-eğitimi)
5. [Inference Çalıştırma](#5-inference-çalıştırma)
6. [GUI Uygulaması](#6-gui-uygulaması)
7. [Konfigürasyon](#7-konfigürasyon)
8. [Python API Kullanımı](#8-python-api-kullanımı)
9. [Sorun Giderme](#9-sorun-giderme)

---

## 1. Gereksinimler

### Donanım
- **GPU**: NVIDIA GPU (CUDA destekli) - önerilir
- **RAM**: Minimum 8GB, önerilen 16GB+
- **Depolama**: Dataset boyutuna bağlı, minimum 10GB boş alan

### Yazılım
- Python 3.10 veya üzeri
- CUDA 11.8+ (GPU kullanımı için)
- cuDNN 8.6+ (GPU kullanımı için)

### GPU Kontrolü
```bash
# NVIDIA GPU kontrolü
nvidia-smi

# CUDA versiyonu
nvcc --version
```

---

## 2. Kurulum

### Otomatik Kurulum (Önerilen)

Kurulum işlemini otomatik olarak yapmak için hazır script'leri kullanabilirsiniz:

**Windows:**
```cmd
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh  # İlk seferde çalıştırma izni ver
./setup.sh
```

Script size kurulum türünü soracak ve tüm adımları otomatik olarak gerçekleştirecektir.

---

### Manuel Kurulum

Aşağıdaki adımları takip ederek manuel kurulum da yapabilirsiniz:

### Adım 2.1: Proje Dizinine Git
```bash
cd hunter_drone
```

### Adım 2.2: Virtual Environment Oluştur
```bash
# Virtual environment oluştur
python -m venv venv

# Aktif et (Linux/Mac)
source venv/bin/activate

# Aktif et (Windows)
venv\Scripts\activate
```

### Adım 2.3: Paketi Yükle
```bash
# Temel kurulum
pip install -e .

# Geliştirme araçları ile (test, lint)
pip install -e ".[dev]"

# Eğitim araçları ile (tensorboard, mlflow)
pip install -e ".[training]"

# Tüm bağımlılıklar
pip install -e ".[dev,training]"
```

### Adım 2.4: Kurulumu Doğrula
```bash
# Python import testi
python -c "from hunter import Pipeline, HunterConfig; print('Kurulum başarılı!')"

# CUDA kontrolü
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 3. Dataset Hazırlama

### Adım 3.1: Dataset Yapısı

Dataset'inizi `database/` klasörüne aşağıdaki yapıda yerleştirin:

```
database/
├── images/
│   ├── train/
│   │   ├── drone_001.jpg
│   │   ├── drone_002.jpg
│   │   ├── drone_003.png
│   │   └── ...
│   └── val/
│       ├── drone_101.jpg
│       ├── drone_102.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── drone_001.txt
│   │   ├── drone_002.txt
│   │   ├── drone_003.txt
│   │   └── ...
│   └── val/
│       ├── drone_101.txt
│       ├── drone_102.txt
│       └── ...
└── drone_dataset.yaml
```

### Adım 3.2: Label Formatı (YOLO Format)

Her görüntü için aynı isimde `.txt` dosyası oluşturun. Her satır bir nesne:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Sınıf numarası (drone için 0)
- `x_center`: Bbox merkezi X (0-1 arası, normalize)
- `y_center`: Bbox merkezi Y (0-1 arası, normalize)
- `width`: Bbox genişliği (0-1 arası, normalize)
- `height`: Bbox yüksekliği (0-1 arası, normalize)

**Örnek label dosyası (drone_001.txt):**
```
0 0.456 0.523 0.120 0.085
0 0.782 0.234 0.095 0.067
```

### Adım 3.3: Dataset Config Dosyası

`database/drone_dataset.yaml` dosyasını oluşturun:

```yaml
# Dataset root path
path: /Users/kutinyo/Desktop/Dilos/AvcıDroneTespit/hunter_drone/database

# Görüntü dizinleri (path'e göre relative)
train: images/train
val: images/val

# Sınıf isimleri
names:
  0: drone

# Opsiyonel: Birden fazla sınıf
# names:
#   0: drone
#   1: bird
#   2: airplane
```

### Adım 3.4: Dataset'i Doğrula
```bash
# Dosya sayılarını kontrol et
echo "Train images:" && ls database/images/train | wc -l
echo "Train labels:" && ls database/labels/train | wc -l
echo "Val images:" && ls database/images/val | wc -l
echo "Val labels:" && ls database/labels/val | wc -l
```

> **Not**: Görüntü ve label sayıları eşleşmeli!

---

## 4. Model Eğitimi

### Adım 4.1: Base Model İndir

YOLO11 modellerinden birini seçin:

| Model | Boyut | Hız | Doğruluk |
|-------|-------|-----|----------|
| yolo11n.pt | 6MB | En hızlı | Düşük |
| yolo11s.pt | 22MB | Hızlı | Orta |
| yolo11m.pt | 42MB | Dengeli | İyi |
| yolo11l.pt | 87MB | Yavaş | Yüksek |
| yolo11x.pt | 137MB | En yavaş | En yüksek |

```bash
# Model otomatik indirilir, veya manuel:
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

### Adım 4.2: Eğitimi Başlat

```bash
# Temel eğitim
python scripts/run_training.py \
    --data database/drone_dataset.yaml \
    --model yolo11m.pt \
    --epochs 100

# Özelleştirilmiş eğitim
python scripts/run_training.py \
    --data database/drone_dataset.yaml \
    --model yolo11m.pt \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name drone_detector_v1
```

### Adım 4.3: Eğitim Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--data` | - | Dataset YAML dosyası |
| `--model` | yolo11m.pt | Base model |
| `--epochs` | 100 | Eğitim epoch sayısı |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Görüntü boyutu |
| `--device` | 0 | GPU ID (0, 1, cpu, mps) |
| `--name` | - | Deney adı |

### Adım 4.4: Eğitimi İzle

Eğitim sırasında TensorBoard kullanabilirsiniz:
```bash
tensorboard --logdir runs/detect
```
Tarayıcıda: http://localhost:6006

### Adım 4.5: Eğitim Çıktıları

Eğitim tamamlandığında:
```
runs/detect/drone_detector_v1/
├── weights/
│   ├── best.pt      # En iyi model (val loss'a göre)
│   └── last.pt      # Son epoch modeli
├── results.csv      # Metrikler
├── confusion_matrix.png
├── results.png      # Loss/metric grafikleri
└── ...
```

En iyi model otomatik olarak `models/yolo11_drone.pt` olarak kopyalanır.

---

## 5. Inference Çalıştırma

### Adım 5.1: Config Dosyasını Düzenle

`configs/default.yaml` dosyasını açın ve model yolunu güncelleyin:

```yaml
detector:
  model_path: models/yolo11_drone.pt  # Eğitilmiş modeliniz
  confidence_threshold: 0.5
  device: cuda  # veya cpu, mps
```

### Adım 5.2: Video ile Çalıştır

```bash
# Temel kullanım
python scripts/run_inference.py \
    --config configs/default.yaml \
    --video path/to/your/video.mp4

# Sonuçları dosyaya kaydet
python scripts/run_inference.py \
    --config configs/default.yaml \
    --video input.mp4 \
    --output results.jsonl

# Farklı profil kullan
python scripts/run_inference.py \
    --config configs/profiles/low_latency.yaml \
    --video input.mp4
```

### Adım 5.3: Inference Parametreleri

| Parametre | Açıklama |
|-----------|----------|
| `--config` | Konfigürasyon dosyası |
| `--video` | Girdi video dosyası |
| `--output` | Çıktı JSON dosyası |
| `--model` | Model yolunu override et |
| `--device` | Cihazı override et (cuda/cpu/mps) |
| `--confidence` | Güven eşiğini override et |

### Adım 5.4: Çıktı Formatı

JSON Lines formatında çıktı (her satır bir frame):

```json
{
  "msg_version": "1.0",
  "timestamp_ms": 1702732800000,
  "frame_id": 100,
  "model": {
    "detector_name": "yolo11_drone",
    "detector_hash": "a1b2c3d4..."
  },
  "pipeline_metrics": {
    "detect_ms": 18.5,
    "total_e2e_ms": 35.2
  },
  "tracks": [
    {
      "track_id": 1,
      "state": "TRACK",
      "confidence": 0.95,
      "bbox_xyxy": [100, 150, 200, 250],
      "velocity_px_per_s": [50.0, -10.0],
      "trajectory_tail": [
        {"t_ms": 1000, "cx": 150, "cy": 200},
        {"t_ms": 1033, "cx": 152, "cy": 199}
      ]
    }
  ]
}
```

---

## 6. GUI Uygulaması

Komut satırı yerine görsel arayüz kullanmak istiyorsanız, Hunter Drone GUI uygulamasını kullanabilirsiniz.

### 6.1 GUI'yi Başlatma

**Windows:**
```cmd
launch_gui.bat
```

**Mac/Linux:**
```bash
./launch_gui.sh
```

**Veya doğrudan Python ile:**
```bash
source venv/bin/activate  # Mac/Linux
# veya: venv\Scripts\activate  # Windows

python hunter_gui.py
```

### 6.2 GUI Özellikleri

GUI uygulaması dört ana sekmeden oluşur:

#### Dataset Sekmesi
- Dataset klasörünü seçme ve doğrulama
- Görüntü ve label sayılarını kontrol etme
- `drone_dataset.yaml` dosyasını otomatik oluşturma
- Eksik dosyaları tespit etme

#### Eğitim Sekmesi
- Base model seçimi (yolo11n/s/m/l/x)
- Eğitim parametrelerini ayarlama (epochs, batch size, image size)
- Eğitimi başlatma/durdurma
- Gerçek zamanlı eğitim loglarını izleme
- TensorBoard'u açma

#### Inference Sekmesi
- Video dosyası seçme
- Konfigürasyon profili seçme
- Model override (opsiyonel)
- Confidence threshold ayarlama
- Inference başlatma/durdurma
- Sonuçları JSON dosyasına kaydetme

#### Ayarlar Sekmesi
- Sistem bilgilerini görüntüleme (Python, PyTorch, CUDA durumu)
- Proje klasörlerine hızlı erişim
- Yardım bilgileri

### 6.3 GUI Ekran Görüntüsü

```
┌─────────────────────────────────────────────────────────────────┐
│  Hunter Drone Detection System                           v1.0.0 │
├─────────────────────────────────────────────────────────────────┤
│  [ Dataset ]  [ Eğitim ]  [ Inference ]  [ Ayarlar ]           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Dataset Yolu ──────────────────────────────────────────┐   │
│  │ /path/to/hunter_drone/database          [ Gözat... ]    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [ Dataset'i Doğrula ]  [ YAML Oluştur ]                       │
│                                                                 │
│  ┌─ Doğrulama Sonuçları ───────────────────────────────────┐   │
│  │ [OK] images/train: 1500 dosya                           │   │
│  │ [OK] images/val: 300 dosya                              │   │
│  │ [OK] labels/train: 1500 dosya                           │   │
│  │ [OK] labels/val: 300 dosya                              │   │
│  │ [OK] drone_dataset.yaml mevcut                          │   │
│  │                                                         │   │
│  │ [BAŞARILI] Dataset kullanıma hazır!                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Hazır                                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Konfigürasyon

### 7.1 Profil Seçimi

| Profil | Kullanım Alanı | FPS |
|--------|----------------|-----|
| `default.yaml` | Genel kullanım | ~25 |
| `low_latency.yaml` | Gerçek zamanlı | ~50+ |
| `high_accuracy.yaml` | Offline analiz | ~10 |

### 7.2 Önemli Parametreler

```yaml
# Tespit ayarları
detector:
  confidence_threshold: 0.5  # Düşük = daha fazla tespit, daha fazla false positive
  nms_threshold: 0.45        # NMS için IoU eşiği

# Takip ayarları
tracking:
  # State machine (Eagle Model)
  lock_confirm_frames: 3     # LOCK→TRACK için gereken frame sayısı
  lost_timeout_frames: 30    # TRACK→LOST için gereken kayıp frame sayısı

  # Association
  iou_threshold: 0.3         # Eşleştirme için minimum IoU
  embedding_weight: 0.3      # Görünüm vs hareket ağırlığı (0=sadece IoU, 1=sadece embedding)
```

### 7.3 State Machine (Eagle Model)

```
     ┌─────────────────────────────────────────────────────┐
     │                                                     │
     ▼                                                     │
  SEARCH ──match──► LOCK ──confirm──► TRACK ──lost──► LOST │
                      │                  │              │   │
                      │                  │              │   │
                   timeout            ◄──┘           match  │
                      │                                │   │
                      ▼                                ▼   │
                   DROPPED ◄─────timeout────────── RECOVER─┘
```

---

## 8. Python API Kullanımı

### 8.1 Temel Kullanım

```python
from hunter import Pipeline, HunterConfig

# Config yükle
config = HunterConfig.from_yaml("configs/default.yaml")

# Video kaynağını ayarla
config.ingest.source_uri = "video.mp4"
config.ingest.source_type = "file"

# Pipeline çalıştır
with Pipeline(config) as pipeline:
    for message in pipeline.run():
        print(f"Frame {message.frame_id}: {message.track_count} tracks")

        for track in message.tracks:
            print(f"  Track {track.track_id}: {track.state}")
            print(f"    BBox: {track.bbox_xyxy}")
            print(f"    Confidence: {track.confidence:.2f}")
            print(f"    Velocity: {track.velocity_px_per_s}")
```

### 8.2 Callback ile Kullanım

```python
from hunter import Pipeline, HunterConfig
from hunter.pipeline.output import CallbackSink

def process_tracks(message):
    """Her frame için çağrılır."""
    for track in message.tracks:
        if track.state == "TRACK":
            # Aktif track işle
            print(f"Active drone at {track.bbox_xyxy}")

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

# Callback sink kullan
pipeline = Pipeline(config)
pipeline._output = CallbackSink(process_tracks)

for _ in pipeline.run():
    pass

pipeline.close()
```

### 8.3 Metrikleri Al

```python
with Pipeline(config) as pipeline:
    for message in pipeline.run():
        pass

    # Final metrikler
    metrics = pipeline.get_metrics()

    print(f"İşlenen frame: {metrics['frames_processed']}")
    print(f"Çalışma süresi: {metrics['runtime_seconds']:.1f}s")
    print(f"Ortalama FPS: {metrics['throughput']['fps']:.1f}")
    print(f"Latency p95: {metrics['latency']['p95_ms']:.1f}ms")
    print(f"Toplam track: {metrics['tracker']['next_track_id']}")
```

---

## 9. Sorun Giderme

### 9.1 CUDA Hataları

**Hata:** `CUDA out of memory`
```bash
# Batch size'ı düşür
python scripts/run_training.py --batch 8 ...

# Veya image size'ı düşür
python scripts/run_training.py --imgsz 416 ...
```

**Hata:** `CUDA not available`
```bash
# CUDA kurulumunu kontrol et
python -c "import torch; print(torch.cuda.is_available())"

# CPU kullan
python scripts/run_inference.py --device cpu ...
```

### 9.2 Import Hataları

**Hata:** `ModuleNotFoundError: No module named 'hunter'`
```bash
# Paketi yeniden yükle
pip install -e .

# PYTHONPATH kontrol et
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 9.3 Dataset Hataları

**Hata:** `No labels found`
```bash
# Label dosyalarının varlığını kontrol et
ls database/labels/train/

# Label formatını kontrol et (boşlukla ayrılmış olmalı)
head -1 database/labels/train/sample.txt
```

**Hata:** `Image and label count mismatch`
- Her görüntü için aynı isimde label dosyası olmalı
- Boş label dosyası = o görüntüde nesne yok

### 9.4 Düşük Performans

**Düşük FPS:**
1. `configs/profiles/low_latency.yaml` kullanın
2. Daha küçük model seçin (yolo11n veya yolo11s)
3. Input size'ı düşürün (416x416)
4. Embedder'ı devre dışı bırakın (`embedder.enabled: false`)

**Düşük Doğruluk:**
1. `configs/profiles/high_accuracy.yaml` kullanın
2. Daha büyük model seçin (yolo11l veya yolo11x)
3. Confidence threshold'u düşürün
4. Daha fazla epoch eğitin

### 9.5 Tracking Sorunları

**Çok fazla ID switch:**
- `embedding_weight` değerini artırın (0.5-0.7)
- `iou_threshold` değerini artırın
- Siamese embedder kullanın

**Track'ler çok erken kayboluyor:**
- `lost_timeout_frames` değerini artırın
- `recover_max_frames` değerini artırın

**Çok fazla false positive track:**
- `lock_confirm_frames` değerini artırın
- `confidence_threshold` değerini artırın

---

## Hızlı Başlangıç Özeti

```bash
# 1. Kurulum
cd hunter_drone
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,training]"

# 2. Dataset'i database/ klasörüne kopyala
# (images/train, images/val, labels/train, labels/val)

# 3. Dataset config oluştur
cp database/drone_dataset.yaml.template database/drone_dataset.yaml
# Düzenle: path'leri ayarla

# 4. Eğitim
python scripts/run_training.py \
    --data database/drone_dataset.yaml \
    --model yolo11m.pt \
    --epochs 100

# 5. Inference
python scripts/run_inference.py \
    --config configs/default.yaml \
    --video test_video.mp4 \
    --output results.jsonl
```

---

## Yardım ve Destek

- **Dokümantasyon**: Bu dosya ve `README.md`
- **Testler**: `pytest tests/` ile testleri çalıştırın
- **Loglar**: `logs/` klasöründe detaylı loglar

İyi çalışmalar!
