#!/usr/bin/env python3
"""
Hunter Drone - Desktop GUI Application

Dataset dogrulama, model egitimi ve inference islemleri icin
basit bir masaustu arayuzu.

Kullanim:
    python hunter_gui.py
"""

import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.absolute()


class HunterGUI:
    """Ana GUI uygulamasi."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hunter Drone - Control Panel")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Tema ayarlari
        self.style = ttk.Style()
        self._setup_theme()

        # Degiskenler
        self.training_process: Optional[subprocess.Popen] = None
        self.inference_process: Optional[subprocess.Popen] = None

        # Ana cerceve
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Baslik
        self._create_header()

        # Tab kontrol
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Tablari olustur
        self._create_dataset_tab()
        self._create_training_tab()
        self._create_inference_tab()
        self._create_settings_tab()

        # Status bar
        self._create_status_bar()

    def _setup_theme(self):
        """Tema ayarlari."""
        self.style.theme_use("clam")

        # Ozel renkler
        self.colors = {
            "primary": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "danger": "#f44336",
            "bg": "#f5f5f5",
            "text": "#212121",
        }

        # Button stilleri
        self.style.configure("Primary.TButton", font=("Helvetica", 10))
        self.style.configure("Success.TButton", font=("Helvetica", 10))
        self.style.configure("Danger.TButton", font=("Helvetica", 10))

    def _create_header(self):
        """Baslik bolumu."""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(
            header_frame,
            text="Hunter Drone Detection System",
            font=("Helvetica", 18, "bold"),
        )
        title_label.pack(side=tk.LEFT)

        version_label = ttk.Label(
            header_frame, text="v1.0.0", font=("Helvetica", 10), foreground="gray"
        )
        version_label.pack(side=tk.RIGHT, padx=10)

    def _create_status_bar(self):
        """Status bar."""
        self.status_var = tk.StringVar(value="Hazir")
        status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))

    # ==================== DATASET TAB ====================

    def _create_dataset_tab(self):
        """Dataset dogrulama tabi."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Dataset  ")

        # Dataset yolu
        path_frame = ttk.LabelFrame(tab, text="Dataset Yolu", padding="10")
        path_frame.pack(fill=tk.X, pady=(0, 10))

        self.dataset_path_var = tk.StringVar(
            value=str(PROJECT_ROOT / "database")
        )
        path_entry = ttk.Entry(path_frame, textvariable=self.dataset_path_var, width=60)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="Gozat...", command=self._browse_dataset
        )
        browse_btn.pack(side=tk.RIGHT)

        # Dogrulama butonu
        validate_frame = ttk.Frame(tab)
        validate_frame.pack(fill=tk.X, pady=(0, 10))

        validate_btn = ttk.Button(
            validate_frame,
            text="Dataset'i Dogrula",
            command=self._validate_dataset,
            style="Primary.TButton",
        )
        validate_btn.pack(side=tk.LEFT)

        create_yaml_btn = ttk.Button(
            validate_frame,
            text="YAML Olustur",
            command=self._create_dataset_yaml,
        )
        create_yaml_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Sonuc alani
        result_frame = ttk.LabelFrame(tab, text="Dogrulama Sonuclari", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.dataset_result_text = scrolledtext.ScrolledText(
            result_frame, height=20, font=("Consolas", 10)
        )
        self.dataset_result_text.pack(fill=tk.BOTH, expand=True)

    def _browse_dataset(self):
        """Dataset klasoru sec."""
        path = filedialog.askdirectory(
            initialdir=PROJECT_ROOT,
            title="Dataset Klasoru Sec",
        )
        if path:
            self.dataset_path_var.set(path)

    def _validate_dataset(self):
        """Dataset'i dogrula."""
        dataset_path = Path(self.dataset_path_var.get())
        self.dataset_result_text.delete(1.0, tk.END)

        results = []
        results.append(f"Dataset Yolu: {dataset_path}\n")
        results.append("=" * 50 + "\n\n")

        # Klasor kontrolu
        if not dataset_path.exists():
            results.append("[HATA] Dataset klasoru bulunamadi!\n")
            self._update_dataset_result(results)
            return

        # Alt klasorler
        images_train = dataset_path / "images" / "train"
        images_val = dataset_path / "images" / "val"
        labels_train = dataset_path / "labels" / "train"
        labels_val = dataset_path / "labels" / "val"

        # Her klasoru kontrol et
        folders = [
            ("images/train", images_train),
            ("images/val", images_val),
            ("labels/train", labels_train),
            ("labels/val", labels_val),
        ]

        all_ok = True
        for name, folder in folders:
            if folder.exists():
                count = len(list(folder.iterdir()))
                results.append(f"[OK] {name}: {count} dosya\n")
            else:
                results.append(f"[EKSIK] {name}: Klasor bulunamadi\n")
                all_ok = False

        results.append("\n")

        # Goruntu-label eslesmesi
        if images_train.exists() and labels_train.exists():
            train_images = {f.stem for f in images_train.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
            train_labels = {f.stem for f in labels_train.iterdir() if f.suffix == ".txt"}

            missing_labels = train_images - train_labels
            extra_labels = train_labels - train_images

            results.append(f"Train Eslesmesi:\n")
            results.append(f"  Goruntuler: {len(train_images)}\n")
            results.append(f"  Labellar: {len(train_labels)}\n")

            if missing_labels:
                results.append(f"  [UYARI] Label eksik: {len(missing_labels)} goruntu\n")
                if len(missing_labels) <= 5:
                    for m in missing_labels:
                        results.append(f"    - {m}\n")
            if extra_labels:
                results.append(f"  [UYARI] Fazla label: {len(extra_labels)}\n")

            if not missing_labels and not extra_labels:
                results.append(f"  [OK] Tum goruntuler eslesiyor\n")

        results.append("\n")

        # Val set
        if images_val.exists() and labels_val.exists():
            val_images = {f.stem for f in images_val.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
            val_labels = {f.stem for f in labels_val.iterdir() if f.suffix == ".txt"}

            results.append(f"Val Eslesmesi:\n")
            results.append(f"  Goruntuler: {len(val_images)}\n")
            results.append(f"  Labellar: {len(val_labels)}\n")

            missing = val_images - val_labels
            if not missing:
                results.append(f"  [OK] Tum goruntuler eslesiyor\n")
            else:
                results.append(f"  [UYARI] Label eksik: {len(missing)} goruntu\n")

        results.append("\n")

        # YAML kontrolu
        yaml_file = dataset_path / "drone_dataset.yaml"
        if yaml_file.exists():
            results.append(f"[OK] drone_dataset.yaml mevcut\n")
        else:
            results.append(f"[EKSIK] drone_dataset.yaml bulunamadi\n")
            results.append(f"       'YAML Olustur' butonunu kullanin\n")

        results.append("\n" + "=" * 50 + "\n")
        if all_ok:
            results.append("\n[BASARILI] Dataset kullanima hazir!\n")
            self.status_var.set("Dataset dogrulama: Basarili")
        else:
            results.append("\n[UYARI] Bazi dosyalar eksik!\n")
            self.status_var.set("Dataset dogrulama: Eksikler var")

        self._update_dataset_result(results)

    def _update_dataset_result(self, results: list):
        """Sonuclari goster."""
        self.dataset_result_text.delete(1.0, tk.END)
        self.dataset_result_text.insert(tk.END, "".join(results))

    def _create_dataset_yaml(self):
        """Dataset YAML dosyasi olustur."""
        dataset_path = Path(self.dataset_path_var.get())

        yaml_content = f"""# Hunter Drone Dataset Configuration
# Otomatik olusturuldu: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Dataset root path
path: {dataset_path}

# Train/val image directories (relative to path)
train: images/train
val: images/val

# Class names
names:
  0: drone
"""
        yaml_file = dataset_path / "drone_dataset.yaml"

        try:
            yaml_file.write_text(yaml_content)
            messagebox.showinfo(
                "Basarili",
                f"YAML dosyasi olusturuldu:\n{yaml_file}",
            )
            self._validate_dataset()
        except Exception as e:
            messagebox.showerror("Hata", f"YAML olusturulamadi:\n{e}")

    # ==================== TRAINING TAB ====================

    def _create_training_tab(self):
        """Model egitimi tabi."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Egitim  ")

        # Sol panel - Ayarlar
        left_frame = ttk.Frame(tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Dataset secimi
        dataset_frame = ttk.LabelFrame(left_frame, text="Dataset", padding="10")
        dataset_frame.pack(fill=tk.X, pady=(0, 10))

        self.train_dataset_var = tk.StringVar(
            value=str(PROJECT_ROOT / "database" / "drone_dataset.yaml")
        )
        ttk.Entry(dataset_frame, textvariable=self.train_dataset_var, width=40).pack(
            fill=tk.X
        )
        ttk.Button(
            dataset_frame, text="Sec...", command=self._browse_train_dataset
        ).pack(anchor=tk.E, pady=(5, 0))

        # Model secimi
        model_frame = ttk.LabelFrame(left_frame, text="Base Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.base_model_var = tk.StringVar(value="yolo11m.pt")
        models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        model_combo = ttk.Combobox(
            model_frame, textvariable=self.base_model_var, values=models, state="readonly"
        )
        model_combo.pack(fill=tk.X)

        model_info = ttk.Label(
            model_frame,
            text="n:Hizli  s:Dengeli  m:Onerilen  l:Buyuk  x:EnIyi",
            font=("Helvetica", 8),
            foreground="gray",
        )
        model_info.pack(anchor=tk.W, pady=(5, 0))

        # Egitim parametreleri
        params_frame = ttk.LabelFrame(left_frame, text="Parametreler", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(
            params_frame, from_=10, to=500, textvariable=self.epochs_var, width=10
        ).grid(row=0, column=1, sticky=tk.W, pady=2)

        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.batch_var = tk.IntVar(value=16)
        ttk.Spinbox(
            params_frame, from_=1, to=64, textvariable=self.batch_var, width=10
        ).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Image size
        ttk.Label(params_frame, text="Image Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.imgsz_var = tk.IntVar(value=640)
        imgsz_combo = ttk.Combobox(
            params_frame,
            textvariable=self.imgsz_var,
            values=[320, 416, 512, 640, 800, 1024],
            width=8,
        )
        imgsz_combo.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Device
        ttk.Label(params_frame, text="Device:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar(value="0")
        device_combo = ttk.Combobox(
            params_frame,
            textvariable=self.device_var,
            values=["0", "cpu", "mps"],
            width=8,
        )
        device_combo.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Proje ismi
        ttk.Label(params_frame, text="Proje Adi:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.project_name_var = tk.StringVar(value="drone_detector")
        ttk.Entry(params_frame, textvariable=self.project_name_var, width=15).grid(
            row=4, column=1, sticky=tk.W, pady=2
        )

        # Butonlar
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.train_btn = ttk.Button(
            btn_frame,
            text="Egitimi Baslat",
            command=self._start_training,
            style="Success.TButton",
        )
        self.train_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_train_btn = ttk.Button(
            btn_frame,
            text="Egitimi Durdur",
            command=self._stop_training,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.stop_train_btn.pack(fill=tk.X)

        # TensorBoard
        tb_btn = ttk.Button(
            btn_frame, text="TensorBoard Ac", command=self._open_tensorboard
        )
        tb_btn.pack(fill=tk.X, pady=(10, 0))

        # Sag panel - Log
        right_frame = ttk.LabelFrame(tab, text="Egitim Logu", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.train_log_text = scrolledtext.ScrolledText(
            right_frame, height=25, font=("Consolas", 9)
        )
        self.train_log_text.pack(fill=tk.BOTH, expand=True)

    def _browse_train_dataset(self):
        """Egitim dataset'i sec."""
        path = filedialog.askopenfilename(
            initialdir=PROJECT_ROOT / "database",
            title="Dataset YAML Sec",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.train_dataset_var.set(path)

    def _start_training(self):
        """Egitimi baslat."""
        dataset = self.train_dataset_var.get()
        if not Path(dataset).exists():
            messagebox.showerror("Hata", "Dataset YAML dosyasi bulunamadi!")
            return

        self.train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.NORMAL)
        self.train_log_text.delete(1.0, tk.END)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_training.py"),
            "--data", dataset,
            "--model", self.base_model_var.get(),
            "--epochs", str(self.epochs_var.get()),
            "--batch", str(self.batch_var.get()),
            "--imgsz", str(self.imgsz_var.get()),
            "--device", self.device_var.get(),
            "--name", self.project_name_var.get(),
        ]

        self._log_training(f"Komut: {' '.join(cmd)}\n\n")
        self.status_var.set("Egitim baslatiliyor...")

        # Thread ile calistir
        thread = threading.Thread(target=self._run_training, args=(cmd,), daemon=True)
        thread.start()

    def _run_training(self, cmd: list):
        """Egitimi ayri thread'de calistir."""
        try:
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT,
            )

            for line in iter(self.training_process.stdout.readline, ""):
                if line:
                    self.root.after(0, self._log_training, line)

            self.training_process.wait()

            if self.training_process.returncode == 0:
                self.root.after(0, self._training_complete, True)
            else:
                self.root.after(0, self._training_complete, False)

        except Exception as e:
            self.root.after(0, self._log_training, f"\nHata: {e}\n")
            self.root.after(0, self._training_complete, False)

    def _log_training(self, text: str):
        """Egitim loguna yaz."""
        self.train_log_text.insert(tk.END, text)
        self.train_log_text.see(tk.END)

    def _training_complete(self, success: bool):
        """Egitim tamamlandi."""
        self.train_btn.config(state=tk.NORMAL)
        self.stop_train_btn.config(state=tk.DISABLED)
        self.training_process = None

        if success:
            self.status_var.set("Egitim tamamlandi!")
            messagebox.showinfo("Basarili", "Model egitimi tamamlandi!")
        else:
            self.status_var.set("Egitim hatasi!")

    def _stop_training(self):
        """Egitimi durdur."""
        if self.training_process:
            self.training_process.terminate()
            self._log_training("\n\n[DURDURULDU] Egitim kullanici tarafindan durduruldu.\n")
            self.status_var.set("Egitim durduruldu")
            self.train_btn.config(state=tk.NORMAL)
            self.stop_train_btn.config(state=tk.DISABLED)

    def _open_tensorboard(self):
        """TensorBoard ac."""
        try:
            subprocess.Popen(
                [sys.executable, "-m", "tensorboard", "--logdir", "runs/detect"],
                cwd=PROJECT_ROOT,
            )
            messagebox.showinfo(
                "TensorBoard",
                "TensorBoard baslatildi!\n\nTarayicinizda acin:\nhttp://localhost:6006",
            )
        except Exception as e:
            messagebox.showerror("Hata", f"TensorBoard baslatilamadi:\n{e}")

    # ==================== INFERENCE TAB ====================

    def _create_inference_tab(self):
        """Inference tabi."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Inference  ")

        # Sol panel
        left_frame = ttk.Frame(tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Video secimi
        video_frame = ttk.LabelFrame(left_frame, text="Video Dosyasi", padding="10")
        video_frame.pack(fill=tk.X, pady=(0, 10))

        self.video_path_var = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.video_path_var, width=40).pack(fill=tk.X)
        ttk.Button(video_frame, text="Sec...", command=self._browse_video).pack(
            anchor=tk.E, pady=(5, 0)
        )

        # Config secimi
        config_frame = ttk.LabelFrame(left_frame, text="Konfigurasyon", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        self.config_var = tk.StringVar(value="default.yaml")
        configs = ["default.yaml", "profiles/low_latency.yaml", "profiles/high_accuracy.yaml"]
        ttk.Combobox(
            config_frame, textvariable=self.config_var, values=configs, width=35
        ).pack(fill=tk.X)

        # Model override
        model_frame = ttk.LabelFrame(left_frame, text="Model (Opsiyonel)", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.inf_model_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.inf_model_var, width=40).pack(fill=tk.X)
        ttk.Button(model_frame, text="Sec...", command=self._browse_model).pack(
            anchor=tk.E, pady=(5, 0)
        )

        # Parametreler
        params_frame = ttk.LabelFrame(left_frame, text="Parametreler", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(params_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            params_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
        ).grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(params_frame, text="Device:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.inf_device_var = tk.StringVar(value="cuda")
        ttk.Combobox(
            params_frame,
            textvariable=self.inf_device_var,
            values=["cuda", "cpu", "mps"],
            width=10,
        ).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Cikti
        output_frame = ttk.LabelFrame(left_frame, text="Cikti Dosyasi", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        self.output_var = tk.StringVar(value="output/results.jsonl")
        ttk.Entry(output_frame, textvariable=self.output_var, width=40).pack(fill=tk.X)

        # Butonlar
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.inf_btn = ttk.Button(
            btn_frame,
            text="Inference Baslat",
            command=self._start_inference,
            style="Success.TButton",
        )
        self.inf_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_inf_btn = ttk.Button(
            btn_frame,
            text="Durdur",
            command=self._stop_inference,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.stop_inf_btn.pack(fill=tk.X)

        # Sag panel - Log
        right_frame = ttk.LabelFrame(tab, text="Inference Logu", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.inf_log_text = scrolledtext.ScrolledText(
            right_frame, height=25, font=("Consolas", 9)
        )
        self.inf_log_text.pack(fill=tk.BOTH, expand=True)

    def _browse_video(self):
        """Video dosyasi sec."""
        path = filedialog.askopenfilename(
            title="Video Sec",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path_var.set(path)

    def _browse_model(self):
        """Model dosyasi sec."""
        path = filedialog.askopenfilename(
            initialdir=PROJECT_ROOT / "models",
            title="Model Sec",
            filetypes=[("Model files", "*.pt *.onnx"), ("All files", "*.*")],
        )
        if path:
            self.inf_model_var.set(path)

    def _start_inference(self):
        """Inference baslat."""
        video = self.video_path_var.get()
        if not video or not Path(video).exists():
            messagebox.showerror("Hata", "Lutfen gecerli bir video dosyasi secin!")
            return

        self.inf_btn.config(state=tk.DISABLED)
        self.stop_inf_btn.config(state=tk.NORMAL)
        self.inf_log_text.delete(1.0, tk.END)

        config_path = PROJECT_ROOT / "configs" / self.config_var.get()

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_inference.py"),
            "--config", str(config_path),
            "--video", video,
            "--output", self.output_var.get(),
            "--device", self.inf_device_var.get(),
            "--confidence", str(self.confidence_var.get()),
        ]

        if self.inf_model_var.get():
            cmd.extend(["--model", self.inf_model_var.get()])

        self._log_inference(f"Komut: {' '.join(cmd)}\n\n")
        self.status_var.set("Inference baslatiliyor...")

        thread = threading.Thread(target=self._run_inference, args=(cmd,), daemon=True)
        thread.start()

    def _run_inference(self, cmd: list):
        """Inference'i ayri thread'de calistir."""
        try:
            self.inference_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT,
            )

            for line in iter(self.inference_process.stdout.readline, ""):
                if line:
                    self.root.after(0, self._log_inference, line)

            self.inference_process.wait()

            if self.inference_process.returncode == 0:
                self.root.after(0, self._inference_complete, True)
            else:
                self.root.after(0, self._inference_complete, False)

        except Exception as e:
            self.root.after(0, self._log_inference, f"\nHata: {e}\n")
            self.root.after(0, self._inference_complete, False)

    def _log_inference(self, text: str):
        """Inference loguna yaz."""
        self.inf_log_text.insert(tk.END, text)
        self.inf_log_text.see(tk.END)

    def _inference_complete(self, success: bool):
        """Inference tamamlandi."""
        self.inf_btn.config(state=tk.NORMAL)
        self.stop_inf_btn.config(state=tk.DISABLED)
        self.inference_process = None

        if success:
            self.status_var.set("Inference tamamlandi!")
            output_file = self.output_var.get()
            messagebox.showinfo(
                "Basarili",
                f"Inference tamamlandi!\n\nSonuclar: {output_file}",
            )
        else:
            self.status_var.set("Inference hatasi!")

    def _stop_inference(self):
        """Inference durdur."""
        if self.inference_process:
            self.inference_process.terminate()
            self._log_inference("\n\n[DURDURULDU] Inference durduruldu.\n")
            self.status_var.set("Inference durduruldu")
            self.inf_btn.config(state=tk.NORMAL)
            self.stop_inf_btn.config(state=tk.DISABLED)

    # ==================== SETTINGS TAB ====================

    def _create_settings_tab(self):
        """Ayarlar tabi."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Ayarlar  ")

        # Sistem bilgisi
        info_frame = ttk.LabelFrame(tab, text="Sistem Bilgisi", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_text = scrolledtext.ScrolledText(info_frame, height=15, font=("Consolas", 10))
        info_text.pack(fill=tk.X)

        # Sistem bilgilerini topla
        info_lines = [
            f"Proje Dizini: {PROJECT_ROOT}",
            f"Python: {sys.version}",
            "",
        ]

        # PyTorch/CUDA kontrolu
        try:
            import torch
            info_lines.append(f"PyTorch: {torch.__version__}")
            info_lines.append(f"CUDA Mevcut: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                info_lines.append(f"CUDA Versiyon: {torch.version.cuda}")
                info_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            if hasattr(torch.backends, "mps"):
                info_lines.append(f"MPS Mevcut: {torch.backends.mps.is_available()}")
        except ImportError:
            info_lines.append("PyTorch: Yuklu degil")

        info_lines.append("")

        # Ultralytics kontrolu
        try:
            import ultralytics
            info_lines.append(f"Ultralytics: {ultralytics.__version__}")
        except ImportError:
            info_lines.append("Ultralytics: Yuklu degil")

        info_text.insert(tk.END, "\n".join(info_lines))
        info_text.config(state=tk.DISABLED)

        # Hizli erisim
        quick_frame = ttk.LabelFrame(tab, text="Hizli Erisim", padding="10")
        quick_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            quick_frame,
            text="Proje Klasorunu Ac",
            command=lambda: self._open_folder(PROJECT_ROOT),
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            quick_frame,
            text="Models Klasorunu Ac",
            command=lambda: self._open_folder(PROJECT_ROOT / "models"),
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            quick_frame,
            text="Output Klasorunu Ac",
            command=lambda: self._open_folder(PROJECT_ROOT / "output"),
        ).pack(side=tk.LEFT)

        # Yardim
        help_frame = ttk.LabelFrame(tab, text="Yardim", padding="10")
        help_frame.pack(fill=tk.X)

        help_text = """
Hunter Drone Detection System v1.0

Kullanim:
1. Dataset tabinda dataset'inizi dogrulayin
2. Egitim tabinda model egitimini yapin
3. Inference tabinda video uzerinde tespit yapin

Dokumantasyon icin STARTING_GUIDE.md dosyasina bakin.
        """
        ttk.Label(help_frame, text=help_text.strip(), justify=tk.LEFT).pack(anchor=tk.W)

    def _open_folder(self, path: Path):
        """Klasoru dosya yoneticisinde ac."""
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])


def main():
    """Ana fonksiyon."""
    root = tk.Tk()

    # Icon ayarla (varsa)
    icon_path = PROJECT_ROOT / "assets" / "icon.png"
    if icon_path.exists():
        try:
            icon = tk.PhotoImage(file=str(icon_path))
            root.iconphoto(True, icon)
        except Exception:
            pass

    app = HunterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
