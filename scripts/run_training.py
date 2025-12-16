#!/usr/bin/env python3
"""
Train YOLO11 detector on drone dataset.

Usage:
    python scripts/run_training.py --data database/drone_dataset.yaml --model yolo11m.pt --epochs 100
    python scripts/run_training.py --data database/drone_dataset.yaml --model yolo11n.pt --epochs 50 --batch 32
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def train_detector(args):
    """Train YOLO11 detector."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package required. Install with: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("YOLO11 Drone Detector Training")
    print("=" * 60)
    print(f"  Dataset: {args.data}")
    print(f"  Base model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    print()

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/detect",
        name=args.name or "drone_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

    # Copy best model to models folder
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        dest = Path("models") / "yolo11_drone.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(best_model, dest)
        print(f"Copied best model to: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 drone detector")
    parser.add_argument("--data", "-d", type=Path, required=True,
                        help="Path to dataset YAML file")
    parser.add_argument("--model", "-m", type=str, default="yolo11m.pt",
                        help="Base model (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (0, 1, cpu, mps)")
    parser.add_argument("--name", "-n", type=str,
                        help="Experiment name")
    args = parser.parse_args()

    train_detector(args)


if __name__ == "__main__":
    main()
