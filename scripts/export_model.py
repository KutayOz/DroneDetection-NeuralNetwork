#!/usr/bin/env python3
"""
Export trained models to different formats.

Usage:
    python scripts/export_model.py --model models/yolo11_drone.pt --format onnx
    python scripts/export_model.py --model models/yolo11_drone.pt --format engine --half
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def export_yolo(args):
    """Export YOLO model to specified format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package required")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))

    print(f"Exporting to: {args.format}")

    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "simplify": True,
    }

    if args.format == "engine":
        export_args["device"] = args.device

    path = model.export(**export_args)
    print(f"Exported model saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument("--model", "-m", type=Path, required=True,
                        help="Path to trained model (.pt)")
    parser.add_argument("--format", "-f", type=str, default="onnx",
                        choices=["onnx", "engine", "torchscript", "coreml"],
                        help="Export format")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--half", action="store_true",
                        help="FP16 quantization")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic input shapes (ONNX)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device for TensorRT export")
    args = parser.parse_args()

    export_yolo(args)


if __name__ == "__main__":
    main()
