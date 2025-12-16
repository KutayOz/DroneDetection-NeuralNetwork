#!/usr/bin/env python3
"""
Hunter Drone Detection & Tracking System - CLI Entry Point.

This module provides the main command-line interface for Hunter Drone.
Run with: python -m hunter [command] [options]

Commands:
    run         Run drone detection on video source
    train       Train a custom drone detection model
    evaluate    Evaluate model performance
    validate    Validate configuration file
    info        Show system information

Examples:
    python -m hunter run --config configs/default.yaml --source video.mp4
    python -m hunter validate configs/my_config.yaml
    python -m hunter info
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="hunter",
        description="""
╔═══════════════════════════════════════════════════════════════╗
║     HUNTER DRONE DETECTION & TRACKING SYSTEM                  ║
║     Real-time drone detection using YOLO11 + Siamese          ║
╚═══════════════════════════════════════════════════════════════╝
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --config configs/default.yaml --source video.mp4
  %(prog)s run --config configs/default.yaml --source rtsp://camera/stream
  %(prog)s validate configs/my_config.yaml
  %(prog)s info

Documentation:
  - User Guide:     docs/user-guide.md
  - Configuration:  docs/configuration.md
  - Training:       docs/training.md

For more help on a specific command:
  %(prog)s <command> --help
        """,
    )

    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show version and exit",
    )

    # Create subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available commands:",
        metavar="<command>",
    )

    # =========================================================================
    # RUN Command
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run drone detection on a video source",
        description="Run the drone detection pipeline on a video file or stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configs/default.yaml --source video.mp4
  %(prog)s --config configs/default.yaml --source 0  # webcam
  %(prog)s -c configs/default.yaml -s rtsp://192.168.1.100/stream
        """,
    )
    run_parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    run_parser.add_argument(
        "--source", "-s",
        type=str,
        help="Video source (file path, RTSP URL, or camera index)",
    )
    run_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results (JSONL format)",
    )
    run_parser.add_argument(
        "--device", "-d",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for inference",
    )
    run_parser.add_argument(
        "--confidence",
        type=float,
        help="Detection confidence threshold (0.0-1.0)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )
    run_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    # =========================================================================
    # TRAIN Command
    # =========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a custom drone detection model",
        description="Train or fine-tune a YOLO model for drone detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data datasets/drones --preset balanced
  %(prog)s --data datasets/drones --model yolo11m --epochs 100

For GUI-based training, use: hunter-studio
        """,
    )
    train_parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Path to dataset directory (YOLO format)",
    )
    train_parser.add_argument(
        "--preset", "-p",
        choices=["quick", "balanced", "accurate", "edge"],
        default="balanced",
        help="Training preset (default: balanced)",
    )
    train_parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11m",
        help="Base model to use (default: yolo11m)",
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("runs/train"),
        help="Output directory for trained model",
    )
    train_parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Enable automatic hyperparameter tuning",
    )

    # =========================================================================
    # EVALUATE Command
    # =========================================================================
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance",
        description="Evaluate a trained model on a test dataset.",
    )
    eval_parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to trained model (.pt file)",
    )
    eval_parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Path to test dataset",
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for evaluation report",
    )

    # =========================================================================
    # VALIDATE Command
    # =========================================================================
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
        description="Check if a configuration file is valid and show its settings.",
    )
    validate_parser.add_argument(
        "config_path",
        type=Path,
        help="Path to configuration file to validate",
    )
    validate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all configuration values",
    )

    # =========================================================================
    # INFO Command
    # =========================================================================
    subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display Hunter Drone version and system information.",
    )

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    from hunter.main import main as hunter_main

    # Build argv for the main function
    argv = ["--config", str(args.config)]

    if args.source:
        argv.extend(["--video", args.source])
    if args.output:
        argv.extend(["--output", str(args.output)])
    if args.device:
        argv.extend(["--device", args.device])
    if args.confidence:
        argv.extend(["--confidence", str(args.confidence)])
    if args.verbose:
        argv.append("--verbose")
    if args.quiet:
        argv.append("--quiet")

    return hunter_main(argv)


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    print("=" * 60)
    print("HUNTER DRONE - MODEL TRAINING")
    print("=" * 60)
    print()

    # Validate dataset path
    if not args.data.exists():
        print(f"Error: Dataset not found: {args.data}")
        print()
        print("Suggestions:")
        print("  1. Check that the path is correct")
        print("  2. Ensure dataset follows YOLO format:")
        print("     datasets/")
        print("     └── drones/")
        print("         ├── images/")
        print("         │   ├── train/")
        print("         │   └── val/")
        print("         ├── labels/")
        print("         │   ├── train/")
        print("         │   └── val/")
        print("         └── data.yaml")
        return 1

    # Training presets
    presets = {
        "quick": {"model": "yolo11n", "epochs": 30, "imgsz": 416},
        "balanced": {"model": "yolo11m", "epochs": 100, "imgsz": 640},
        "accurate": {"model": "yolo11x", "epochs": 300, "imgsz": 1280},
        "edge": {"model": "yolo11n", "epochs": 50, "imgsz": 320},
    }

    preset = presets.get(args.preset, presets["balanced"])
    model = args.model or preset["model"]
    epochs = args.epochs or preset["epochs"]

    print(f"Dataset:  {args.data}")
    print(f"Preset:   {args.preset}")
    print(f"Model:    {model}")
    print(f"Epochs:   {epochs}")
    print(f"Output:   {args.output_dir}")
    print()

    try:
        from ultralytics import YOLO

        print("Loading model...")
        yolo = YOLO(f"{model}.pt")

        print("Starting training...")
        print("(Press Ctrl+C to stop)")
        print()

        results = yolo.train(
            data=str(args.data / "data.yaml"),
            epochs=epochs,
            imgsz=preset["imgsz"],
            project=str(args.output_dir),
            name="drone_detector",
            exist_ok=True,
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best model saved to: {args.output_dir}/drone_detector/weights/best.pt")
        print()
        print("Next steps:")
        print("  1. Test the model:")
        print(f"     hunter run --config configs/default.yaml --source test.mp4")
        print("  2. Evaluate performance:")
        print(f"     hunter evaluate --model {args.output_dir}/drone_detector/weights/best.pt --data {args.data}")

        return 0

    except ImportError:
        print("Error: Ultralytics package not installed.")
        print()
        print("Install with: pip install ultralytics")
        return 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate command."""
    print("=" * 60)
    print("HUNTER DRONE - MODEL EVALUATION")
    print("=" * 60)
    print()

    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        print()
        print("Suggestions:")
        print("  1. Check that the model path is correct")
        print("  2. Train a model first: hunter train --data datasets/drones")
        return 1

    if not args.data.exists():
        print(f"Error: Dataset not found: {args.data}")
        return 1

    try:
        from ultralytics import YOLO

        print(f"Model:   {args.model}")
        print(f"Dataset: {args.data}")
        print()

        model = YOLO(str(args.model))
        metrics = model.val(data=str(args.data / "data.yaml"))

        print()
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"mAP50:      {metrics.box.map50:.4f}")
        print(f"mAP50-95:   {metrics.box.map:.4f}")
        print(f"Precision:  {metrics.box.mp:.4f}")
        print(f"Recall:     {metrics.box.mr:.4f}")

        return 0

    except ImportError:
        print("Error: Ultralytics package not installed.")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    from hunter import HunterConfig, ConfigError

    print("=" * 60)
    print("HUNTER DRONE - CONFIG VALIDATION")
    print("=" * 60)
    print()

    config_path = args.config_path

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print()
        print("Suggestions:")
        print("  1. Check that the path is correct")
        print("  2. Copy the default config:")
        print(f"     cp configs/default.yaml {config_path}")
        return 1

    try:
        config = HunterConfig.from_yaml(config_path)

        print(f"Config file: {config_path}")
        print(f"Status: VALID")
        print()

        if args.verbose:
            print("Configuration Summary:")
            print("-" * 40)
            print(f"  Ingest:")
            print(f"    Source type: {config.ingest.source_type}")
            print(f"    Source URI:  {config.ingest.source_uri or '(not set)'}")
            print()
            print(f"  Detector:")
            print(f"    Model:      {config.detector.model_path}")
            print(f"    Device:     {config.detector.device}")
            print(f"    Confidence: {config.detector.confidence_threshold}")
            print()
            print(f"  Tracking:")
            print(f"    Lock frames:    {config.tracking.lock_confirm_frames}")
            print(f"    Lost timeout:   {config.tracking.lost_timeout_frames}")
            print(f"    IoU threshold:  {config.tracking.iou_threshold}")
            print()
            print(f"  Output:")
            print(f"    Sink type: {config.output.sink_type}")
            print(f"    Path:      {config.output.output_path or '(not set)'}")

        return 0

    except ConfigError as e:
        print(f"Error: Invalid configuration")
        print(f"  {e}")
        print()
        print("Suggestions:")
        print("  1. Check YAML syntax (indentation, colons)")
        print("  2. Verify all required fields are present")
        print("  3. Check field value ranges (e.g., confidence 0.0-1.0)")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    import hunter

    hunter.print_info()

    print()
    print("-" * 40)

    # Check dependencies
    print("Dependencies:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  PyTorch:  {torch.__version__}")
        print(f"  CUDA:     {'Available' if cuda_available else 'Not available'}")
        if cuda_available:
            print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch:  Not installed")

    try:
        import ultralytics
        print(f"  YOLO:     {ultralytics.__version__}")
    except ImportError:
        print("  YOLO:     Not installed")

    try:
        import filterpy
        print(f"  FilterPy: Installed")
    except ImportError:
        print("  FilterPy: Not installed")

    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point for Hunter Drone CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle version flag
    if args.version:
        from hunter import __version__
        print(f"Hunter Drone v{__version__}")
        return 0

    # Handle no command
    if not args.command:
        parser.print_help()
        print()
        print("Tip: Use 'hunter <command> --help' for help on a specific command.")
        return 0

    # Dispatch to command handler
    commands = {
        "run": cmd_run,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "validate": cmd_validate,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
