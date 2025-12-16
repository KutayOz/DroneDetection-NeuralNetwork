"""
Hunter Drone Detection & Tracking System - Main Entry Point.

Usage:
    hunter-run --config config.yaml
    hunter-run --config config.yaml --video video.mp4
    hunter-run --help

Or run directly:
    python -m hunter.main --config config.yaml
"""

import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

from .core.config import HunterConfig
from .core.exceptions import HunterError
from .pipeline.orchestrator import Pipeline


# Global pipeline reference for signal handling
_pipeline: Optional[Pipeline] = None


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    print("\nShutting down...")
    if _pipeline:
        _pipeline.close()
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hunter Drone Detection & Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  hunter-run --config configs/default.yaml

  # Override video source
  hunter-run --config configs/default.yaml --video test.mp4

  # Run with different model
  hunter-run --config configs/default.yaml --model yolo11m.pt
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--video", "-v",
        type=Path,
        help="Override video source path",
    )

    parser.add_argument(
        "--model", "-m",
        type=Path,
        help="Override detector model path",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path for JSON results",
    )

    parser.add_argument(
        "--device", "-d",
        choices=["cuda", "cpu", "mps"],
        help="Override compute device",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        help="Override detection confidence threshold",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Hunter Drone v1.0.0",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> HunterConfig:
    """
    Load and override configuration.

    Args:
        args: Parsed command line arguments

    Returns:
        Final configuration with any overrides applied
    """
    # Load base config
    config = HunterConfig.from_yaml(args.config)

    # Apply overrides
    if args.video:
        config.ingest.source_uri = str(args.video)
        config.ingest.source_type = "file"

    if args.model:
        config.detector.model_path = args.model

    if args.output:
        config.output.output_path = args.output
        config.output.sink_type = "json"

    if args.device:
        config.detector.device = args.device
        config.embedder.device = args.device

    if args.confidence:
        config.detector.confidence_threshold = args.confidence

    if args.verbose:
        config.logging.level = "DEBUG"
        config.logging.format = "console"

    if args.quiet:
        config.logging.log_stage_timings = False
        config.logging.log_track_changes = False

    return config


def run_pipeline(config: HunterConfig) -> None:
    """
    Run the detection and tracking pipeline.

    Args:
        config: Pipeline configuration
    """
    global _pipeline

    print(f"Starting Hunter Drone Detection System...")
    print(f"  Detector: {config.detector.model_path}")
    print(f"  Source: {config.ingest.source_uri}")
    print(f"  Device: {config.detector.device}")
    print()

    with Pipeline(config) as pipeline:
        _pipeline = pipeline

        for message in pipeline.run():
            # Print progress
            if message.frame_id % 30 == 0:
                metrics = pipeline.get_metrics()
                print(
                    f"\rFrame {message.frame_id:6d} | "
                    f"FPS: {metrics['throughput']['fps']:5.1f} | "
                    f"Tracks: {message.track_count:3d} | "
                    f"Latency: {message.pipeline_metrics['total_e2e_ms']:6.1f}ms",
                    end="",
                    flush=True,
                )

        print()  # New line after progress

    # Print final statistics
    metrics = pipeline.get_metrics()
    print("\n" + "=" * 50)
    print("Pipeline Statistics")
    print("=" * 50)
    print(f"  Total frames:     {metrics['frames_processed']}")
    print(f"  Runtime:          {metrics['runtime_seconds']:.1f}s")
    print(f"  Average FPS:      {metrics['frames_processed'] / max(metrics['runtime_seconds'], 1):.1f}")
    print(f"  Latency p50:      {metrics['latency']['p50_ms']:.1f}ms")
    print(f"  Latency p95:      {metrics['latency']['p95_ms']:.1f}ms")
    print(f"  Total tracks:     {metrics['tracker']['next_track_id']}")


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 = success)
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Parse arguments
        args = parse_args()

        # Load configuration
        config = load_config(args)

        # Run pipeline
        run_pipeline(config)

        return 0

    except HunterError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
