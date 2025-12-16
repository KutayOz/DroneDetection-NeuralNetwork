#!/usr/bin/env python3
"""
Run inference on video file or camera stream.

Usage:
    python scripts/run_inference.py --config configs/default.yaml --video input.mp4
    python scripts/run_inference.py --config configs/default.yaml --video input.mp4 --output results.jsonl
    python scripts/run_inference.py --config configs/profiles/low_latency.yaml --video input.mp4
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hunter import Pipeline, HunterConfig


def main():
    parser = argparse.ArgumentParser(description="Run Hunter Drone inference")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Config file")
    parser.add_argument("--video", "-v", type=Path, required=True, help="Input video")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--model", "-m", type=Path, help="Override model path")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--confidence", type=float, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show visualization")
    args = parser.parse_args()

    # Load config
    config = HunterConfig.from_yaml(args.config)

    # Apply overrides
    config.ingest.source_uri = str(args.video)
    config.ingest.source_type = "file"

    if args.output:
        config.output.output_path = args.output
        config.output.sink_type = "json"

    if args.model:
        config.detector.model_path = args.model

    if args.device:
        config.detector.device = args.device
        config.embedder.device = args.device

    if args.confidence:
        config.detector.confidence_threshold = args.confidence

    # Run pipeline
    print(f"Processing: {args.video}")
    print(f"Model: {config.detector.model_path}")
    print(f"Device: {config.detector.device}")
    print()

    with Pipeline(config) as pipeline:
        for message in pipeline.run():
            if message.frame_id % 30 == 0:
                print(
                    f"\rFrame {message.frame_id:6d} | "
                    f"FPS: {pipeline.fps:5.1f} | "
                    f"Tracks: {message.track_count:3d} | "
                    f"Latency: {message.pipeline_metrics['total_e2e_ms']:6.1f}ms",
                    end="",
                    flush=True,
                )

    print("\n\nDone!")
    metrics = pipeline.get_metrics()
    print(f"Processed {metrics['frames_processed']} frames in {metrics['runtime_seconds']:.1f}s")
    print(f"Average FPS: {metrics['frames_processed'] / max(metrics['runtime_seconds'], 1):.1f}")


if __name__ == "__main__":
    main()
