"""
Main pipeline orchestrator.

Coordinates all pipeline components:
- Video ingest
- Preprocessing
- Detection (YOLO11)
- Embedding (Siamese)
- Tracking (Kalman + Hungarian)
- Output

Follows Facade pattern: Simple interface to complex subsystem.
Follows DIP: Depends on abstractions, not concrete implementations.
"""

import time
from typing import Iterator, List, Optional

import numpy as np

from ..core.config import HunterConfig
from ..core.logger import PipelineLogger, setup_logger
from ..core.timer import PipelineTimer
from ..core.metrics import MetricsCollector
from ..core.exceptions import HunterError

from .ingest.base import BaseIngest
from .ingest.file_ingest import FileIngest, StubIngest
from .ingest.frame_packet import FramePacket
from .preprocess.transforms import Preprocessor, crop_bbox
from .output.base import BaseOutput
from .output.stub_sink import StubSink
from .output.track_message import (
    TrackMessage,
    TrackInfo,
    ModelInfo,
    TrajectoryPoint,
)

from ..models.detector.base import BaseDetector
from ..models.detector.yolo_detector import YOLODetector, StubDetector
from ..models.embedder.base import BaseEmbedder
from ..models.embedder.siamese_embedder import SiameseEmbedder

from ..tracking.tracker import MultiTargetTracker
from ..tracking.association import Detection


class Pipeline:
    """
    Main pipeline orchestrator.

    Processes video frames through the complete detection and tracking pipeline.

    Pipeline Flow:
        Frame → Decode → Preprocess → Detect → Embed → Track → Output

    Example usage:
        config = HunterConfig.from_yaml("config.yaml")
        pipeline = Pipeline(config)

        for message in pipeline.run():
            print(f"Frame {message.frame_id}: {len(message.tracks)} tracks")

        pipeline.close()
    """

    def __init__(self, config: HunterConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration

        Raises:
            HunterError: On initialization failure
        """
        self._config = config
        self._start_time: Optional[float] = None
        self._frame_count = 0

        # Setup logging
        base_logger = setup_logger(config.logging)
        self._logger = PipelineLogger(base_logger, config.logging)

        # Setup timing and metrics
        self._timer = PipelineTimer()
        self._metrics = MetricsCollector()

        # Initialize components
        self._ingest = self._create_ingest()
        self._preprocessor = Preprocessor(config.preprocess)
        self._detector = self._create_detector()
        self._embedder = self._create_embedder()
        self._tracker = MultiTargetTracker(
            config.tracking,
            logger=self._logger,
            fps=self._ingest.fps,
        )
        self._output = self._create_output()

        # Model info for output messages
        self._model_info = ModelInfo(
            detector_name=self._detector.name,
            detector_hash=self._detector.hash[:16],
            embedder_name=self._embedder.name if self._embedder else None,
            embedder_hash=self._embedder.hash[:16] if self._embedder else None,
        )

        # Warm up models
        self._warmup()

        self._logger.pipeline_started(config.model_dump_safe())

    def _create_ingest(self) -> BaseIngest:
        """Create video ingest based on config."""
        source_type = self._config.ingest.source_type

        if source_type == "file":
            return FileIngest(self._config.ingest)
        elif source_type == "stub":
            return StubIngest()
        else:
            raise HunterError(f"Unknown ingest type: {source_type}")

    def _create_detector(self) -> BaseDetector:
        """Create detector based on config."""
        model_type = self._config.detector.model_type

        if model_type == "yolo":
            return YOLODetector(self._config.detector)
        else:
            raise HunterError(f"Unknown detector type: {model_type}")

    def _create_embedder(self) -> Optional[BaseEmbedder]:
        """Create embedder if enabled."""
        if not self._config.embedder.enabled:
            return None

        if self._config.embedder.model_path is None:
            return None

        return SiameseEmbedder(self._config.embedder)

    def _create_output(self) -> BaseOutput:
        """Create output sink based on config."""
        return StubSink(self._config.output)

    def _warmup(self) -> None:
        """Warm up models."""
        self._logger.info("Warming up models...")
        self._detector.warmup()

        if self._embedder:
            self._embedder.warmup()

    def run(self) -> Iterator[TrackMessage]:
        """
        Run pipeline and yield track messages.

        Yields:
            TrackMessage for each processed frame

        Example:
            for message in pipeline.run():
                print(message.track_count)
        """
        self._start_time = time.time()

        for frame in self._ingest:
            try:
                message = self.process_frame(frame)
                if message:
                    yield message
            except Exception as e:
                self._logger.error(f"Frame processing failed: {e}", frame_id=frame.frame_id)
                continue

    def process_frame(self, frame: FramePacket) -> Optional[TrackMessage]:
        """
        Process single frame through pipeline.

        Args:
            frame: Input frame packet

        Returns:
            TrackMessage or None on error
        """
        self._timer.reset()
        timestamp_ms = frame.effective_timestamp

        # ─────────────────────────────────────────────────────
        # 1. PREPROCESS (not timed separately for YOLO)
        # ─────────────────────────────────────────────────────
        with self._timer.measure("preprocess"):
            # YOLO handles preprocessing internally
            # Just convert to RGB if needed
            image = frame.image
            if frame.pixel_format == "BGR":
                image = self._preprocessor.process_for_yolo(image, "BGR")

        # ─────────────────────────────────────────────────────
        # 2. DETECT
        # ─────────────────────────────────────────────────────
        with self._timer.measure("detect"):
            det_outputs = self._detector.detect(image)

        # ─────────────────────────────────────────────────────
        # 3. EMBED (if enabled)
        # ─────────────────────────────────────────────────────
        embeddings: List[Optional[np.ndarray]] = [None] * len(det_outputs)

        if self._embedder and det_outputs:
            with self._timer.measure("embed"):
                # Crop detections from original frame
                crops = []
                for det in det_outputs:
                    crop = crop_bbox(frame.image, det.bbox_xyxy, padding=0.1)
                    crops.append(crop)

                embeddings = self._embedder.embed_batch(crops)

        # ─────────────────────────────────────────────────────
        # 4. CREATE DETECTION OBJECTS
        # ─────────────────────────────────────────────────────
        detections = []
        for i, det in enumerate(det_outputs):
            detections.append(
                Detection(
                    bbox_xyxy=det.bbox_xyxy,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    embedding=embeddings[i],
                )
            )

        # ─────────────────────────────────────────────────────
        # 5. TRACK
        # ─────────────────────────────────────────────────────
        with self._timer.measure("associate"):
            tracks = self._tracker.update(detections, timestamp_ms)

        # ─────────────────────────────────────────────────────
        # 6. CREATE OUTPUT MESSAGE
        # ─────────────────────────────────────────────────────
        with self._timer.measure("output"):
            message = self._create_track_message(frame, tracks)
            self._output.write(message)

        # Update metrics
        timings = self._timer.get_dict()
        self._metrics.latency.add_sample(timings["total_e2e_ms"])
        self._metrics.throughput.tick()
        self._frame_count += 1

        # Log frame processed
        self._logger.frame_processed(
            frame.frame_id,
            timings,
            len(self._tracker.get_visible_tracks()),
            len(det_outputs),
        )

        return message

    def _create_track_message(
        self,
        frame: FramePacket,
        tracks: list,
    ) -> TrackMessage:
        """Create TrackMessage from current state."""
        visible_tracks = self._tracker.get_visible_tracks()

        track_infos = []
        for track in visible_tracks:
            # Get trajectory tail
            trajectory_tail = [
                TrajectoryPoint(t_ms=p["t_ms"], cx=p["cx"], cy=p["cy"])
                for p in track.get_trajectory_tail(self._config.tracking.trajectory_output_points)
            ]

            track_info = TrackInfo(
                track_id=track.track_id,
                state=track.state_name,
                confidence=track.confidence,
                bbox_xyxy=track.bbox_int,
                predicted_bbox_xyxy=tuple(int(x) for x in track.predicted_bbox),
                velocity_px_per_s=track.velocity_per_second,
                trajectory_tail=trajectory_tail,
                age_frames=track.age_frames,
                hits=track.hits,
                time_since_update=track.time_since_update,
            )
            track_infos.append(track_info)

        return TrackMessage.create(
            frame_id=frame.frame_id,
            model=self._model_info,
            pipeline_metrics=self._timer.get_dict(),
            tracks=track_infos,
        )

    def close(self) -> None:
        """Close pipeline and release resources."""
        runtime = time.time() - self._start_time if self._start_time else 0

        self._logger.pipeline_stopped(self._frame_count, runtime)

        self._ingest.close()
        self._output.close()

    def get_metrics(self) -> dict:
        """Get pipeline metrics."""
        return {
            "frames_processed": self._frame_count,
            "runtime_seconds": time.time() - self._start_time if self._start_time else 0,
            **self._metrics.to_dict(),
            "tracker": self._tracker.get_stats(),
        }

    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._metrics.throughput.fps

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count

    def __enter__(self) -> "Pipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"Pipeline(frames={self._frame_count}, "
            f"fps={self.fps:.1f}, "
            f"tracks={self._tracker.visible_track_count})"
        )
