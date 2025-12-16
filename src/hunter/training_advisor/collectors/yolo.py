"""
YOLO training metrics collector.

Parses YOLO training output (results.csv) format.
"""

import csv
from pathlib import Path
from typing import List, Optional

from .base import BaseCollector
from ..domain.metrics import EpochMetrics


class YOLOCollector(BaseCollector):
    """
    Collector for YOLO training results.

    Parses the results.csv file produced by Ultralytics YOLO training.
    Expected columns:
        - epoch
        - train/box_loss, train/cls_loss, train/dfl_loss
        - val/box_loss, val/cls_loss, val/dfl_loss
        - metrics/precision(B), metrics/recall(B)
        - metrics/mAP50(B), metrics/mAP50-95(B)
        - lr/pg0, lr/pg1, lr/pg2
    """

    def __init__(self, results_filename: str = "results.csv"):
        """
        Initialize YOLO collector.

        Args:
            results_filename: Name of results CSV file
        """
        self._results_filename = results_filename

    @property
    def source_type(self) -> str:
        """Source type identifier."""
        return "yolo"

    def _parse_source(self, source: Path) -> List[EpochMetrics]:
        """
        Parse YOLO results file.

        Args:
            source: Path to results.csv or directory containing it

        Returns:
            List of epoch metrics
        """
        # Find results file
        if source.is_dir():
            results_file = source / self._results_filename
            if not results_file.exists():
                # Try common YOLO directory structures
                for subdir in ["", "train", "exp"]:
                    candidate = source / subdir / self._results_filename if subdir else source / self._results_filename
                    if candidate.exists():
                        results_file = candidate
                        break
        else:
            results_file = source

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        return self._parse_csv(results_file)

    def _parse_csv(self, csv_path: Path) -> List[EpochMetrics]:
        """
        Parse YOLO results CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of epoch metrics
        """
        epochs = []

        with open(csv_path, "r") as f:
            # Handle potential whitespace in headers
            reader = csv.DictReader(f)
            # Clean up field names (YOLO CSV has spaces after commas)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                # Clean up row keys
                row = {k.strip(): v.strip() for k, v in row.items()}

                epoch_metrics = self._parse_row(row)
                if epoch_metrics:
                    epochs.append(epoch_metrics)

        return epochs

    def _parse_row(self, row: dict) -> Optional[EpochMetrics]:
        """
        Parse a single CSV row into EpochMetrics.

        Args:
            row: Dictionary of column values

        Returns:
            EpochMetrics or None if parsing fails
        """
        try:
            epoch = int(row.get("epoch", 0))

            # Calculate total train loss (sum of component losses)
            box_loss = float(row.get("train/box_loss", 0))
            cls_loss = float(row.get("train/cls_loss", 0))
            dfl_loss = float(row.get("train/dfl_loss", 0))
            train_loss = box_loss + cls_loss + dfl_loss

            # Calculate total val loss
            val_box_loss = float(row.get("val/box_loss", 0))
            val_cls_loss = float(row.get("val/cls_loss", 0))
            val_dfl_loss = float(row.get("val/dfl_loss", 0))
            val_loss = val_box_loss + val_cls_loss + val_dfl_loss

            # Extract metrics
            precision = self._safe_float(row.get("metrics/precision(B)"))
            recall = self._safe_float(row.get("metrics/recall(B)"))
            map50 = self._safe_float(row.get("metrics/mAP50(B)"))
            map50_95 = self._safe_float(row.get("metrics/mAP50-95(B)"))

            # Learning rate (use pg0 as primary)
            lr = self._safe_float(row.get("lr/pg0"))

            return EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=lr,
                map50=map50,
                map50_95=map50_95,
                precision=precision,
                recall=recall,
                box_loss=box_loss,
                cls_loss=cls_loss,
                dfl_loss=dfl_loss,
            )

        except (ValueError, KeyError) as e:
            # Skip malformed rows
            return None

    def _safe_float(self, value: Optional[str]) -> Optional[float]:
        """
        Safely convert string to float.

        Args:
            value: String value or None

        Returns:
            Float value or None
        """
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None
