"""
Generic CSV training metrics collector.

Parses standard CSV training logs with configurable column names.
"""

import csv
from pathlib import Path
from typing import List, Optional

from .base import BaseCollector
from ..domain.metrics import EpochMetrics


class CSVCollector(BaseCollector):
    """
    Generic CSV collector for training metrics.

    Handles standard CSV format with configurable column names.
    Default expected columns: epoch, train_loss, val_loss, learning_rate
    """

    def __init__(
        self,
        epoch_col: str = "epoch",
        train_loss_col: str = "train_loss",
        val_loss_col: str = "val_loss",
        lr_col: str = "learning_rate",
        delimiter: str = ",",
    ):
        """
        Initialize CSV collector.

        Args:
            epoch_col: Column name for epoch number
            train_loss_col: Column name for training loss
            val_loss_col: Column name for validation loss
            lr_col: Column name for learning rate
            delimiter: CSV delimiter character
        """
        self._epoch_col = epoch_col
        self._train_loss_col = train_loss_col
        self._val_loss_col = val_loss_col
        self._lr_col = lr_col
        self._delimiter = delimiter

    @property
    def source_type(self) -> str:
        """Source type identifier."""
        return "csv"

    def _parse_source(self, source: Path) -> List[EpochMetrics]:
        """
        Parse CSV file.

        Args:
            source: Path to CSV file

        Returns:
            List of epoch metrics
        """
        epochs = []

        with open(source, "r") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            # Clean up field names
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                epoch_metrics = self._parse_row(row)
                if epoch_metrics:
                    epochs.append(epoch_metrics)

        return epochs

    def _parse_row(self, row: dict) -> Optional[EpochMetrics]:
        """
        Parse a single CSV row.

        Args:
            row: Dictionary of column values

        Returns:
            EpochMetrics or None if parsing fails
        """
        try:
            epoch = int(row.get(self._epoch_col, 0))
            train_loss = float(row.get(self._train_loss_col, 0))
            val_loss = float(row.get(self._val_loss_col, 0))

            # Optional learning rate
            lr_value = row.get(self._lr_col)
            lr = float(lr_value) if lr_value and lr_value.strip() else None

            return EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=lr,
            )

        except (ValueError, KeyError):
            return None
