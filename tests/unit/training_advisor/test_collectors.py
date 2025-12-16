"""
Unit tests for training_advisor collectors module.

Tests for base collector, YOLO collector, and CSV collector.
"""

import pytest
from pathlib import Path
from typing import List
import tempfile
import csv


# ============================================
# BaseCollector Tests
# ============================================


class TestBaseCollector:
    """Tests for BaseCollector abstract base class."""

    def test_class_exists(self):
        """BaseCollector class exists."""
        from hunter.training_advisor.collectors.base import BaseCollector
        assert BaseCollector is not None

    def test_implements_protocol(self):
        """BaseCollector implements ICollector protocol."""
        from hunter.training_advisor.collectors.base import BaseCollector
        from hunter.training_advisor.interfaces import ICollector

        # Create concrete implementation for test
        class TestCollector(BaseCollector):
            @property
            def source_type(self) -> str:
                return "test"

            def _parse_source(self, source: Path):
                return []

        collector = TestCollector()
        assert isinstance(collector, ICollector)

    def test_collect_validates_path(self, tmp_path):
        """BaseCollector validates source path exists."""
        from hunter.training_advisor.collectors.base import BaseCollector

        class TestCollector(BaseCollector):
            @property
            def source_type(self) -> str:
                return "test"

            def _parse_source(self, source: Path):
                return []

        collector = TestCollector()
        non_existent = tmp_path / "non_existent.csv"

        with pytest.raises(FileNotFoundError):
            collector.collect(non_existent)


# ============================================
# YOLOCollector Tests
# ============================================


class TestYOLOCollector:
    """Tests for YOLOCollector."""

    def test_class_exists(self):
        """YOLOCollector class exists."""
        from hunter.training_advisor.collectors.yolo import YOLOCollector
        assert YOLOCollector is not None

    def test_source_type(self):
        """YOLOCollector has correct source type."""
        from hunter.training_advisor.collectors.yolo import YOLOCollector

        collector = YOLOCollector()
        assert collector.source_type == "yolo"

    def test_collect_from_results_csv(self, tmp_path):
        """YOLOCollector parses YOLO results.csv format."""
        from hunter.training_advisor.collectors.yolo import YOLOCollector
        from hunter.training_advisor.domain.metrics import TrainingMetrics

        # Create sample YOLO results.csv
        csv_file = tmp_path / "results.csv"
        csv_content = """epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
0,1.5,2.0,1.0,0.5,0.4,0.3,0.2,1.6,2.1,1.1,0.01,0.01,0.01
1,1.3,1.8,0.9,0.55,0.45,0.35,0.25,1.4,1.9,1.0,0.01,0.01,0.01
2,1.1,1.5,0.8,0.6,0.5,0.4,0.3,1.2,1.6,0.9,0.01,0.01,0.01
"""
        csv_file.write_text(csv_content)

        collector = YOLOCollector()
        metrics = collector.collect(csv_file)

        assert isinstance(metrics, TrainingMetrics)
        assert len(metrics.epochs) == 3
        assert metrics.epochs[0].epoch == 0
        assert metrics.epochs[2].map50 == pytest.approx(0.4)

    def test_collect_from_directory(self, tmp_path):
        """YOLOCollector can collect from training directory."""
        from hunter.training_advisor.collectors.yolo import YOLOCollector

        # Create directory structure like YOLO training output
        train_dir = tmp_path / "train"
        train_dir.mkdir()

        csv_file = train_dir / "results.csv"
        csv_content = """epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
0,1.5,2.0,1.0,0.5,0.4,0.3,0.2,1.6,2.1,1.1,0.01,0.01,0.01
"""
        csv_file.write_text(csv_content)

        collector = YOLOCollector()
        metrics = collector.collect(train_dir)

        assert len(metrics.epochs) == 1

    def test_extracts_learning_rate(self, tmp_path):
        """YOLOCollector extracts learning rate."""
        from hunter.training_advisor.collectors.yolo import YOLOCollector

        csv_file = tmp_path / "results.csv"
        csv_content = """epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
0,1.5,2.0,1.0,0.5,0.4,0.3,0.2,1.6,2.1,1.1,0.001,0.001,0.001
"""
        csv_file.write_text(csv_content)

        collector = YOLOCollector()
        metrics = collector.collect(csv_file)

        assert metrics.epochs[0].learning_rate == pytest.approx(0.001)

    def test_calculates_total_loss(self, tmp_path):
        """YOLOCollector calculates total train and val loss."""
        from hunter.training_advisor.collectors.yolo import YOLOCollector

        csv_file = tmp_path / "results.csv"
        csv_content = """epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
0,1.0,1.0,1.0,0.5,0.4,0.3,0.2,1.1,1.1,1.1,0.01,0.01,0.01
"""
        csv_file.write_text(csv_content)

        collector = YOLOCollector()
        metrics = collector.collect(csv_file)

        # train_loss should be sum of box + cls + dfl
        assert metrics.epochs[0].train_loss == pytest.approx(3.0)
        # val_loss should be sum of val box + cls + dfl
        assert metrics.epochs[0].val_loss == pytest.approx(3.3)


# ============================================
# CSVCollector Tests
# ============================================


class TestCSVCollector:
    """Tests for CSVCollector."""

    def test_class_exists(self):
        """CSVCollector class exists."""
        from hunter.training_advisor.collectors.csv_collector import CSVCollector
        assert CSVCollector is not None

    def test_source_type(self):
        """CSVCollector has correct source type."""
        from hunter.training_advisor.collectors.csv_collector import CSVCollector

        collector = CSVCollector()
        assert collector.source_type == "csv"

    def test_collect_basic_csv(self, tmp_path):
        """CSVCollector parses basic CSV format."""
        from hunter.training_advisor.collectors.csv_collector import CSVCollector
        from hunter.training_advisor.domain.metrics import TrainingMetrics

        csv_file = tmp_path / "training.csv"
        csv_content = """epoch,train_loss,val_loss
0,0.5,0.6
1,0.4,0.5
2,0.35,0.45
"""
        csv_file.write_text(csv_content)

        collector = CSVCollector()
        metrics = collector.collect(csv_file)

        assert isinstance(metrics, TrainingMetrics)
        assert len(metrics.epochs) == 3
        assert metrics.epochs[0].train_loss == pytest.approx(0.5)

    def test_collect_with_learning_rate(self, tmp_path):
        """CSVCollector parses CSV with learning rate column."""
        from hunter.training_advisor.collectors.csv_collector import CSVCollector

        csv_file = tmp_path / "training.csv"
        csv_content = """epoch,train_loss,val_loss,learning_rate
0,0.5,0.6,0.01
1,0.4,0.5,0.005
"""
        csv_file.write_text(csv_content)

        collector = CSVCollector()
        metrics = collector.collect(csv_file)

        assert metrics.epochs[0].learning_rate == pytest.approx(0.01)
        assert metrics.epochs[1].learning_rate == pytest.approx(0.005)

    def test_collect_with_custom_columns(self, tmp_path):
        """CSVCollector handles custom column names."""
        from hunter.training_advisor.collectors.csv_collector import CSVCollector

        csv_file = tmp_path / "training.csv"
        csv_content = """step,loss,validation_loss
0,0.5,0.6
1,0.4,0.5
"""
        csv_file.write_text(csv_content)

        collector = CSVCollector(
            epoch_col="step",
            train_loss_col="loss",
            val_loss_col="validation_loss",
        )
        metrics = collector.collect(csv_file)

        assert len(metrics.epochs) == 2
        assert metrics.epochs[0].train_loss == pytest.approx(0.5)


# ============================================
# StubCollector Tests
# ============================================


class TestStubCollector:
    """Tests for StubCollector (for testing)."""

    def test_class_exists(self):
        """StubCollector class exists."""
        from hunter.training_advisor.collectors.stub import StubCollector
        assert StubCollector is not None

    def test_source_type(self):
        """StubCollector has correct source type."""
        from hunter.training_advisor.collectors.stub import StubCollector

        collector = StubCollector()
        assert collector.source_type == "stub"

    def test_returns_configured_metrics(self):
        """StubCollector returns pre-configured metrics."""
        from hunter.training_advisor.collectors.stub import StubCollector
        from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics

        custom_metrics = TrainingMetrics(epochs=[
            EpochMetrics(epoch=0, train_loss=0.5, val_loss=0.6),
            EpochMetrics(epoch=1, train_loss=0.4, val_loss=0.5),
        ])

        collector = StubCollector(metrics=custom_metrics)
        result = collector.collect("any_path")

        assert result is custom_metrics
        assert len(result.epochs) == 2

    def test_generates_default_metrics(self):
        """StubCollector generates default metrics if none provided."""
        from hunter.training_advisor.collectors.stub import StubCollector

        collector = StubCollector(num_epochs=5)
        result = collector.collect("any_path")

        assert len(result.epochs) == 5


# ============================================
# Collector Factory Tests
# ============================================


class TestCollectorFactory:
    """Tests for collector factory function."""

    def test_factory_exists(self):
        """get_collector factory function exists."""
        from hunter.training_advisor.collectors import get_collector
        assert callable(get_collector)

    def test_get_yolo_collector(self):
        """Factory returns YOLO collector."""
        from hunter.training_advisor.collectors import get_collector
        from hunter.training_advisor.collectors.yolo import YOLOCollector

        collector = get_collector("yolo")
        assert isinstance(collector, YOLOCollector)

    def test_get_csv_collector(self):
        """Factory returns CSV collector."""
        from hunter.training_advisor.collectors import get_collector
        from hunter.training_advisor.collectors.csv_collector import CSVCollector

        collector = get_collector("csv")
        assert isinstance(collector, CSVCollector)

    def test_get_stub_collector(self):
        """Factory returns stub collector."""
        from hunter.training_advisor.collectors import get_collector
        from hunter.training_advisor.collectors.stub import StubCollector

        collector = get_collector("stub")
        assert isinstance(collector, StubCollector)

    def test_unknown_collector_raises(self):
        """Factory raises for unknown collector type."""
        from hunter.training_advisor.collectors import get_collector

        with pytest.raises(ValueError, match="Unknown collector"):
            get_collector("unknown_type")


# ============================================
# Collector Exports Tests
# ============================================


class TestCollectorExports:
    """Tests for collectors module exports."""

    def test_exports_base_collector(self):
        """BaseCollector is exported."""
        from hunter.training_advisor.collectors import BaseCollector
        assert BaseCollector is not None

    def test_exports_yolo_collector(self):
        """YOLOCollector is exported."""
        from hunter.training_advisor.collectors import YOLOCollector
        assert YOLOCollector is not None

    def test_exports_csv_collector(self):
        """CSVCollector is exported."""
        from hunter.training_advisor.collectors import CSVCollector
        assert CSVCollector is not None

    def test_exports_stub_collector(self):
        """StubCollector is exported."""
        from hunter.training_advisor.collectors import StubCollector
        assert StubCollector is not None

    def test_exports_get_collector(self):
        """get_collector is exported."""
        from hunter.training_advisor.collectors import get_collector
        assert callable(get_collector)
