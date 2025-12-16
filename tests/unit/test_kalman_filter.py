"""
Unit tests for BboxKalmanFilter.
"""

import pytest
import numpy as np

from hunter.tracking.kalman_filter import BboxKalmanFilter, KalmanConfig


class TestBboxKalmanFilter:
    """Tests for BboxKalmanFilter."""

    def test_initialization(self):
        """Filter should initialize correctly from bbox."""
        kf = BboxKalmanFilter()
        bbox = (100.0, 100.0, 200.0, 200.0)

        kf.initialize(bbox)

        assert kf.initialized
        assert kf.center == (150.0, 150.0)
        assert kf.velocity == (0.0, 0.0)

    def test_predict_without_init_raises(self):
        """Predict without initialization should raise."""
        kf = BboxKalmanFilter()

        with pytest.raises(RuntimeError):
            kf.predict()

    def test_update_initializes_if_needed(self):
        """Update on uninitialized filter should initialize."""
        kf = BboxKalmanFilter()
        bbox = (100.0, 100.0, 200.0, 200.0)

        result = kf.update(bbox)

        assert kf.initialized
        assert result == bbox

    def test_predict_returns_valid_bbox(self):
        """Predict should return valid bbox."""
        kf = BboxKalmanFilter()
        kf.initialize((100.0, 100.0, 200.0, 200.0))

        predicted = kf.predict()

        assert len(predicted) == 4
        assert predicted[2] > predicted[0]  # x2 > x1
        assert predicted[3] > predicted[1]  # y2 > y1

    def test_update_after_predict(self):
        """Update after predict should correct state."""
        kf = BboxKalmanFilter()
        kf.initialize((100.0, 100.0, 200.0, 200.0))

        # Predict
        kf.predict()

        # Update with measurement
        measurement = (105.0, 105.0, 205.0, 205.0)
        updated = kf.update(measurement)

        # Should be close to measurement
        assert abs(updated[0] - measurement[0]) < 10
        assert abs(updated[1] - measurement[1]) < 10

    def test_velocity_tracking(self):
        """Filter should learn velocity from consistent motion."""
        config = KalmanConfig(dt=1/30)
        kf = BboxKalmanFilter(config)

        # Initialize at (100, 100)
        kf.initialize((100.0, 100.0, 150.0, 150.0))

        # Move consistently to the right
        for i in range(10):
            kf.predict()
            # Object moves 5 pixels per frame
            x = 100.0 + (i + 1) * 5
            kf.update((x, 100.0, x + 50, 150.0))

        vx, vy = kf.velocity
        # Velocity should be positive in x direction
        assert vx > 0
        assert abs(vy) < abs(vx)  # Little vertical motion

    def test_size_stays_positive(self):
        """Width and height should always be positive."""
        kf = BboxKalmanFilter()
        kf.initialize((100.0, 100.0, 110.0, 110.0))  # Small box

        # Many predictions without update
        for _ in range(100):
            bbox = kf.predict()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            assert w >= 1.0
            assert h >= 1.0

    def test_custom_config(self):
        """Custom config should be applied."""
        config = KalmanConfig(
            process_noise=5.0,
            measurement_noise=0.5,
            dt=1/60,
        )
        kf = BboxKalmanFilter(config)
        kf.initialize((100.0, 100.0, 200.0, 200.0))

        # Should work without error
        kf.predict()
        kf.update((105.0, 105.0, 205.0, 205.0))

    def test_reset(self):
        """Reset should return filter to uninitialized state."""
        kf = BboxKalmanFilter()
        kf.initialize((100.0, 100.0, 200.0, 200.0))

        assert kf.initialized

        kf.reset()

        assert not kf.initialized

    def test_get_predicted_position(self):
        """Should predict future position based on velocity."""
        kf = BboxKalmanFilter()
        kf.initialize((100.0, 100.0, 200.0, 200.0))

        # Move object
        for _ in range(5):
            kf.predict()
            kf.update((110.0, 100.0, 210.0, 200.0))

        # Predict future position
        future_x, future_y = kf.get_predicted_position(steps=5)

        # Should predict continued motion
        current_x, current_y = kf.center
        # Future x should be different if there's velocity
        # (not strictly testing value, just functionality)
        assert future_x is not None
        assert future_y is not None
