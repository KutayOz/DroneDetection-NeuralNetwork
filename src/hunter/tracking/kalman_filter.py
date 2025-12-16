"""
Kalman filter for bounding box tracking.

Uses constant velocity motion model:
- State: [cx, cy, w, h, vx, vy, vw, vh]
- Measurement: [cx, cy, w, h]

Follows SRP: Only responsible for motion prediction.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KalmanConfig:
    """Kalman filter configuration."""

    process_noise: float = 1.0
    measurement_noise: float = 1.0
    dt: float = 1 / 30  # Time step (1/fps)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.process_noise <= 0:
            raise ValueError("process_noise must be positive")
        if self.measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")


class BboxKalmanFilter:
    """
    Kalman filter for bounding box tracking.

    State Vector (8D):
        [cx, cy, w, h, vx, vy, vw, vh]

        - cx, cy: center position
        - w, h: width, height
        - vx, vy: velocity of center
        - vw, vh: rate of change of size

    Measurement Vector (4D):
        [cx, cy, w, h]

    Uses constant velocity motion model.
    """

    DIM_STATE = 8
    DIM_MEASUREMENT = 4

    def __init__(self, config: Optional[KalmanConfig] = None):
        """
        Initialize Kalman filter.

        Args:
            config: Configuration (uses defaults if None)
        """
        self._config = config or KalmanConfig()
        dt = self._config.dt

        # State transition matrix F (8x8)
        # Position updates with velocity: x_new = x_old + v * dt
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = dt  # cx += vx * dt
        self.F[1, 5] = dt  # cy += vy * dt
        self.F[2, 6] = dt  # w += vw * dt
        self.F[3, 7] = dt  # h += vh * dt

        # Measurement matrix H (4x8)
        # We observe [cx, cy, w, h]
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1  # cx
        self.H[1, 1] = 1  # cy
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

        # Process noise covariance Q (8x8)
        q = self._config.process_noise
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= q
        self.Q[4:, 4:] *= q * 0.1  # Velocity has less process noise

        # Measurement noise covariance R (4x4)
        r = self._config.measurement_noise
        self.R = np.eye(4, dtype=np.float32) * r

        # State vector and covariance (initialized on first measurement)
        self.x: Optional[np.ndarray] = None  # Shape: (8,)
        self.P: Optional[np.ndarray] = None  # Shape: (8, 8)

        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Whether filter has been initialized."""
        return self._initialized

    def initialize(self, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        """
        Initialize with first measurement.

        Args:
            bbox_xyxy: Bounding box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Convert to center format
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Initial state: position known, velocity zero
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        # Initial covariance: moderate position uncertainty, high velocity uncertainty
        self.P = np.eye(8, dtype=np.float32)
        self.P[:4, :4] *= 10  # Position uncertainty
        self.P[4:, 4:] *= 100  # Velocity uncertainty (unknown initially)

        self._initialized = True

    def predict(self) -> Tuple[float, float, float, float]:
        """
        Predict next state.

        Runs the prediction step of the Kalman filter.

        Returns:
            Predicted bbox (x1, y1, x2, y2)

        Raises:
            RuntimeError: If filter not initialized
        """
        if not self._initialized:
            raise RuntimeError("Kalman filter not initialized")

        # State prediction: x = F @ x
        self.x = self.F @ self.x

        # Covariance prediction: P = F @ P @ F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self._state_to_xyxy()

    def update(
        self, bbox_xyxy: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Update state with measurement.

        Runs the update step of the Kalman filter.

        Args:
            bbox_xyxy: Measured bbox (x1, y1, x2, y2)

        Returns:
            Updated bbox (x1, y1, x2, y2)
        """
        if not self._initialized:
            self.initialize(bbox_xyxy)
            return bbox_xyxy

        # Convert measurement to center format
        x1, y1, x2, y2 = bbox_xyxy
        z = np.array(
            [
                (x1 + x2) / 2,  # cx
                (y1 + y2) / 2,  # cy
                x2 - x1,  # w
                y2 - y1,  # h
            ],
            dtype=np.float32,
        )

        # Innovation (measurement residual): y = z - H @ x
        y = z - self.H @ self.x

        # Innovation covariance: S = H @ P @ H.T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P @ H.T @ S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K @ y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K @ H) @ P
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self._state_to_xyxy()

    def _state_to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert state vector to xyxy bbox format."""
        cx, cy, w, h = self.x[:4]

        # Ensure positive dimensions
        w = max(w, 1.0)
        h = max(h, 1.0)

        return (
            float(cx - w / 2),
            float(cy - h / 2),
            float(cx + w / 2),
            float(cy + h / 2),
        )

    @property
    def velocity(self) -> Tuple[float, float]:
        """
        Get velocity (vx, vy) in pixels per time step.

        Returns:
            Tuple of (vx, vy)
        """
        if not self._initialized:
            return (0.0, 0.0)
        return (float(self.x[4]), float(self.x[5]))

    @property
    def velocity_per_second(self) -> Tuple[float, float]:
        """
        Get velocity in pixels per second.

        Returns:
            Tuple of (vx, vy) per second
        """
        vx, vy = self.velocity
        fps = 1.0 / self._config.dt
        return (vx * fps, vy * fps)

    @property
    def center(self) -> Tuple[float, float]:
        """Get current center position (cx, cy)."""
        if not self._initialized:
            return (0.0, 0.0)
        return (float(self.x[0]), float(self.x[1]))

    @property
    def size(self) -> Tuple[float, float]:
        """Get current size (width, height)."""
        if not self._initialized:
            return (0.0, 0.0)
        return (float(max(self.x[2], 1.0)), float(max(self.x[3], 1.0)))

    @property
    def state_vector(self) -> Optional[np.ndarray]:
        """Get full state vector."""
        return self.x.copy() if self._initialized else None

    @property
    def covariance_diagonal(self) -> Optional[np.ndarray]:
        """Get diagonal of covariance matrix (uncertainties)."""
        return np.diag(self.P).copy() if self._initialized else None

    def get_predicted_position(self, steps: int = 1) -> Tuple[float, float]:
        """
        Predict future center position.

        Args:
            steps: Number of time steps to predict

        Returns:
            Predicted center (cx, cy)
        """
        if not self._initialized:
            return (0.0, 0.0)

        vx, vy = self.velocity
        cx, cy = self.center

        return (cx + vx * steps, cy + vy * steps)

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self.x = None
        self.P = None
        self._initialized = False

    def __repr__(self) -> str:
        if not self._initialized:
            return "BboxKalmanFilter(uninitialized)"

        cx, cy = self.center
        vx, vy = self.velocity
        return (
            f"BboxKalmanFilter(center=({cx:.1f}, {cy:.1f}), "
            f"velocity=({vx:.2f}, {vy:.2f}))"
        )
