"""Constant-velocity Kalman filter for 2D ball tracking.

State vector:  [x, y, vx, vy]   (position + velocity, pixels & px/frame)
Measurement:   [x, y]

Optional constant downward acceleration `gravity` (px/frame^2) is injected on
the y-velocity to approximate ballistic motion between keyframes.

Implemented with numpy only (no external Kalman dependency) so it stays light
and predictable. Used by TrajectoryEngine for gap filling, motion prediction
and uncertainty estimation.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    def __init__(
        self,
        process_var: float = 1.0,
        measurement_var: float = 4.0,
        gravity: float = 0.0,
    ) -> None:
        self.gravity = gravity
        # State transition (dt = 1 frame).
        self.F = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=float,
        )
        # Control: gravity adds to vy each step.
        self.B = np.array([0, 0, 0, 1], dtype=float)
        # Measurement matrix (observe position only).
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * measurement_var
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1000.0  # large initial uncertainty

    def reset(self, x: float, y: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 1000.0

    def predict(self) -> tuple[float, float]:
        self.x = self.F @ self.x + self.B * self.gravity
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, x: float, y: float) -> None:
        z = np.array([x, y])
        y_res = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_res
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def uncertainty(self) -> float:
        """Scalar positional uncertainty ~ sqrt of position covariance trace."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
