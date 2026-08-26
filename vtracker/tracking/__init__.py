"""Shared tracking primitives: Kalman filtering and gap interpolation.

This subpackage is the *shared kernel* between the tracker pipeline and the
annotator tool — both need constant-velocity Kalman prediction and the same
gap-filling strategies. It depends only on numpy (SciPy optional), never on
OpenCV/torch/Qt, so either side can import it freely.
"""

from vtracker.tracking.interpolator import cubic_interp, has_scipy, linear_interp
from vtracker.tracking.kalman import KalmanFilter2D

__all__ = ["KalmanFilter2D", "linear_interp", "cubic_interp", "has_scipy"]
