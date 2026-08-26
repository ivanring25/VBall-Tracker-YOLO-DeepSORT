"""Interpolation strategies for filling gaps between keyframes.

Priority order (TrajectoryEngine picks the best available for the segment):
  1) Linear         — always works, fallback
  2) Cubic spline   — SciPy, needs >= 4 points, smooth
  3) Kalman         — handled in trajectory_engine (stateful), motion-aware

This module covers the stateless geometric interpolators (1 and 2).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    _HAS_SCIPY = False


def linear_interp(
    frames: list[int], xs: list[float], ys: list[float], targets: list[int]
) -> dict[int, tuple[float, float]]:
    fx = np.interp(targets, frames, xs)
    fy = np.interp(targets, frames, ys)
    return {f: (float(x), float(y)) for f, x, y in zip(targets, fx, fy, strict=True)}


def cubic_interp(
    frames: list[int], xs: list[float], ys: list[float], targets: list[int]
) -> dict[int, tuple[float, float]]:
    """Cubic-spline interpolation; falls back to linear if SciPy is missing
    or there are too few points."""
    if not _HAS_SCIPY or len(frames) < 4:
        return linear_interp(frames, xs, ys, targets)
    csx = CubicSpline(frames, xs)
    csy = CubicSpline(frames, ys)
    return {f: (float(csx(f)), float(csy(f))) for f in targets}


def has_scipy() -> bool:
    return _HAS_SCIPY
