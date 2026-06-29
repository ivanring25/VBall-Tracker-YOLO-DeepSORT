"""Lightweight shared value types (stdlib + numpy only).

These deliberately have no OpenCV/torch/Qt dependency so the domain layer stays
testable without heavy runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A decoded frame: HxWx3 uint8. Kept as a plain alias for readability.
FrameBGR = np.ndarray
FrameRGB = np.ndarray


@dataclass(frozen=True)
class Point:
    """A 2D point in image (pixel) coordinates."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in xywh (top-left + size) pixel coordinates."""

    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_ltrb(cls, left: float, top: float, right: float, bottom: float) -> "BBox":
        return cls(left, top, right - left, bottom - top)

    @property
    def ltrb(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h / 2)

    @property
    def foot(self) -> Point:
        """Bottom-center point (where a player touches the ground)."""
        return Point(self.x + self.w / 2, self.y + self.h)
