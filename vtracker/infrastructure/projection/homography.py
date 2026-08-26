"""Homography projection of image points onto the court and net planes.

Native implementation (ported from the old ``homography/homography.py``). Builds
two homographies from the image field points + the real-world court/net
constants and exposes the typed ``Projector`` Protocol.
"""

from __future__ import annotations

import cv2
import numpy as np

from vtracker.core.types import Point
from vtracker.domain.field import REAL_COURT_POINTS, REAL_NET_POINTS


class HomographyProjector:
    def __init__(self, field_points: dict) -> None:
        court_img = np.array(field_points["court"], dtype=np.float32)
        # Net image points: first 2 net markers + court points 9 and 8.
        net_img = np.array(
            field_points["net"][:2] + field_points["court"][9:]
            + field_points["court"][8:9], dtype=np.float32)
        self._h_field, _ = cv2.findHomography(court_img, REAL_COURT_POINTS)
        self._h_net, _ = cv2.findHomography(net_img, REAL_NET_POINTS)

    def project(self, point: Point, plane: str = "field") -> Point:
        if plane not in ("field", "net"):
            raise ValueError(f"plane must be 'field' or 'net', got {plane!r}")
        h = self._h_field if plane == "field" else self._h_net
        src = np.array([[[point.x, point.y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, h)
        x, y = dst[0][0]
        return Point(float(x), float(y))
