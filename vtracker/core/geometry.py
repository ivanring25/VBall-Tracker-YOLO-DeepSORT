"""Pure geometry helpers (stdlib + numpy). Ported from the old
``utils/helpers.GeometryUtils`` so there is a single source of truth — the
duplicate speed calc that lived in ``DeepSortBallTracker.get_track_speed`` is
gone.
"""

from __future__ import annotations

from vtracker.core.types import BBox, Point


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.ltrb
    bx1, by1, bx2, by2 = b.ltrb
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def velocity(p_prev: Point, p_curr: Point) -> tuple[float, float]:
    return (p_curr.x - p_prev.x, p_curr.y - p_prev.y)
