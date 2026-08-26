"""Unit tests for the pure geometry layer (no OpenCV/torch needed)."""

from __future__ import annotations

import pytest

from vtracker.core.geometry import iou, velocity
from vtracker.core.types import BBox, Point


class TestBBox:
    def test_ltrb_roundtrip(self):
        box = BBox.from_ltrb(10, 20, 30, 60)
        assert (box.x, box.y, box.w, box.h) == (10, 20, 20, 40)
        assert box.ltrb == (10, 20, 30, 60)

    def test_center(self):
        assert BBox(0, 0, 10, 20).center == Point(5, 10)

    def test_foot_is_bottom_center(self):
        """Players are placed on the court by their feet, not their centre."""
        assert BBox(0, 0, 10, 20).foot == Point(5, 20)


class TestIou:
    def test_identical_boxes(self):
        box = BBox(0, 0, 10, 10)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou(BBox(0, 0, 10, 10), BBox(100, 100, 10, 10)) == 0.0

    def test_touching_boxes_do_not_overlap(self):
        assert iou(BBox(0, 0, 10, 10), BBox(10, 0, 10, 10)) == 0.0

    def test_half_overlap(self):
        a, b = BBox(0, 0, 10, 10), BBox(5, 0, 10, 10)
        # intersection 50, union 150
        assert iou(a, b) == pytest.approx(50 / 150)

    def test_degenerate_box_is_safe(self):
        """Zero-area boxes must not divide by zero."""
        assert iou(BBox(0, 0, 0, 0), BBox(0, 0, 0, 0)) == 0.0


class TestVelocity:
    def test_positive_direction(self):
        assert velocity(Point(0, 0), Point(3, 4)) == (3, 4)

    def test_negative_direction(self):
        assert velocity(Point(5, 5), Point(2, 1)) == (-3, -4)
