"""Unit tests for HomographyProjector.

Uses a synthetic, exactly-known camera: an identity-scaled view of the court so
the expected real-world coordinates can be computed by hand.
"""

from __future__ import annotations

import pytest

from vtracker.core.types import Point
from vtracker.domain.field import COURT_LENGTH_M, COURT_WIDTH_M, NET_HEIGHT_M
from vtracker.infrastructure.projection.homography import HomographyProjector

# Image points laid out so that x_px = x_m * 10 and y_px = (18 - y_m) * 10,
# i.e. a top-down "camera" 10 px per metre with y flipped (image y grows down).
_SCALE = 10.0


def _img(x_m: float, y_m: float) -> list[float]:
    return [x_m * _SCALE, (COURT_LENGTH_M - y_m) * _SCALE]


@pytest.fixture
def field_points() -> dict:
    # Must match the ordering of REAL_COURT_POINTS in domain/field.py.
    court = [
        _img(0.0, 18.0), _img(9.0, 18.0), _img(9.0, 0.0), _img(0.0, 0.0),
        _img(0.0, 12.0), _img(9.0, 12.0), _img(9.0, 6.0), _img(0.0, 6.0),
        _img(0.0, 9.0), _img(9.0, 9.0),
    ]
    # Net markers: the projector uses net[:2] plus court[9] and court[8].
    # Real net points are [[0,2.43],[9,2.43],[9,0],[0,0]], so net[0]/net[1] are
    # the top of the net above the left/right ends of the centre line.
    net = [
        [0.0, (COURT_LENGTH_M - 9.0) * _SCALE - NET_HEIGHT_M * _SCALE],
        [9.0 * _SCALE, (COURT_LENGTH_M - 9.0) * _SCALE - NET_HEIGHT_M * _SCALE],
    ]
    return {"court": court, "net": net}


class TestFieldPlane:
    def test_corners_map_to_court_corners(self, field_points):
        proj = HomographyProjector(field_points)
        origin = proj.project(Point(*_img(0.0, 0.0)), plane="field")
        assert origin.x == pytest.approx(0.0, abs=1e-3)
        assert origin.y == pytest.approx(0.0, abs=1e-3)

        far = proj.project(Point(*_img(9.0, 18.0)), plane="field")
        assert far.x == pytest.approx(COURT_WIDTH_M, abs=1e-3)
        assert far.y == pytest.approx(COURT_LENGTH_M, abs=1e-3)

    def test_court_centre(self, field_points):
        proj = HomographyProjector(field_points)
        centre = proj.project(Point(*_img(4.5, 9.0)), plane="field")
        assert centre.x == pytest.approx(4.5, abs=1e-3)
        assert centre.y == pytest.approx(9.0, abs=1e-3)

    def test_projection_is_stable(self, field_points):
        """Same input must give the same output (no accumulated state)."""
        proj = HomographyProjector(field_points)
        p = Point(*_img(3.0, 5.0))
        assert proj.project(p).as_tuple() == proj.project(p).as_tuple()


class TestNetPlane:
    def test_net_top_maps_to_net_height(self, field_points):
        proj = HomographyProjector(field_points)
        top_left = proj.project(Point(*field_points["net"][0]), plane="net")
        assert top_left.x == pytest.approx(0.0, abs=1e-3)
        assert top_left.y == pytest.approx(NET_HEIGHT_M, abs=1e-3)

    def test_floor_at_centre_line_maps_to_zero_height(self, field_points):
        proj = HomographyProjector(field_points)
        floor = proj.project(Point(*_img(0.0, 9.0)), plane="net")
        assert floor.y == pytest.approx(0.0, abs=1e-3)


class TestPlaneArgument:
    def test_invalid_plane_rejected(self, field_points):
        proj = HomographyProjector(field_points)
        with pytest.raises(ValueError, match="plane"):
            proj.project(Point(0, 0), plane="ceiling")
