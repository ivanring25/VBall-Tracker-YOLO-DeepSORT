"""Unit tests for the shared tracking kernel: Kalman filter + interpolators."""

from __future__ import annotations

import pytest

from vtracker.tracking.interpolator import cubic_interp, linear_interp
from vtracker.tracking.kalman import KalmanFilter2D


class TestKalmanFilter2D:
    def test_reset_sets_position_and_zero_velocity(self):
        kf = KalmanFilter2D()
        kf.reset(10.0, 20.0)
        assert kf.position == (10.0, 20.0)
        assert kf.velocity == (0.0, 0.0)

    def test_learns_constant_velocity(self):
        """After tracking a straight line, prediction should continue it."""
        kf = KalmanFilter2D(process_var=1.0, measurement_var=1.0)
        kf.reset(0.0, 0.0)
        for step in range(1, 16):
            kf.predict()
            kf.update(step * 5.0, step * 2.0)
        vx, vy = kf.velocity
        assert vx == pytest.approx(5.0, abs=0.6)
        assert vy == pytest.approx(2.0, abs=0.6)

    def test_prediction_extrapolates_forward(self):
        kf = KalmanFilter2D(measurement_var=0.5)
        kf.reset(0.0, 0.0)
        for step in range(1, 11):
            kf.predict()
            kf.update(step * 10.0, 0.0)
        px, _ = kf.predict()
        assert px > 100.0  # moved beyond the last measurement

    def test_gravity_pulls_velocity_down(self):
        kf = KalmanFilter2D(gravity=2.0)
        kf.reset(0.0, 0.0)
        kf.predict()
        assert kf.velocity[1] == pytest.approx(2.0)

    def test_uncertainty_shrinks_after_updates(self):
        kf = KalmanFilter2D()
        kf.reset(0.0, 0.0)
        initial = kf.uncertainty
        for _ in range(5):
            kf.predict()
            kf.update(1.0, 1.0)
        assert kf.uncertainty < initial


class TestInterpolators:
    def test_linear_hits_midpoint(self):
        out = linear_interp([0, 10], [0.0, 100.0], [0.0, 50.0], [5])
        assert out[5] == pytest.approx((50.0, 25.0))

    def test_linear_preserves_known_frames(self):
        out = linear_interp([0, 10], [0.0, 100.0], [0.0, 50.0], [0, 10])
        assert out[0] == pytest.approx((0.0, 0.0))
        assert out[10] == pytest.approx((100.0, 50.0))

    def test_cubic_falls_back_to_linear_when_too_few_points(self):
        frames, xs, ys = [0, 10], [0.0, 100.0], [0.0, 50.0]
        assert cubic_interp(frames, xs, ys, [5]) == linear_interp(frames, xs, ys, [5])

    def test_cubic_matches_linear_on_a_straight_line(self):
        """A cubic spline through collinear points is still that line."""
        frames = [0, 5, 10, 15, 20]
        xs = [float(f) * 2 for f in frames]
        ys = [float(f) for f in frames]
        out = cubic_interp(frames, xs, ys, [7])
        assert out[7] == pytest.approx((14.0, 7.0), abs=1e-6)
