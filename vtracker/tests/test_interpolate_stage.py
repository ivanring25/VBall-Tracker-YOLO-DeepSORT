"""Unit tests for InterpolateStage — the Kalman gap-bridging behaviour that
keeps the ball from vanishing on a missed detection."""

from __future__ import annotations

import numpy as np
import pytest

from vtracker.core.types import BBox
from vtracker.domain.entities import BallDetection
from vtracker.pipeline.context import FrameContext
from vtracker.pipeline.stages.interpolate import InterpolateStage


def _ctx(index: int, detections: list[BallDetection]) -> FrameContext:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return FrameContext(index=index, frame=frame, detections=list(detections))


def _det(x: float, y: float, conf: float = 0.9) -> BallDetection:
    return BallDetection(box=BBox(x, y, 10, 10), confidence=conf)


def _feed_straight_line(stage: InterpolateStage, steps: int = 8) -> None:
    for i in range(steps):
        stage(_ctx(i, [_det(i * 10.0, 50.0)]))


class TestMeasuredDetections:
    def test_measurements_pass_through_untouched(self):
        stage = InterpolateStage()
        det = _det(10, 20)
        ctx = stage(_ctx(0, [det]))
        assert ctx.detections == [det]
        assert stage.filled_frames == 0

    def test_picks_highest_confidence_candidate(self):
        stage = InterpolateStage()
        stage(_ctx(0, [_det(0, 0, conf=0.3), _det(100, 100, conf=0.95)]))
        # After a gap the prediction should sit near the confident detection.
        ctx = stage(_ctx(1, []))
        assert ctx.detections[0].center.x == pytest.approx(105.0, abs=15.0)


class TestGapBridging:
    def test_no_prediction_before_first_detection(self):
        """Nothing to coast from — must not invent a ball."""
        stage = InterpolateStage()
        assert stage(_ctx(0, [])).detections == []

    def test_fills_a_missed_frame(self):
        stage = InterpolateStage(max_gap=5)
        _feed_straight_line(stage)
        ctx = stage(_ctx(99, []))
        assert len(ctx.detections) == 1
        assert ctx.detections[0].estimated is True
        assert stage.filled_frames == 1

    def test_prediction_continues_the_motion(self):
        stage = InterpolateStage(max_gap=5)
        _feed_straight_line(stage, steps=10)  # last measured x = 90
        ctx = stage(_ctx(99, []))
        # Constant velocity of +10 px/frame should carry it forward, not stall.
        assert ctx.detections[0].center.x > 90.0

    def test_stops_after_max_gap(self):
        stage = InterpolateStage(max_gap=3)
        _feed_straight_line(stage)
        filled = [len(stage(_ctx(100 + i, [])).detections) for i in range(6)]
        assert filled == [1, 1, 1, 0, 0, 0]
        assert stage.filled_frames == 3

    def test_confidence_decays_across_the_gap(self):
        stage = InterpolateStage(max_gap=4)
        _feed_straight_line(stage)
        confs = [stage(_ctx(100 + i, [])).detections[0].confidence for i in range(4)]
        assert confs == sorted(confs, reverse=True), confs
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_recovers_after_reacquisition(self):
        stage = InterpolateStage(max_gap=2)
        _feed_straight_line(stage)
        stage(_ctx(100, []))
        stage(_ctx(101, [_det(200, 50)]))       # ball found again
        ctx = stage(_ctx(102, []))              # gap counter must have reset
        assert ctx.detections and ctx.detections[0].estimated

    def test_estimated_input_is_not_treated_as_measurement(self):
        """A synthetic detection must not be fed back as ground truth."""
        stage = InterpolateStage(max_gap=2)
        _feed_straight_line(stage)
        synthetic = BallDetection(box=BBox(0, 0, 10, 10), confidence=0.5, estimated=True)
        stage(_ctx(100, [synthetic]))
        stage(_ctx(101, [synthetic]))
        # Two frames of "estimated only" count as a 2-frame gap; the next is over.
        assert stage(_ctx(102, [synthetic])).detections[0].estimated
        assert stage(_ctx(103, [])).detections == []


class TestDisabledPath:
    def test_zero_size_box_still_produces_a_box(self):
        stage = InterpolateStage(max_gap=2)
        stage(_ctx(0, [BallDetection(box=BBox(5, 5, 0, 0), confidence=0.9)]))
        stage(_ctx(1, [BallDetection(box=BBox(6, 5, 0, 0), confidence=0.9)]))
        ctx = stage(_ctx(2, []))
        assert ctx.detections[0].box.w == 0
