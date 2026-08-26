"""Unit tests for IntervalPeopleDetector (run people inference every N frames)."""

from __future__ import annotations

import numpy as np
import pytest

from vtracker.core.types import BBox
from vtracker.domain.entities import PeopleFrame, Player
from vtracker.pipeline.stages.people_interval import IntervalPeopleDetector

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def process(self, frame) -> PeopleFrame:
        self.calls += 1
        tid = str(self.calls)
        return PeopleFrame(players={tid: Player(track_id=tid, box=BBox(0, 0, 1, 1))})


class TestInterval:
    def test_interval_one_calls_every_frame(self):
        inner = CountingDetector()
        det = IntervalPeopleDetector(inner, interval=1)
        for _ in range(5):
            det.process(FRAME)
        assert inner.calls == 5

    def test_interval_three_calls_every_third_frame(self):
        inner = CountingDetector()
        det = IntervalPeopleDetector(inner, interval=3)
        for _ in range(9):
            det.process(FRAME)
        assert inner.calls == 3
        assert det.inferences == 3

    def test_first_frame_always_runs(self):
        """Never return an empty placeholder when a real result is available."""
        inner = CountingDetector()
        det = IntervalPeopleDetector(inner, interval=10)
        assert det.process(FRAME).players != {}
        assert inner.calls == 1

    def test_result_is_reused_between_refreshes(self):
        inner = CountingDetector()
        det = IntervalPeopleDetector(inner, interval=3)
        first = det.process(FRAME)
        second = det.process(FRAME)
        third = det.process(FRAME)
        assert second is first and third is first
        fourth = det.process(FRAME)
        assert fourth is not first, "should refresh on the interval boundary"

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValueError, match="interval"):
            IntervalPeopleDetector(CountingDetector(), interval=0)


class TestSavings:
    def test_inference_count_scales_with_interval(self):
        for interval, expected in ((1, 30), (2, 15), (5, 6), (10, 3)):
            inner = CountingDetector()
            det = IntervalPeopleDetector(inner, interval=interval)
            for _ in range(30):
                det.process(FRAME)
            assert inner.calls == expected, interval
