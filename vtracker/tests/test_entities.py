"""Unit tests for the domain entities that replaced the old magic
tuples/dicts."""

from __future__ import annotations

import dataclasses

import pytest

from vtracker.core.types import BBox, Point
from vtracker.domain.entities import (
    BallDetection,
    PeopleFrame,
    Player,
    Referee,
    TrackState,
)


class TestBallDetection:
    def test_defaults(self):
        det = BallDetection(box=BBox(0, 0, 4, 4), confidence=0.5)
        assert det.label == "ball"
        assert det.estimated is False

    def test_center_delegates_to_box(self):
        assert BallDetection(box=BBox(0, 0, 10, 10), confidence=1.0).center == Point(5, 5)

    def test_is_immutable(self):
        """Detections are values; mutating one in place would silently affect
        every consumer holding the same reference."""
        det = BallDetection(box=BBox(0, 0, 1, 1), confidence=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            det.confidence = 0.9  # type: ignore[misc]


class TestPlayer:
    def test_foot_used_for_court_projection(self):
        assert Player(track_id="1", box=BBox(0, 0, 10, 40)).foot == Point(5, 40)

    def test_team_defaults_to_unknown(self):
        assert Player(track_id="1", box=BBox(0, 0, 1, 1)).team is None

    def test_libero_flag_defaults_false(self):
        assert Player(track_id="1", box=BBox(0, 0, 1, 1)).is_libero is False


class TestReferee:
    def test_foot(self):
        assert Referee(track_id="r", box=BBox(0, 0, 10, 20)).foot == Point(5, 20)


class TestPeopleFrame:
    def test_defaults_to_empty(self):
        frame = PeopleFrame()
        assert frame.players == {} and frame.referees == {}

    def test_independent_defaults(self):
        """Mutable defaults must not be shared between instances."""
        a, b = PeopleFrame(), PeopleFrame()
        a.players["1"] = Player(track_id="1", box=BBox(0, 0, 1, 1))
        assert b.players == {}


class TestTrackState:
    def test_last_position_none_when_empty(self):
        assert TrackState().last_position is None

    def test_last_position_returns_latest(self):
        state = TrackState()
        state.positions.extend([(1, 2), (3, 4)])
        assert state.last_position == Point(3, 4)

    def test_history_is_bounded(self):
        """Unbounded history would leak memory over a long match."""
        state = TrackState()
        for i in range(100):
            state.positions.append((i, i))
        assert len(state.positions) == state.positions.maxlen

    def test_independent_defaults(self):
        a, b = TrackState(), TrackState()
        a.positions.append((1, 1))
        assert len(b.positions) == 0
