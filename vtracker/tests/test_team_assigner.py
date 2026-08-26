"""Unit tests for TeamAssigner.

Built on a synthetic frame with two clearly separated jersey colours, so the
clustering outcome is deterministic and the caching behaviour is observable.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtracker.core.types import BBox
from vtracker.domain.services.team_assigner import TeamAssigner

_RED = (0, 0, 200)
_BLUE = (200, 0, 0)


@pytest.fixture
def frame_and_players():
    """12 players in two colour groups, laid out in a row."""
    frame = np.zeros((200, 700, 3), dtype=np.uint8)
    players: dict[str, BBox] = {}
    for i in range(12):
        x = 10 + i * 55
        colour = _RED if i < 6 else _BLUE
        frame[50:150, x:x + 40] = colour
        players[str(i)] = BBox(x, 50, 40, 100)
    return frame, players


class TestInitialisation:
    def test_starts_uninitialised(self):
        assert TeamAssigner().initialized is False

    def test_becomes_initialised_after_assignment(self, frame_and_players):
        frame, players = frame_and_players
        ta = TeamAssigner(kmeans_n_init=3)
        ta.assign_team_colors(frame, players)
        assert ta.initialized is True

    def test_splits_into_two_teams(self, frame_and_players):
        frame, players = frame_and_players
        ta = TeamAssigner(kmeans_n_init=3)
        ta.assign_team_colors(frame, players)
        first_six = {ta.player_team[str(i)] for i in range(6)}
        last_six = {ta.player_team[str(i)] for i in range(6, 12)}
        assert len(first_six) == 1 and len(last_six) == 1
        assert first_six != last_six

    def test_assigns_only_teams_one_and_two(self, frame_and_players):
        frame, players = frame_and_players
        ta = TeamAssigner(kmeans_n_init=3)
        ta.assign_team_colors(frame, players)
        assert set(ta.player_team.values()) <= {1, 2}


class TestLookupAndCaching:
    def test_returns_none_before_initialisation(self, frame_and_players):
        frame, _ = frame_and_players
        ta = TeamAssigner()
        assert ta.team_of(frame, BBox(10, 50, 40, 100), "new") is None

    def test_known_player_uses_cache_without_resampling(self, frame_and_players, monkeypatch):
        """The audit flagged KMeans running per-player per-frame; a cached
        track id must not touch the sampler at all."""
        frame, players = frame_and_players
        ta = TeamAssigner(kmeans_n_init=3)
        ta.assign_team_colors(frame, players)
        calls: list[int] = []
        original = ta._player_color

        def counting(f, b):
            calls.append(1)
            return original(f, b)

        monkeypatch.setattr(ta, "_player_color", counting)
        ta.team_of(frame, players["0"], "0")
        assert calls == []

    def test_new_player_is_classified_and_then_cached(self, frame_and_players):
        frame, players = frame_and_players
        ta = TeamAssigner(kmeans_n_init=3)
        ta.assign_team_colors(frame, players)
        team_first = ta.team_of(frame, players["0"], "newbie")
        assert team_first in (1, 2)
        assert "newbie" in ta.player_team
        assert ta.team_of(frame, players["11"], "newbie") == team_first

    def test_new_player_matches_its_colour_group(self, frame_and_players):
        frame, players = frame_and_players
        ta = TeamAssigner(kmeans_n_init=3)
        ta.assign_team_colors(frame, players)
        red_team = ta.player_team["0"]
        assert ta.team_of(frame, players["1"], "red_newbie") == red_team


class TestRobustness:
    def test_tiny_box_returns_zero_colour(self, frame_and_players):
        frame, _ = frame_and_players
        ta = TeamAssigner()
        assert np.array_equal(ta._player_color(frame, BBox(0, 0, 2, 2)), np.zeros(3))

    def test_n_init_is_configurable(self):
        assert TeamAssigner(kmeans_n_init=3)._n_init == 3

    def test_color_threshold_is_configurable(self):
        assert TeamAssigner(color_threshold=42.0)._color_threshold == 42.0
