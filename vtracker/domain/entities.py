"""Domain entities — strict dataclasses replacing the old "magic" tuples/dicts.

Before this module the codebase passed detections as ``(x, y, w, h, conf)``
tuples and tracks/players as bare dicts (``data['positions']``,
``player['bbox']``). Those contracts lived only in the head of whoever wrote the
unpacking line. These dataclasses make the contracts explicit and typo-proof.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from vtracker.core.types import BBox, Point


@dataclass(frozen=True)
class BallDetection:
    """A single ball candidate produced by a Detector.

    Replaces the old ``(x, y, w, h, conf)`` tuple from
    ``YOLOBallDetector.detect``.
    """

    box: BBox
    confidence: float
    label: str = "ball"

    @property
    def center(self) -> Point:
        return self.box.center


@dataclass
class Player:
    """A tracked player. Replaces the old ``player_info`` dict."""

    track_id: str
    box: BBox
    team: Optional[int] = None
    is_libero: bool = False

    @property
    def foot(self) -> Point:
        return self.box.foot


@dataclass
class Referee:
    """A tracked referee. Replaces the old ``referee_info`` dict."""

    track_id: str
    box: BBox

    @property
    def foot(self) -> Point:
        return self.box.foot


@dataclass
class PeopleFrame:
    """Result of one people-tracking pass over a frame."""

    players: dict[str, Player] = field(default_factory=dict)
    referees: dict[str, Referee] = field(default_factory=dict)


@dataclass
class TrackState:
    """Per-track history for the ball. Replaces the nested ``defaultdict`` in
    ``DeepSortBallTracker.track_history``."""

    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    speeds: deque = field(default_factory=lambda: deque(maxlen=5))
    last_seen: int = 0
    active: bool = False

    @property
    def last_position(self) -> Optional[Point]:
        return Point(*self.positions[-1]) if self.positions else None
