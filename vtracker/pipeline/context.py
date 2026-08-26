"""FrameContext — the mutable object that flows through pipeline stages.

Each Stage reads what it needs and writes its result back, so stages stay
decoupled (they share data through the context, not through direct calls into
one another). This replaces the old monolithic ``run()`` that wired every step
inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vtracker.core.types import FrameBGR
from vtracker.domain.entities import BallDetection, PeopleFrame, TrackState


@dataclass
class FrameContext:
    index: int
    frame: FrameBGR                      # working frame (resized BGR)
    display: FrameBGR | None = None   # frame to draw overlays on
    detections: list[BallDetection] = field(default_factory=list)
    people: PeopleFrame = field(default_factory=PeopleFrame)
    ball_tracks: dict[str, TrackState] = field(default_factory=dict)
    ball_speeds: dict[str, float] = field(default_factory=dict)
    fps: float = 0.0

    @property
    def surface(self) -> FrameBGR:
        """The image drawers paint on, created on first use.

        Drawers used to reach for the optional ``display`` and each had to
        handle the ``None`` case (or quietly assume it was set); this makes the
        overlay target a single non-optional accessor.
        """
        if self.display is None:
            self.display = self.frame.copy()
        return self.display
