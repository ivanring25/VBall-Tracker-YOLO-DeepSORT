"""IntervalPeopleDetector — runs people detection every N frames.

People inference is a full-frame YOLO pass, the most expensive single item in
the loop, and players move slowly compared with the ball. Running it every
``interval`` frames and reusing the last result in between trades a little
positional freshness for a near-linear cut in that cost.

Implemented as a decorator over any ``PeopleDetector`` (Composition over
Inheritance), so it is a composition-root choice and the real detector stays
unaware of it. ``interval=1`` is exactly the undecorated behaviour.

Trade-off worth stating plainly: between refreshes, player boxes are stale by
up to ``interval - 1`` frames. At 30 fps and interval 3 that is ~66 ms of lag on
overlays and the court minimap. Keep it low if you need tight boxes.
"""

from __future__ import annotations

from vtracker.core.types import FrameBGR
from vtracker.domain.entities import PeopleFrame
from vtracker.domain.interfaces import PeopleDetector


class IntervalPeopleDetector:
    def __init__(self, inner: PeopleDetector, interval: int = 1) -> None:
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self._inner = inner
        self._interval = interval
        self._calls = 0
        self._last = PeopleFrame()
        self.inferences = 0

    def process(self, frame: FrameBGR) -> PeopleFrame:
        if self._calls % self._interval == 0:
            self._last = self._inner.process(frame)
            self.inferences += 1
        self._calls += 1
        return self._last
