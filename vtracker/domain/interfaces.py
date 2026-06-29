"""Abstractions the pipeline depends on (Dependency Inversion).

The pipeline imports ONLY these Protocols, never concrete classes from
``infrastructure``. Concrete detectors/trackers/exporters are injected at the
composition root (``vtracker.app``). This is what makes "add a new detector"
a registration rather than an edit to the orchestrator.

Protocol (structural typing) is used instead of ABC inheritance so existing
third-party-wrapping adapters satisfy the contract without a base class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vtracker.core.types import FrameBGR, Point
from vtracker.domain.entities import BallDetection, PeopleFrame


@runtime_checkable
class VideoSource(Protocol):
    """A source of frames. File, RTSP, image folder — all look the same here."""

    @property
    def fps(self) -> float: ...

    @property
    def frame_size(self) -> tuple[int, int]: ...

    def frames(self): ...  # -> Iterator[FrameBGR]

    def release(self) -> None: ...


@runtime_checkable
class BallDetector(Protocol):
    def detect(self, frame: FrameBGR) -> list[BallDetection]: ...


@runtime_checkable
class BallTracker(Protocol):
    def update(self, detections: list[BallDetection], frame: FrameBGR) -> None: ...


@runtime_checkable
class PeopleDetector(Protocol):
    """Detects + tracks players/referees in one call (wraps YOLO+ByteTrack)."""

    def process(self, frame: FrameBGR) -> PeopleFrame: ...


@runtime_checkable
class Projector(Protocol):
    def project(self, point: Point, plane: str = "field") -> Point: ...


@runtime_checkable
class FrameExporter(Protocol):
    """Consumes the rendered/annotated frame (video writer, JSON log, …)."""

    def write(self, frame: FrameBGR) -> None: ...

    def close(self) -> None: ...
