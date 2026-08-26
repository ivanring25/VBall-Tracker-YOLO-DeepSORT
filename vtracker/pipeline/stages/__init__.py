"""Pipeline stages — each is a small callable ``(ctx) -> ctx``.

Stages depend only on domain interfaces, so any conforming implementation can be
injected. Order is decided by the composition root, not hard-coded here.
"""

from __future__ import annotations

from vtracker.core.logging import get_logger
from vtracker.domain.interfaces import (
    BallDetector,
    BallTracker,
    FrameExporter,
    PeopleDetector,
)
from vtracker.pipeline.context import FrameContext
from vtracker.pipeline.stages.interpolate import InterpolateStage

_log = get_logger("vtracker.stages")

__all__ = [
    "DetectBallStage", "InterpolateStage", "TrackBallStage", "TrackPeopleStage",
    "VisualizeStage", "ExportStage", "ExportContextStage",
]


class DetectBallStage:
    def __init__(self, detector: BallDetector) -> None:
        self._detector = detector

    def __call__(self, ctx: FrameContext) -> FrameContext:
        try:
            ctx.detections = self._detector.detect(ctx.frame)
        except Exception:  # detection must not kill the run, but log it
            _log.exception("ball detection failed on frame %d", ctx.index)
            ctx.detections = []
        return ctx


class TrackBallStage:
    def __init__(self, tracker: BallTracker) -> None:
        self._tracker = tracker

    def __call__(self, ctx: FrameContext) -> FrameContext:
        self._tracker.update(ctx.detections, ctx.frame)
        tracks = self._tracker.tracks
        ctx.ball_tracks = tracks
        # Only active tracks are drawn or exported, so don't run the median
        # over the speed history of tracks nobody will look at.
        ctx.ball_speeds = {tid: self._tracker.speed(tid)
                           for tid, state in tracks.items() if state.active}
        return ctx


class TrackPeopleStage:
    def __init__(self, people: PeopleDetector) -> None:
        self._people = people

    def __call__(self, ctx: FrameContext) -> FrameContext:
        try:
            ctx.people = self._people.process(ctx.frame)
        except Exception:
            _log.exception("people tracking failed on frame %d", ctx.index)
        return ctx


class VisualizeStage:
    """Renders overlays. The renderer is any object with ``render(ctx)``."""

    def __init__(self, renderer) -> None:
        self._renderer = renderer

    def __call__(self, ctx: FrameContext) -> FrameContext:
        self._renderer.render(ctx)
        return ctx


class ExportStage:
    def __init__(self, exporter: FrameExporter) -> None:
        self._exporter = exporter

    def __call__(self, ctx: FrameContext) -> FrameContext:
        self._exporter.write(ctx.display if ctx.display is not None else ctx.frame)
        return ctx


class ExportContextStage:
    """Export sinks that record structured results rather than pixels
    (e.g. ``JsonDetectionExporter``)."""

    def __init__(self, exporter) -> None:
        self._exporter = exporter

    def __call__(self, ctx: FrameContext) -> FrameContext:
        self._exporter.write_context(ctx)
        return ctx
