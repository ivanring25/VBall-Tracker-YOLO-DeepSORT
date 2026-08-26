"""InterpolateStage — bridges short ball-detection gaps with Kalman prediction.

The old pipeline had no gap handling at all: one missed frame and the ball
simply vanished from the overlay, and DeepSORT's track aged out. The annotator
solves the same problem *offline* (interpolating between keyframes it can see on
both sides), but a streaming pipeline has no future frames — so the online
analogue is Kalman coasting:

    detection present  -> update the filter, pass the measurement through
    detection missing  -> predict; while the gap is shorter than ``max_gap``
                          emit a synthetic ``estimated`` detection so the track
                          survives
    gap too long       -> stop emitting and reset (the ball is genuinely gone)

Placed between DetectBallStage and TrackBallStage so the synthetic detection
reaches the tracker and keeps its track alive.
"""

from __future__ import annotations

from vtracker.core.logging import get_logger
from vtracker.core.types import BBox
from vtracker.domain.entities import BallDetection
from vtracker.pipeline.context import FrameContext
from vtracker.tracking.kalman import KalmanFilter2D

_log = get_logger("vtracker.interpolate")


class InterpolateStage:
    def __init__(self, max_gap: int = 5, process_var: float = 1.0,
                 measurement_var: float = 4.0, gravity: float = 0.0) -> None:
        self._max_gap = max_gap
        self._kf = KalmanFilter2D(process_var=process_var,
                                  measurement_var=measurement_var,
                                  gravity=gravity)
        self._initialized = False
        self._gap = 0
        self._last_size = (10.0, 10.0)
        self.filled_frames = 0

    def _best(self, detections: list[BallDetection]) -> BallDetection:
        return max(detections, key=lambda d: d.confidence)

    def __call__(self, ctx: FrameContext) -> FrameContext:
        measured = [d for d in ctx.detections if not d.estimated]

        if measured:
            det = self._best(measured)
            centre = det.center
            if not self._initialized:
                self._kf.reset(centre.x, centre.y)
                self._initialized = True
            else:
                self._kf.predict()
                self._kf.update(centre.x, centre.y)
            self._last_size = (det.box.w, det.box.h)
            self._gap = 0
            return ctx

        if not self._initialized:
            return ctx

        self._gap += 1
        if self._gap > self._max_gap:
            if self._gap == self._max_gap + 1:
                _log.debug("ball lost after %d empty frames (frame %d)",
                           self._max_gap, ctx.index)
            self._initialized = False
            return ctx

        px, py = self._kf.predict()
        w, h = self._last_size
        # Confidence decays across the gap so consumers can weight it down.
        confidence = max(0.0, 1.0 - self._gap / (self._max_gap + 1))
        ctx.detections = [BallDetection(
            box=BBox(px - w / 2, py - h / 2, w, h),
            confidence=confidence,
            estimated=True,
        )]
        self.filled_frames += 1
        return ctx
