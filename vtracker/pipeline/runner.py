"""PipelineRunner — composes Stages over a VideoSource (Composition over
Inheritance).

A run is just ``for frame in source: for stage in stages: stage(ctx)``. Adding a
step = appending a Stage; you never edit the loop. This is the structural
replacement for ``BallTrackingPipeline.run()`` (the old God Object).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import cv2

from vtracker.core.logging import get_logger
from vtracker.core.types import FrameBGR
from vtracker.domain.interfaces import VideoSource
from vtracker.pipeline.context import FrameContext

# A Stage is any callable taking the context and mutating/returning it.
Stage = Callable[[FrameContext], FrameContext]


class PipelineRunner:
    def __init__(self, source: VideoSource, stages: Sequence[Stage],
                 *, frame_size: tuple[int, int], skip_frames: int = 1) -> None:
        self._source = source
        self._stages = list(stages)
        self._frame_size = frame_size
        self._skip = max(1, skip_frames)
        self._log = get_logger("vtracker.pipeline")
        self.processed = 0

    def run(self) -> None:
        start = time.perf_counter()
        # Throughput of the previous frame. Stages (HUD, JSON export) read
        # ctx.fps, so it must be known *before* they run — it used to be
        # assigned after the stage loop, which meant every consumer saw 0.0.
        last_fps = 0.0
        try:
            for index, raw in enumerate(self._source.frames()):
                if index % self._skip != 0:
                    continue
                t0 = time.perf_counter()
                frame: FrameBGR = cv2.resize(raw, self._frame_size)
                # No eager copy: FrameContext.surface allocates the overlay
                # buffer only if something actually draws on it, so headless
                # runs don't pay a full-frame memcpy per frame.
                ctx = FrameContext(index=index, frame=frame, fps=last_fps)
                for stage in self._stages:
                    ctx = stage(ctx)
                self.processed += 1
                dt = time.perf_counter() - t0
                last_fps = (1.0 / dt) if dt > 0 else 0.0
                if self.processed % 50 == 0:
                    self._log.info("processed %d frames (%.1f fps)",
                                   self.processed, last_fps)
        except KeyboardInterrupt:
            self._log.info("interrupted by user")
        finally:
            self._source.release()
            total = time.perf_counter() - start
            avg = self.processed / total if total else 0.0
            self._log.info("done: %d frames in %.1fs (avg %.1f fps)",
                           self.processed, total, avg)
