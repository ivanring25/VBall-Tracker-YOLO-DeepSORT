"""Optional on-screen preview stage (separated from file export).

Kept out of the export path so headless runs need no display. Raises
KeyboardInterrupt on ESC, which the runner treats as a clean stop.
"""

from __future__ import annotations

import cv2

from vtracker.pipeline.context import FrameContext


class DisplayStage:
    def __init__(self, window: str = "VBall Tracking") -> None:
        self._window = window

    def __call__(self, ctx: FrameContext) -> FrameContext:
        cv2.imshow(self._window, ctx.display if ctx.display is not None else ctx.frame)
        if cv2.waitKey(1) == 27:  # ESC
            raise KeyboardInterrupt
        return ctx
