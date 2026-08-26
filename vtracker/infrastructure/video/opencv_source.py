"""OpenCV-backed VideoSource. Swappable for an RTSP/folder source later
because the pipeline only depends on the ``VideoSource`` Protocol."""

from __future__ import annotations

from collections.abc import Iterator

import cv2

from vtracker.core.types import FrameBGR


class OpenCvVideoSource:
    def __init__(self, path: str) -> None:
        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise OSError(f"Could not open video: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._w, self._h)

    def frames(self) -> Iterator[FrameBGR]:
        if self._cap is None:
            raise RuntimeError("video source already released")
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
