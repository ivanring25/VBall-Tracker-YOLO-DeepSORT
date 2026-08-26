"""Writes annotated frames to an mp4 file (FrameExporter Protocol).

A no-op-safe ``close()`` and lazy writer creation keep it simple to compose.
New export targets (JSON detections log, per-frame PNGs, websocket) just
implement the same Protocol and get appended as another ExportStage.
"""

from __future__ import annotations

import cv2

from vtracker.core.types import FrameBGR


class VideoExporter:
    def __init__(self, path: str, fps: float, frame_size: tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self._writer: cv2.VideoWriter | None = cv2.VideoWriter(
            path, fourcc, fps or 25.0, frame_size)
        if not self._writer.isOpened():
            raise OSError(f"Could not open video writer: {path}")

    def write(self, frame: FrameBGR) -> None:
        if self._writer is None:
            raise RuntimeError("video exporter already closed")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
