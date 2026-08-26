"""PrefetchVideoSource — decodes frames on a background thread.

Decoding, the CPU pre-processing in the ball detector and GPU inference all ran
on one thread, so each waited for the others: the GPU idled during
``cv2.VideoCapture.read()`` and the motion-mask work, and the CPU idled during
inference. This decorator overlaps decoding with everything else.

It wraps any ``VideoSource`` (Decorator pattern) rather than replacing the
OpenCV one, so the choice stays a composition-root decision and the existing
source keeps its single responsibility.

The queue is bounded: a slow consumer must not let the decoder read the whole
file into memory.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from vtracker.core.logging import get_logger
from vtracker.core.types import FrameBGR
from vtracker.domain.interfaces import VideoSource

_log = get_logger("vtracker.video")

# Sentinels distinguish "producer finished" from "producer raised".
_DONE = object()


class PrefetchVideoSource:
    def __init__(self, inner: VideoSource, queue_size: int = 8) -> None:
        self._inner = inner
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def fps(self) -> float:
        return self._inner.fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._inner.frame_size

    def _produce(self) -> None:
        try:
            for frame in self._inner.frames():
                if self._stop.is_set():
                    break
                # Time out so a consumer that stopped draining cannot wedge
                # this thread forever.
                while not self._stop.is_set():
                    try:
                        self._queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # surface it on the consumer thread
            self._queue.put(exc)
            return
        self._queue.put(_DONE)

    def frames(self) -> Iterator[FrameBGR]:
        self._thread = threading.Thread(target=self._produce, name="video-prefetch",
                                        daemon=True)
        self._thread.start()
        try:
            while True:
                item = self._queue.get()
                if item is _DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self._stop.set()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                _log.warning("prefetch thread did not stop within 2s")
        self._thread = None
        self._inner.release()
