"""Unit tests for PrefetchVideoSource (threaded frame decoding)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from vtracker.infrastructure.video.prefetch_source import PrefetchVideoSource


class Inner:
    fps = 25.0
    frame_size = (64, 48)

    def __init__(self, count=5, delay=0.0, fail_at=None):
        self._count = count
        self._delay = delay
        self._fail_at = fail_at
        self.released = False
        self.produced = 0

    def frames(self):
        for i in range(self._count):
            if self._fail_at is not None and i == self._fail_at:
                raise RuntimeError("decode exploded")
            if self._delay:
                time.sleep(self._delay)
            self.produced += 1
            yield np.full((48, 64, 3), i, dtype=np.uint8)

    def release(self):
        self.released = True


class TestPassthrough:
    def test_forwards_metadata(self):
        src = PrefetchVideoSource(Inner())
        assert src.fps == 25.0 and src.frame_size == (64, 48)

    def test_yields_every_frame_in_order(self):
        src = PrefetchVideoSource(Inner(count=5))
        values = [int(f[0, 0, 0]) for f in src.frames()]
        src.release()
        assert values == [0, 1, 2, 3, 4]

    def test_releases_inner_source(self):
        inner = Inner()
        src = PrefetchVideoSource(inner)
        list(src.frames())
        src.release()
        assert inner.released is True


class TestOverlap:
    def test_decoding_runs_ahead_of_the_consumer(self):
        """The point of the class: frames are produced while we're busy."""
        inner = Inner(count=6)
        src = PrefetchVideoSource(inner, queue_size=4)
        it = src.frames()
        next(it)                 # take one frame
        time.sleep(0.2)          # simulate slow processing
        assert inner.produced > 1, "producer should have run ahead"
        src.release()

    def test_queue_is_bounded(self):
        """A stalled consumer must not let the decoder buffer everything."""
        inner = Inner(count=100)
        src = PrefetchVideoSource(inner, queue_size=3)
        it = src.frames()
        next(it)
        time.sleep(0.2)
        assert inner.produced < 20, f"unbounded buffering: {inner.produced}"
        src.release()


class TestFailureAndShutdown:
    def test_producer_exception_reaches_the_consumer(self):
        src = PrefetchVideoSource(Inner(count=5, fail_at=2))
        with pytest.raises(RuntimeError, match="decode exploded"):
            list(src.frames())
        src.release()

    def test_release_stops_the_thread(self):
        before = threading.active_count()
        src = PrefetchVideoSource(Inner(count=1000, delay=0.001))
        it = src.frames()
        next(it)
        src.release()
        time.sleep(0.2)
        assert threading.active_count() <= before + 1

    def test_abandoning_the_iterator_does_not_leak(self):
        src = PrefetchVideoSource(Inner(count=1000, delay=0.001), queue_size=2)
        it = src.frames()
        next(it)
        del it
        src.release()
