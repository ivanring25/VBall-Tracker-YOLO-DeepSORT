"""Ball tracker built on deep_sort_realtime. Native port of
``trackers/deepsort_tracker.DeepSortBallTracker``; exposes typed ``TrackState``
history and speeds.

Audit fixes:
  * speed uses video time (frames / fps), not wall-clock ``time.time()``, so it
    no longer depends on CPU load (§7.1 #4);
  * a single expiry path (the duplicate ``_remove_expired_tracks`` is gone).
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from vtracker.core.config import TrackerConfig
from vtracker.core.types import FrameBGR
from vtracker.domain.entities import BallDetection, TrackState


class DeepSortBallTracker:
    def __init__(self, cfg: TrackerConfig, fps: float = 30.0) -> None:
        self._max_age = cfg.max_age
        self._hist_len = cfg.history_length
        self._dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self._tracker = DeepSort(
            max_age=cfg.max_age, n_init=cfg.n_init,
            max_cosine_distance=cfg.max_cosine_distance,
            nn_budget=cfg.nn_budget, embedder_gpu=True)
        self._history: dict[str, TrackState] = defaultdict(self._new_state)
        self._frame = 0
        self.active_tracks: set[str] = set()

    def _new_state(self) -> TrackState:
        return TrackState(positions=deque(maxlen=self._hist_len),
                          timestamps=deque(maxlen=self._hist_len),
                          speeds=deque(maxlen=5))

    def update(self, detections: list[BallDetection], frame: FrameBGR) -> None:
        self._frame += 1
        ds_dets = [([d.box.x, d.box.y, d.box.w, d.box.h], d.confidence, d.label)
                   for d in detections]
        tracks = self._tracker.update_tracks(ds_dets, frame=frame)
        current: set[str] = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = str(track.track_id)
            current.add(tid)
            left, top, right, bottom = track.to_ltrb()
            cx, cy = (left + right) / 2, (top + bottom) / 2
            state = self._history[tid]
            if state.positions:
                px, py = state.positions[-1]
                speed = np.hypot(cx - px, cy - py) / self._dt
                state.speeds.append(speed)
            state.positions.append((cx, cy))
            state.timestamps.append(self._frame)
            state.last_seen = self._frame
            state.active = True
        self._expire()
        self.active_tracks = current

    def _expire(self) -> None:
        dead = [tid for tid, s in self._history.items()
                if self._frame - s.last_seen > self._max_age]
        for tid in dead:
            del self._history[tid]
            self.active_tracks.discard(tid)

    @property
    def tracks(self) -> dict[str, TrackState]:
        return self._history

    def speed(self, track_id: str) -> float:
        state = self._history.get(track_id)
        if not state or not state.speeds:
            return 0.0
        return float(np.median(state.speeds))
