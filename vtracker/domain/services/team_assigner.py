"""Team assignment by jersey colour (KMeans). Ported from
``team_detector/team_assigner.py``.

Improvements over the original:
  * per-track colour cache (``player_team_dict``) is honoured AND the expensive
    KMeans patch sampling is cached by track_id, so a player's colour is
    computed once, not every frame (audit §7.1 #7).
  * typed via ``BBox``; no bare numpy bbox unpacking.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from vtracker.core.types import BBox, FrameBGR


class TeamAssigner:
    def __init__(self, color_threshold: float = 100.0) -> None:
        self.team_colors: dict[int, np.ndarray] = {}
        self.libero_colors: dict[int, list[np.ndarray]] = {}
        self.player_team: dict[str, int | None] = {}
        self._kmeans: KMeans | None = None
        self._color_threshold = color_threshold

    # --- colour sampling ---------------------------------------------------
    @staticmethod
    def _player_color(frame: FrameBGR, box: BBox) -> np.ndarray:
        x1, y1, x2, y2 = (int(v) for v in box.ltrb)
        crop = frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return np.zeros(3)
        cx0, cx1 = int(w * 0.3), int(w * 0.7)
        cy0, cy1 = int(h * 0.3), int(h * 0.7)
        patch = crop[cy0:cy1, cx0:cx1]
        if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
            return np.zeros(3)
        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        km.fit(patch.reshape(-1, 3))
        labels = km.labels_.reshape(patch.shape[0], patch.shape[1])
        unique, counts = np.unique(labels, return_counts=True)
        dominant = unique[np.argmax(counts)]
        return km.cluster_centers_[dominant]

    # --- init from first full frame ---------------------------------------
    def assign_team_colors(self, frame: FrameBGR, players: dict[str, BBox]) -> None:
        ids = list(players.keys())
        colors = np.array([self._player_color(frame, players[i]) for i in ids])
        km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(colors)
        unique, counts = np.unique(labels, return_counts=True)
        top = unique[np.argsort(counts)[-2:]]
        self.team_colors[1] = km.cluster_centers_[top[0]]
        self.team_colors[2] = km.cluster_centers_[top[1]]
        for cluster in unique:
            if cluster in top:
                continue
            d = [np.linalg.norm(km.cluster_centers_[cluster] - self.team_colors[t])
                 for t in (1, 2)]
            team = 1 if d[0] < d[1] else 2
            self.libero_colors.setdefault(team, []).append(km.cluster_centers_[cluster])
        self._kmeans = km
        for pid, label, color in zip(ids, labels, colors):
            if label in top:
                self.player_team[pid] = 1 if label == top[0] else 2
            else:
                d = [np.linalg.norm(color - self.team_colors[t]) for t in (1, 2)]
                self.player_team[pid] = 1 if d[0] < d[1] else 2

    @property
    def initialized(self) -> bool:
        return self._kmeans is not None

    # --- per-player lookup (cached) ---------------------------------------
    def team_of(self, frame: FrameBGR, box: BBox, track_id: str) -> int | None:
        if track_id in self.player_team:
            return self.player_team[track_id]
        if self._kmeans is None:
            self.player_team[track_id] = None
            return None
        color = self._player_color(frame, box)
        best_team, best_dist = None, float("inf")
        for team in (1, 2):
            d = np.linalg.norm(color - self.team_colors[team])
            if d < best_dist:
                best_dist, best_team = d, team
            for libero in self.libero_colors.get(team, []):
                dl = np.linalg.norm(color - libero)
                if dl < best_dist and dl < self._color_threshold:
                    best_dist, best_team = dl, team
        self.player_team[track_id] = best_team
        return best_team
