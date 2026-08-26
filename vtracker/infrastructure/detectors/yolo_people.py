"""People detector/tracker: YOLO + ByteTrack + team assignment. Native port of
``trackers/people_tracker.PeopleTracker``; returns a typed ``PeopleFrame`` and
implements the ``PeopleDetector`` Protocol.
"""

from __future__ import annotations

import supervision as sv
from ultralytics import YOLO

from vtracker.core.config import PeopleConfig
from vtracker.core.logging import get_logger
from vtracker.core.types import BBox, FrameBGR
from vtracker.domain.entities import PeopleFrame, Player, Referee
from vtracker.domain.services.team_assigner import TeamAssigner
from vtracker.infrastructure.detectors._runtime import inference_context, use_half, warmup

_log = get_logger("vtracker.people")


class YoloPeopleTracker:
    def __init__(self, cfg: PeopleConfig, device: str = "cpu",
                 warmup_size: tuple[int, int] = (640, 640)) -> None:
        self._model = YOLO(cfg.model_path)
        self._tracker = sv.ByteTrack()
        self._conf = cfg.confidence_threshold
        self._min_init = cfg.min_players_to_init_teams
        self._teams = TeamAssigner(color_threshold=cfg.team_color_threshold,
                                   kmeans_n_init=cfg.team_kmeans_n_init)
        # Class-name -> id map is fixed for the loaded weights; it used to be
        # rebuilt from result.names on every frame.
        names = getattr(self._model, "names", {}) or {}
        self._player_cls = next((k for k, v in names.items() if v == "Players"), None)
        self._referee_cls = next((k for k, v in names.items() if v == "Referee"), None)
        self._device = device
        self._half = use_half(device)
        warmup(self._model, device, warmup_size)

    def process(self, frame: FrameBGR) -> PeopleFrame:
        with inference_context():
            result = self._model.predict(source=frame, conf=self._conf,
                                         device=self._device, half=self._half,
                                         verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        tracked = self._tracker.update_with_detections(detections)

        player_boxes: dict[str, BBox] = {}
        referee_boxes: dict[str, BBox] = {}
        for det in tracked:
            bbox, _, _, cls_id, track_id, _ = det
            tid = str(track_id)
            box = BBox.from_ltrb(*bbox.tolist())
            if cls_id == self._player_cls:
                player_boxes[tid] = box
            elif cls_id == self._referee_cls:
                referee_boxes[tid] = box

        if not self._teams.initialized and len(player_boxes) >= self._min_init:
            self._teams.assign_team_colors(frame, player_boxes)
            _log.info("teams initialized from %d players", len(player_boxes))

        players = {
            tid: Player(track_id=tid, box=box,
                        team=self._teams.team_of(frame, box, tid)
                        if self._teams.initialized else None)
            for tid, box in player_boxes.items()
        }
        referees = {tid: Referee(track_id=tid, box=box)
                    for tid, box in referee_boxes.items()}
        return PeopleFrame(players=players, referees=referees)
