"""Typed configuration with explicit loading (no hidden disk I/O on construct).

The old ``utils/config.AppConfig`` read the field-config JSON inside
``__post_init__`` and hard-coded absolute paths like
``C:\\work_space\\ww_project\\...``. That made the object impossible to build on
any other machine and coupled "create a config" to "touch the filesystem".

Here construction is pure; loading is an explicit ``Config.load(path)`` that
reads a YAML file and resolves the field-points JSON relative to it. Paths come
from the file / env, never from source.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


@dataclass
class DetectorConfig:
    model_path: str
    confidence_threshold: float = 0.9
    min_area: int = 100
    max_area: int = 10000


@dataclass
class PeopleConfig:
    model_path: str
    confidence_threshold: float = 0.2
    min_players_to_init_teams: int = 12


@dataclass
class TrackerConfig:
    max_age: int = 15
    n_init: int = 2
    max_cosine_distance: float = 0.4
    nn_budget: int = 50
    history_length: int = 10


@dataclass
class VideoConfig:
    input_path: str
    output_path: str = "output.mp4"
    frame_size: tuple[int, int] = (1280, 720)
    skip_frames: int = 1
    save_output: bool = True
    show_output: bool = True


@dataclass
class Config:
    """Root config. Build via ``Config.load(path)``; never instantiate with
    hard-coded paths in source."""

    video: VideoConfig
    detector: DetectorConfig
    people: PeopleConfig
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    field_points: dict = field(default_factory=dict)
    device: str = "auto"
    show_minimap: bool = True

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load YAML config; resolve relative paths against the file's dir."""
        if not _HAS_YAML:
            raise RuntimeError("pyyaml is required to load config files")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        base = os.path.dirname(os.path.abspath(path))
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def rel(p: Optional[str]) -> Optional[str]:
            if not p:
                return p
            return p if os.path.isabs(p) else os.path.join(base, p)

        v = raw["video"]
        d = raw["detector"]
        ppl = raw["people"]
        trk = raw.get("tracker", {})

        field_points = raw.get("field_points")
        fp_path = rel(raw.get("field_points_path"))
        if field_points is None and fp_path:
            with open(fp_path, "r", encoding="utf-8") as f:
                field_points = json.load(f)

        return cls(
            video=VideoConfig(
                input_path=rel(v["input_path"]),
                output_path=rel(v.get("output_path", "output.mp4")),
                frame_size=tuple(v.get("frame_size", (1280, 720))),
                skip_frames=v.get("skip_frames", 1),
                save_output=v.get("save_output", True),
                show_output=v.get("show_output", True),
            ),
            detector=DetectorConfig(model_path=rel(d["model_path"]),
                                    **{k: d[k] for k in
                                       ("confidence_threshold", "min_area", "max_area")
                                       if k in d}),
            people=PeopleConfig(model_path=rel(ppl["model_path"]),
                                **{k: ppl[k] for k in
                                   ("confidence_threshold", "min_players_to_init_teams")
                                   if k in ppl}),
            tracker=TrackerConfig(**trk),
            field_points=field_points or {},
            device=raw.get("device", "auto"),
            show_minimap=raw.get("show_minimap", True),
        )
