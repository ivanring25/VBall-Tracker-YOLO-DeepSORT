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
    # Team assignment (KMeans on jersey colour). Tune per footage: raise the
    # threshold if liberos get merged into the main team colours, lower it if
    # ordinary players are misread as liberos.
    team_color_threshold: float = 100.0
    team_kmeans_n_init: int = 10


@dataclass
class TrackerConfig:
    max_age: int = 15
    n_init: int = 2
    max_cosine_distance: float = 0.4
    nn_budget: int = 50
    history_length: int = 10


@dataclass
class InterpolationConfig:
    """Kalman gap-bridging for missed ball detections."""

    enabled: bool = True
    max_gap: int = 5           # frames to coast before declaring the ball lost
    process_var: float = 1.0
    measurement_var: float = 4.0
    gravity: float = 0.0       # px/frame^2 added to vy (ballistic approximation)


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
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    field_points: dict = field(default_factory=dict)
    device: str = "auto"
    show_minimap: bool = True
    json_output_path: str | None = None

    def validate(self) -> None:
        """Fail fast at startup instead of mid-run.

        The old code only discovered a bad model path or a malformed
        field-points file when the pipeline was already several frames in
        (or, for the homography indices, on the first projected point).
        """
        errors: list[str] = []

        for label, path in (("video.input_path", self.video.input_path),
                            ("detector.model_path", self.detector.model_path),
                            ("people.model_path", self.people.model_path)):
            if not path:
                errors.append(f"{label} is empty")
            elif not os.path.exists(path):
                errors.append(f"{label}: file not found: {path}")

        w, h = self.video.frame_size
        if w <= 0 or h <= 0:
            errors.append(f"video.frame_size must be positive, got {(w, h)}")
        if self.video.skip_frames < 1:
            errors.append("video.skip_frames must be >= 1")

        if not 0.0 <= self.detector.confidence_threshold <= 1.0:
            errors.append("detector.confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.people.confidence_threshold <= 1.0:
            errors.append("people.confidence_threshold must be in [0, 1]")
        if self.detector.min_area >= self.detector.max_area:
            errors.append("detector.min_area must be < detector.max_area")
        if self.interpolation.max_gap < 1:
            errors.append("interpolation.max_gap must be >= 1")

        if self.device not in ("auto", "cpu", "cuda") and not self.device.startswith("cuda:"):
            errors.append(f"device must be auto|cpu|cuda[:N], got {self.device!r}")

        errors.extend(self._validate_field_points())

        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    def _validate_field_points(self) -> list[str]:
        """Check the shape the homography actually relies on."""
        if not self.field_points:
            if self.show_minimap:
                return ["show_minimap is on but no field_points/field_points_path given"]
            return []
        errors: list[str] = []
        court = self.field_points.get("court")
        net = self.field_points.get("net")
        # HomographyProjector indexes court[8] and court[9] for the net plane.
        if not isinstance(court, list) or len(court) < 10:
            errors.append(
                f"field_points.court needs >= 10 points (homography uses indices "
                f"8 and 9), got {len(court) if isinstance(court, list) else 'none'}")
        if not isinstance(net, list) or len(net) < 2:
            errors.append(
                f"field_points.net needs >= 2 points, got "
                f"{len(net) if isinstance(net, list) else 'none'}")
        for name, pts in (("court", court), ("net", net)):
            if isinstance(pts, list):
                bad = [p for p in pts if not (isinstance(p, (list, tuple)) and len(p) == 2)]
                if bad:
                    errors.append(f"field_points.{name} has {len(bad)} malformed point(s); "
                                  "each must be [x, y]")
        return errors

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @classmethod
    def load(cls, path: str, *, validate: bool = True) -> Config:
        """Load YAML config; resolve relative paths against the file's dir.

        Validates by default so a broken config fails at startup; pass
        ``validate=False`` in tests that build partial configs.
        """
        if not _HAS_YAML:
            raise RuntimeError("pyyaml is required to load config files")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        base = os.path.dirname(os.path.abspath(path))
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def rel(p: str | None) -> str | None:
            """Resolve an optional path against the config file's directory."""
            if not p:
                return p
            return p if os.path.isabs(p) else os.path.join(base, p)

        def rel_required(p: object, label: str) -> str:
            """Same, for keys that must be present and non-empty."""
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Config key {label!r} must be a non-empty path")
            resolved = rel(p)
            assert resolved is not None
            return resolved

        for section in ("video", "detector", "people"):
            if section not in raw:
                raise ValueError(f"Config is missing required section: {section!r}")

        v = raw["video"]
        d = raw["detector"]
        ppl = raw["people"]
        trk = raw.get("tracker", {})
        interp = raw.get("interpolation", {})

        field_points = raw.get("field_points")
        fp_path = rel(raw.get("field_points_path"))
        if field_points is None and fp_path:
            with open(fp_path, encoding="utf-8") as f:
                field_points = json.load(f)

        cfg = cls(
            video=VideoConfig(
                input_path=rel_required(v.get("input_path"), "video.input_path"),
                output_path=rel_required(v.get("output_path", "output.mp4"), "video.output_path"),
                frame_size=tuple(v.get("frame_size", (1280, 720))),
                skip_frames=v.get("skip_frames", 1),
                save_output=v.get("save_output", True),
                show_output=v.get("show_output", True),
            ),
            detector=DetectorConfig(model_path=rel_required(d.get("model_path"), "detector.model_path"),
                                    **{k: d[k] for k in
                                       ("confidence_threshold", "min_area", "max_area")
                                       if k in d}),
            people=PeopleConfig(model_path=rel_required(ppl.get("model_path"), "people.model_path"),
                                **{k: ppl[k] for k in
                                   ("confidence_threshold", "min_players_to_init_teams",
                                    "team_color_threshold", "team_kmeans_n_init")
                                   if k in ppl}),
            tracker=TrackerConfig(**trk),
            interpolation=InterpolationConfig(**interp),
            field_points=field_points or {},
            device=raw.get("device", "auto"),
            show_minimap=raw.get("show_minimap", True),
            json_output_path=rel(raw.get("json_output_path")),
        )
        if validate:
            cfg.validate()
        return cfg
