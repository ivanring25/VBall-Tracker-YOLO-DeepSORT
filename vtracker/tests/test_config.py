"""Unit tests for config loading and validation.

The point of validation is failing at startup with a readable message instead
of crashing several frames into a run, so these tests assert on the messages.
"""

from __future__ import annotations

import json

import pytest
import yaml

from vtracker.core.config import Config


@pytest.fixture
def workspace(tmp_path):
    """A directory with real model/video stand-ins and valid field points."""
    for name in ("match.mp4", "ball.pt", "people.pt"):
        (tmp_path / name).write_bytes(b"stub")
    field = {
        "court": [[i, i] for i in range(10)],
        "net": [[0, 0], [10, 0]],
    }
    (tmp_path / "field.json").write_text(json.dumps(field), encoding="utf-8")
    return tmp_path


def _write_config(tmp_path, **overrides) -> str:
    raw: dict = {
        "video": {"input_path": "match.mp4", "output_path": "out.mp4"},
        "detector": {"model_path": "ball.pt"},
        "people": {"model_path": "people.pt"},
        "field_points_path": "field.json",
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(value)
        else:
            raw[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


class TestLoading:
    def test_loads_valid_config(self, workspace):
        cfg = Config.load(_write_config(workspace))
        assert cfg.video.frame_size == (1280, 720)
        assert cfg.tracker.max_age == 15
        assert cfg.interpolation.enabled is True
        assert len(cfg.field_points["court"]) == 10

    def test_relative_paths_resolved_against_config_dir(self, workspace):
        cfg = Config.load(_write_config(workspace))
        assert cfg.video.input_path == str(workspace / "match.mp4")
        assert cfg.detector.model_path == str(workspace / "ball.pt")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.load(str(tmp_path / "nope.yaml"))

    def test_missing_section_names_the_section(self, workspace):
        path = workspace / "bad.yaml"
        path.write_text(yaml.safe_dump({"video": {"input_path": "match.mp4"}}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="detector"):
            Config.load(str(path))

    def test_optional_sections_use_defaults(self, workspace):
        cfg = Config.load(_write_config(workspace))
        assert cfg.people.team_color_threshold == 100.0
        assert cfg.json_output_path is None

    def test_interpolation_section_is_read(self, workspace):
        cfg = Config.load(_write_config(
            workspace, interpolation={"enabled": False, "max_gap": 9}))
        assert cfg.interpolation.enabled is False
        assert cfg.interpolation.max_gap == 9


class TestValidation:
    def test_missing_model_file_is_reported(self, workspace):
        path = _write_config(workspace, detector={"model_path": "ghost.pt"})
        with pytest.raises(ValueError, match="detector.model_path"):
            Config.load(path)

    def test_missing_video_is_reported(self, workspace):
        path = _write_config(workspace, video={"input_path": "ghost.mp4"})
        with pytest.raises(ValueError, match="video.input_path"):
            Config.load(path)

    def test_all_errors_reported_together(self, workspace):
        """One run should surface every problem, not just the first."""
        path = _write_config(workspace,
                             detector={"model_path": "ghost.pt"},
                             people={"model_path": "ghost2.pt"})
        with pytest.raises(ValueError) as excinfo:
            Config.load(path)
        assert "detector.model_path" in str(excinfo.value)
        assert "people.model_path" in str(excinfo.value)

    def test_confidence_out_of_range(self, workspace):
        path = _write_config(workspace, detector={"confidence_threshold": 1.5})
        with pytest.raises(ValueError, match="confidence_threshold"):
            Config.load(path)

    def test_min_area_must_be_below_max_area(self, workspace):
        path = _write_config(workspace, detector={"min_area": 500, "max_area": 100})
        with pytest.raises(ValueError, match="min_area"):
            Config.load(path)

    def test_skip_frames_must_be_positive(self, workspace):
        path = _write_config(workspace, video={"skip_frames": 0})
        with pytest.raises(ValueError, match="skip_frames"):
            Config.load(path)

    def test_bad_device_rejected(self, workspace):
        path = _write_config(workspace, device="tpu")
        with pytest.raises(ValueError, match="device"):
            Config.load(path)

    def test_cuda_index_device_accepted(self, workspace):
        assert Config.load(_write_config(workspace, device="cuda:1")).device == "cuda:1"

    def test_validation_can_be_skipped(self, workspace):
        cfg = Config.load(_write_config(workspace, detector={"model_path": "ghost.pt"}),
                          validate=False)
        assert cfg.detector.model_path.endswith("ghost.pt")


class TestFieldPointValidation:
    """HomographyProjector indexes court[8] and court[9]; a short list used to
    blow up mid-run with an IndexError."""

    def test_too_few_court_points_rejected(self, workspace):
        (workspace / "field.json").write_text(
            json.dumps({"court": [[0, 0]] * 4, "net": [[0, 0], [1, 1]]}),
            encoding="utf-8")
        with pytest.raises(ValueError, match="court"):
            Config.load(_write_config(workspace))

    def test_too_few_net_points_rejected(self, workspace):
        (workspace / "field.json").write_text(
            json.dumps({"court": [[i, i] for i in range(10)], "net": [[0, 0]]}),
            encoding="utf-8")
        with pytest.raises(ValueError, match="net"):
            Config.load(_write_config(workspace))

    def test_malformed_point_rejected(self, workspace):
        (workspace / "field.json").write_text(
            json.dumps({"court": [[i, i] for i in range(9)] + [[1, 2, 3]],
                        "net": [[0, 0], [1, 1]]}),
            encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            Config.load(_write_config(workspace))

    def test_minimap_without_field_points_rejected(self, workspace):
        raw = {
            "video": {"input_path": "match.mp4"},
            "detector": {"model_path": "ball.pt"},
            "people": {"model_path": "people.pt"},
            "show_minimap": True,
        }
        path = workspace / "nofield.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="field_points"):
            Config.load(str(path))

    def test_no_field_points_ok_when_minimap_off(self, workspace):
        raw = {
            "video": {"input_path": "match.mp4"},
            "detector": {"model_path": "ball.pt"},
            "people": {"model_path": "people.pt"},
            "show_minimap": False,
        }
        path = workspace / "nofield.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        assert Config.load(str(path)).field_points == {}
