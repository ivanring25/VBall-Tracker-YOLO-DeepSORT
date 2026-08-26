"""Unit tests for JsonDetectionExporter (JSON Lines detection log)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vtracker.core.types import BBox
from vtracker.domain.entities import BallDetection, PeopleFrame, Player, Referee, TrackState
from vtracker.infrastructure.exporters.json_exporter import JsonDetectionExporter
from vtracker.pipeline.context import FrameContext


@pytest.fixture
def ctx() -> FrameContext:
    state = TrackState()
    state.active = True
    state.positions.append((12.345, 67.891))
    people = PeopleFrame(
        players={"7": Player(track_id="7", box=BBox(1, 2, 3, 4), team=1)},
        referees={"r1": Referee(track_id="r1", box=BBox(5, 6, 7, 8))},
    )
    return FrameContext(
        index=42,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        detections=[BallDetection(box=BBox(10, 20, 5, 5), confidence=0.87)],
        people=people,
        ball_tracks={"1": state},
        ball_speeds={"1": 123.456},
        fps=29.97,
    )


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestExport:
    def test_writes_one_line_per_frame(self, tmp_path, ctx):
        out = tmp_path / "det.jsonl"
        exporter = JsonDetectionExporter(str(out))
        exporter.write_context(ctx)
        exporter.write_context(ctx)
        exporter.close()
        assert len(_read(out)) == 2

    def test_records_all_entity_types(self, tmp_path, ctx):
        out = tmp_path / "det.jsonl"
        exporter = JsonDetectionExporter(str(out))
        exporter.write_context(ctx)
        exporter.close()
        record = _read(out)[0]
        assert record["frame"] == 42
        assert record["ball_detections"][0]["confidence"] == pytest.approx(0.87)
        assert record["ball_detections"][0]["estimated"] is False
        assert record["ball_tracks"]["1"]["speed"] == pytest.approx(123.46)
        assert record["players"][0]["team"] == 1
        assert record["referees"][0]["track_id"] == "r1"

    def test_marks_estimated_detections(self, tmp_path, ctx):
        ctx.detections = [BallDetection(box=BBox(0, 0, 1, 1), confidence=0.4,
                                        estimated=True)]
        out = tmp_path / "det.jsonl"
        exporter = JsonDetectionExporter(str(out))
        exporter.write_context(ctx)
        exporter.close()
        assert _read(out)[0]["ball_detections"][0]["estimated"] is True

    def test_inactive_tracks_are_omitted(self, tmp_path, ctx):
        ctx.ball_tracks["1"].active = False
        out = tmp_path / "det.jsonl"
        exporter = JsonDetectionExporter(str(out))
        exporter.write_context(ctx)
        exporter.close()
        assert _read(out)[0]["ball_tracks"] == {}

    def test_empty_frame_is_still_recorded(self, tmp_path):
        empty = FrameContext(index=0, frame=np.zeros((4, 4, 3), dtype=np.uint8))
        out = tmp_path / "det.jsonl"
        exporter = JsonDetectionExporter(str(out))
        exporter.write_context(empty)
        exporter.close()
        record = _read(out)[0]
        assert record["ball_detections"] == [] and record["players"] == []

    def test_partial_run_is_readable(self, tmp_path, ctx):
        """Lines are flushed per frame, so an interrupted run still parses."""
        out = tmp_path / "det.jsonl"
        exporter = JsonDetectionExporter(str(out))
        exporter.write_context(ctx)
        exporter.close()  # simulates the finally-block on interrupt
        assert len(_read(out)) == 1

    def test_close_is_idempotent(self, tmp_path, ctx):
        exporter = JsonDetectionExporter(str(tmp_path / "det.jsonl"))
        exporter.close()
        exporter.close()
