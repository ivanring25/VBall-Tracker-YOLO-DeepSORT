"""Composition root + CLI entry point.

This is the ONLY place that knows about concrete implementations. It reads a
config, constructs the infrastructure (detectors/trackers/exporters), wires them
into pipeline stages and runs them. Swapping a detector or adding a stage is a
change here, never in the pipeline/domain layers (Dependency Inversion).

Usage:
    python -m vtracker.app --config configs/match.yaml
"""

from __future__ import annotations

import argparse
import sys

from vtracker.core.config import Config
from vtracker.core.logging import get_logger
from vtracker.infrastructure.detectors.yolo_ball import YoloBallDetector
from vtracker.infrastructure.detectors.yolo_people import YoloPeopleTracker
from vtracker.infrastructure.display import DisplayStage
from vtracker.infrastructure.exporters.video_exporter import VideoExporter
from vtracker.infrastructure.projection.homography import HomographyProjector
from vtracker.infrastructure.trackers.deepsort_ball import DeepSortBallTracker
from vtracker.infrastructure.video.opencv_source import OpenCvVideoSource
from vtracker.pipeline.runner import PipelineRunner
from vtracker.pipeline.stages import (
    DetectBallStage,
    ExportStage,
    TrackBallStage,
    TrackPeopleStage,
    VisualizeStage,
)
from vtracker.visualization.drawers import (
    BallTrackDrawer,
    HudDrawer,
    MinimapDrawer,
    NetMinimapDrawer,
    PeopleDrawer,
)
from vtracker.visualization.renderer import Renderer


def build_and_run(config_path: str) -> None:
    log = get_logger("vtracker", log_file="tracker.log")
    cfg = Config.load(config_path)
    device = cfg.resolve_device()
    log.info("device: %s | video: %s", device, cfg.video.input_path)

    source = OpenCvVideoSource(cfg.video.input_path)

    detector = YoloBallDetector(cfg.detector, device)
    tracker = DeepSortBallTracker(cfg.tracker, fps=source.fps)
    people = YoloPeopleTracker(cfg.people)

    drawers = [BallTrackDrawer(), PeopleDrawer()]
    if cfg.show_minimap and cfg.field_points:
        projector = HomographyProjector(cfg.field_points)
        drawers.append(MinimapDrawer(projector))
        drawers.append(NetMinimapDrawer(projector))
    drawers.append(HudDrawer(device))
    renderer = Renderer(drawers)

    stages = [
        DetectBallStage(detector),
        TrackBallStage(tracker),
        TrackPeopleStage(people),
        VisualizeStage(renderer),
    ]

    exporter = None
    if cfg.video.save_output:
        exporter = VideoExporter(cfg.video.output_path, source.fps, cfg.video.frame_size)
        stages.append(ExportStage(exporter))
    if cfg.video.show_output:
        stages.append(DisplayStage())

    runner = PipelineRunner(source, stages,
                            frame_size=cfg.video.frame_size,
                            skip_frames=cfg.video.skip_frames)
    try:
        runner.run()
    finally:
        if exporter is not None:
            exporter.close()
        log.info("output: %s", cfg.video.output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VBall tracking pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args(argv)
    build_and_run(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
