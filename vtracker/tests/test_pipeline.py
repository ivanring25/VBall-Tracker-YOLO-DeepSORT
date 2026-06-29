"""Architecture smoke test — runs the WHOLE pipeline with fakes, no GPU / video.

This is exactly the payoff of the refactor: because the pipeline depends on
Protocols (not concrete YOLO/DeepSORT classes), we can inject trivial fakes and
verify the orchestration end-to-end in milliseconds.

Run:  python -m vtracker.tests.test_pipeline      (or: pytest)
"""

from __future__ import annotations

import numpy as np

from vtracker.core.types import BBox
from vtracker.domain.entities import BallDetection, PeopleFrame, Player
from vtracker.pipeline.context import FrameContext
from vtracker.pipeline.runner import PipelineRunner
from vtracker.pipeline.stages import (
    DetectBallStage,
    TrackBallStage,
    TrackPeopleStage,
    VisualizeStage,
)


class FakeSource:
    fps = 30.0
    frame_size = (64, 48)

    def frames(self):
        for _ in range(5):
            yield np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def detect(self, frame):
        return [BallDetection(box=BBox(10, 10, 5, 5), confidence=0.9)]


class FakeTracker:
    def __init__(self):
        self.calls = 0

    def update(self, detections, frame):
        self.calls += 1
        assert all(isinstance(d, BallDetection) for d in detections)


class FakePeople:
    def process(self, frame) -> PeopleFrame:
        return PeopleFrame(players={"1": Player(track_id="1", box=BBox(0, 0, 4, 8), team=1)})


class RecordingRenderer:
    def __init__(self):
        self.rendered = 0

    def render(self, ctx: FrameContext):
        assert ctx.detections and ctx.people.players
        self.rendered += 1


def run() -> None:
    source = FakeSource()
    tracker = FakeTracker()
    renderer = RecordingRenderer()
    runner = PipelineRunner(
        source,
        [
            DetectBallStage(FakeDetector()),
            TrackBallStage(tracker),
            TrackPeopleStage(FakePeople()),
            VisualizeStage(renderer),
        ],
        frame_size=(64, 48),
        skip_frames=1,
    )
    runner.run()
    assert runner.processed == 5, runner.processed
    assert tracker.calls == 5
    assert renderer.rendered == 5
    assert getattr(source, "released", False)
    print("PIPELINE SMOKE TEST PASSED")


if __name__ == "__main__":
    run()
