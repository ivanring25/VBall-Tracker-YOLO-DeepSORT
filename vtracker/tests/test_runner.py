"""Regression tests for PipelineRunner behaviour that consumers depend on."""

from __future__ import annotations

import numpy as np

from vtracker.pipeline.runner import PipelineRunner


class Source:
    fps = 30.0
    frame_size = (64, 48)

    def __init__(self, count: int = 4) -> None:
        self._count = count
        self.released = False

    def frames(self):
        for _ in range(self._count):
            yield np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _run(stages, count=4, **kw):
    source = Source(count)
    runner = PipelineRunner(source, stages, frame_size=(64, 48), **kw)
    runner.run()
    return runner, source


class TestFpsVisibility:
    def test_stages_see_a_real_fps_after_the_first_frame(self):
        """HUD and the JSON export read ctx.fps; it used to be assigned after
        the stage loop, so every consumer saw 0.0 on every frame."""
        seen: list[float] = []

        def spy(ctx):
            seen.append(ctx.fps)
            return ctx

        _run([spy], count=4)
        assert seen[0] == 0.0, "no measurement exists yet on the first frame"
        assert all(f > 0.0 for f in seen[1:]), seen

    def test_fps_is_not_recomputed_per_stage(self):
        values = []

        def a(ctx):
            values.append(ctx.fps)
            return ctx

        def b(ctx):
            values.append(ctx.fps)
            return ctx

        _run([a, b], count=3)
        # Both stages in a frame must observe the same figure.
        assert values[0] == values[1]
        assert values[2] == values[3]


class TestOverlayAllocation:
    def test_no_overlay_buffer_when_nothing_draws(self):
        """Headless runs (JSON only) must not pay a full-frame copy."""
        states: list[bool] = []

        def spy(ctx):
            states.append(ctx.display is None)
            return ctx

        _run([spy])
        assert all(states), "display should stay unallocated until drawn on"

    def test_surface_allocates_on_demand_and_is_stable(self):
        ids: list[int] = []

        def draw(ctx):
            ctx.surface[0, 0] = (1, 2, 3)
            ids.append(id(ctx.surface))
            ids.append(id(ctx.surface))  # second access must reuse the buffer
            return ctx

        _run([draw], count=1)
        assert ids[0] == ids[1]

    def test_surface_does_not_alias_the_source_frame(self):
        """Drawing must not corrupt the frame detectors were given."""
        def draw(ctx):
            ctx.surface[:] = 255
            assert ctx.frame.max() == 0
            return ctx

        _run([draw], count=1)


class TestLoopMechanics:
    def test_processes_every_frame_by_default(self):
        runner, _ = _run([lambda c: c], count=5)
        assert runner.processed == 5

    def test_skip_frames_processes_every_nth(self):
        indices: list[int] = []

        def spy(ctx):
            indices.append(ctx.index)
            return ctx

        runner, _ = _run([spy], count=6, skip_frames=2)
        assert indices == [0, 2, 4]
        assert runner.processed == 3

    def test_source_released_even_on_stage_failure(self):
        def boom(ctx):
            raise KeyboardInterrupt

        _, source = _run([boom], count=3)
        assert source.released is True
