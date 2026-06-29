"""Renderer — composes drawers into one render pass (Composition over
Inheritance). The VisualizeStage calls ``render(ctx)``; the renderer just runs
each registered drawer in order. Add/remove overlays by changing the list."""

from __future__ import annotations

from typing import Sequence

from vtracker.pipeline.context import FrameContext


class Renderer:
    def __init__(self, drawers: Sequence) -> None:
        self._drawers = list(drawers)

    def render(self, ctx: FrameContext) -> None:
        if ctx.display is None:
            ctx.display = ctx.frame.copy()
        for drawer in self._drawers:
            drawer.draw(ctx)
