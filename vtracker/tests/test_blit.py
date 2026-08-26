"""Proves the ROI-only minimap blit is equivalent to the old full-frame one."""

from __future__ import annotations

import cv2
import numpy as np

from vtracker.pipeline.context import FrameContext
from vtracker.visualization.drawers import _blit


def _reference_blit(surface, patch, x0, y0, alpha=0.5):
    """The previous implementation: copy the whole frame, paste, blend all."""
    h, w = surface.shape[:2]
    mh, mw = patch.shape[:2]
    x0 = max(0, min(x0, w - mw))
    y0 = max(0, min(y0, h - mh))
    overlay = surface.copy()
    overlay[y0:y0 + mh, x0:x0 + mw] = patch[:, :, :3]
    out = surface.copy()
    cv2.addWeighted(overlay, alpha, surface, 1 - alpha, 0, dst=out)
    return out


def _ctx(background):
    ctx = FrameContext(index=0, frame=background.copy())
    ctx.display = background.copy()
    return ctx


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
    patch = rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)
    return background, patch


class TestEquivalence:
    def test_matches_full_frame_blend(self):
        background, patch = _scene()
        ctx = _ctx(background)
        _blit(ctx, patch, x0=10, y0=20)
        assert np.array_equal(ctx.surface, _reference_blit(background, patch, 10, 20))

    def test_matches_at_several_positions(self):
        background, patch = _scene(seed=7)
        for x0, y0 in ((0, 0), (160, 90), (55, 33)):
            ctx = _ctx(background)
            _blit(ctx, patch, x0=x0, y0=y0)
            assert np.array_equal(ctx.surface,
                                  _reference_blit(background, patch, x0, y0)), (x0, y0)

    def test_pixels_outside_the_patch_are_untouched(self):
        background, patch = _scene(seed=3)
        ctx = _ctx(background)
        _blit(ctx, patch, x0=10, y0=20)
        mask = np.ones(background.shape[:2], dtype=bool)
        mask[20:50, 10:50] = False
        assert np.array_equal(ctx.surface[mask], background[mask])


class TestClamping:
    def test_out_of_bounds_position_is_clamped_inside(self):
        background, patch = _scene()
        ctx = _ctx(background)
        _blit(ctx, patch, x0=10_000, y0=10_000)
        assert np.array_equal(ctx.surface, _reference_blit(background, patch, 10_000, 10_000))

    def test_negative_position_is_clamped(self):
        background, patch = _scene()
        ctx = _ctx(background)
        _blit(ctx, patch, x0=-50, y0=-50)
        assert np.array_equal(ctx.surface, _reference_blit(background, patch, -50, -50))

    def test_patch_larger_than_frame_is_skipped(self):
        """The old version would raise on the broadcast; skip instead."""
        background, _ = _scene()
        oversized = np.zeros((500, 500, 3), dtype=np.uint8)
        ctx = _ctx(background)
        _blit(ctx, oversized, x0=0, y0=0)
        assert np.array_equal(ctx.surface, background)
