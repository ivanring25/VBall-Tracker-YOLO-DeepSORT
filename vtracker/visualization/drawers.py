"""Focused drawers — the old 484-line ``Visualizer`` split by responsibility
(SRP). Each drawer renders one concern from typed entities onto the display
frame. New overlays = new drawer, registered in the renderer.
"""

from __future__ import annotations

import cv2
import numpy as np

from vtracker.core.types import Point
from vtracker.domain.field import COURT_LENGTH_M, COURT_WIDTH_M
from vtracker.pipeline.context import FrameContext

_TEAM_COLORS = {1: (0, 0, 255), 2: (255, 0, 0), None: (0, 255, 0)}
_REFEREE_COLOR = (0, 255, 255)


class BallDetectionDrawer:
    """Boxes + confidence for raw ball detections."""

    def __init__(self, color=(255, 0, 0), thickness=2) -> None:
        self._color = color
        self._thickness = thickness

    def draw(self, ctx: FrameContext) -> None:
        for det in ctx.detections:
            x, y, w, h = (int(v) for v in (det.box.x, det.box.y, det.box.w, det.box.h))
            cv2.rectangle(ctx.display, (x, y), (x + w, y + h), self._color, self._thickness)
            cv2.putText(ctx.display, f"{det.confidence:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


class PeopleDrawer:
    """Player/referee ellipses with team colour + id."""

    def draw(self, ctx: FrameContext) -> None:
        for player in ctx.people.players.values():
            color = _TEAM_COLORS.get(player.team, _TEAM_COLORS[None])
            label = (f"T{player.team}-{player.track_id}" if player.team is not None
                     else f"U-{player.track_id}")
            self._ellipse(ctx, player.box, color, label)
        for ref in ctx.people.referees.values():
            self._ellipse(ctx, ref.box, _REFEREE_COLOR, f"Ref-{ref.track_id}")

    @staticmethod
    def _ellipse(ctx: FrameContext, box, color, label) -> None:
        foot = box.foot
        cx, y2 = int(foot.x), int(foot.y)
        width = int(box.w / 2)
        cv2.ellipse(ctx.display, (cx, y2), (width, int(0.35 * width)),
                    0.0, -45, 235, color, 2, cv2.LINE_4)
        cv2.putText(ctx.display, label, (cx - width, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


class BallTrackDrawer:
    """Fading ball trajectory + id/speed label + movement arrow."""

    def __init__(self, history_color=(0, 255, 255)) -> None:
        self._color = np.array(history_color, dtype=np.uint8)

    def draw(self, ctx: FrameContext) -> None:
        for tid, state in ctx.ball_tracks.items():
            if not state.active or len(state.positions) < 2:
                continue
            pts = list(state.positions)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = tuple((self._color * alpha).astype(int).tolist())
                cv2.line(ctx.display, tuple(map(int, pts[i - 1])),
                         tuple(map(int, pts[i])), color, 2, cv2.LINE_AA)
            x, y = pts[-1]
            speed = ctx.ball_speeds.get(tid, 0.0)
            cv2.putText(ctx.display, f"ID:{tid} {speed:.0f}px/s",
                        (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.arrowedLine(ctx.display, tuple(map(int, pts[-2])),
                            tuple(map(int, pts[-1])), (0, 0, 255), 2, tipLength=0.3)


class MinimapDrawer:
    """Top-down court minimap with projected player/referee positions.

    Needs a Projector (court plane). Ported from ``draw_minimap_court``; the
    static court base is cached (audit §7.1 #6 — no per-frame canvas rebuild).
    """

    def __init__(self, projector, size: int = 150, padding: int = 15) -> None:
        self._projector = projector
        self._size = size
        self._pad = padding
        self._base = self._build_base()

    def _build_base(self) -> np.ndarray:
        s, pad = self._size, self._pad
        canvas = np.zeros((s + 2 * pad, s + 2 * pad, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        cv2.rectangle(canvas, (pad, pad), (pad + s, pad + s), (255, 255, 255), 1)
        for ym in (6.0, 9.0, 12.0):
            y = pad + int(((COURT_LENGTH_M - ym) / COURT_LENGTH_M) * s)
            cv2.line(canvas, (pad, y), (pad + s, y), (80, 80, 80), 1)
        return canvas

    def _plot(self, mm, x_m, y_m, color, tid=None) -> None:
        mx = self._pad + int((x_m / COURT_WIDTH_M) * self._size)
        my = self._pad + int((1 - y_m / COURT_LENGTH_M) * self._size)
        cv2.circle(mm, (mx, my), 4, color, -1)
        if tid is not None:
            cv2.putText(mm, str(tid), (mx + 5, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw(self, ctx: FrameContext) -> None:
        mm = self._base.copy()
        for player in ctx.people.players.values():
            self._project_and_plot(mm, player.foot,
                                   _TEAM_COLORS.get(player.team, _TEAM_COLORS[None]),
                                   player.track_id)
        for ref in ctx.people.referees.values():
            self._project_and_plot(mm, ref.foot, _REFEREE_COLOR, ref.track_id)
        self._blit(ctx, mm)

    def _project_and_plot(self, mm, foot: Point, color, tid) -> None:
        try:
            real = self._projector.project(foot, plane="field")
        except Exception:
            return
        self._plot(mm, real.x, real.y, color, tid)

    @staticmethod
    def _blit(ctx: FrameContext, mm: np.ndarray) -> None:
        h, w = ctx.display.shape[:2]
        mh, mw = mm.shape[:2]
        x0, y0 = w - mw - 20, h - mh - 20
        overlay = ctx.display.copy()
        overlay[y0:y0 + mh, x0:x0 + mw] = mm
        cv2.addWeighted(overlay, 0.5, ctx.display, 0.5, 0, dst=ctx.display)


class HudDrawer:
    """Frame index + FPS + counts in the corner."""

    def __init__(self, device: str) -> None:
        self._device = device

    def draw(self, ctx: FrameContext) -> None:
        lines = [
            f"Device: {self._device}",
            f"Frame:  {ctx.index}",
            f"FPS:    {ctx.fps:.1f}",
            f"Players:{len(ctx.people.players)}",
        ]
        y = 25
        for line in lines:
            cv2.putText(ctx.display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
