"""Per-frame detection/track log in JSON.

The old pipeline built a detections log but the write-out was commented out in
``_release_resources``, so a run produced nothing but a rendered video. This
exporter records the structured results so they can be analysed offline.

Written incrementally as JSON Lines (one object per frame) so a long run is
streamed to disk instead of held in memory, and an interrupted run still leaves
valid, readable output.
"""

from __future__ import annotations

import json
from typing import TextIO

from vtracker.pipeline.context import FrameContext


class JsonDetectionExporter:
    """FrameExporter-compatible sink that records context data, not pixels."""

    def __init__(self, path: str) -> None:
        self._path = path
        # Long-lived handle: the pipeline owns it for the whole run and closes
        # it in its finally block (or via the context-manager methods below).
        self._fh: TextIO | None = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def __enter__(self) -> JsonDetectionExporter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def write_context(self, ctx: FrameContext) -> None:
        record = {
            "frame": ctx.index,
            "fps": round(ctx.fps, 2),
            "ball_detections": [
                {
                    "x": round(d.box.x, 2), "y": round(d.box.y, 2),
                    "w": round(d.box.w, 2), "h": round(d.box.h, 2),
                    "confidence": round(d.confidence, 4),
                    "estimated": d.estimated,
                }
                for d in ctx.detections
            ],
            "ball_tracks": {
                tid: {
                    "position": [round(v, 2) for v in state.positions[-1]],
                    "speed": round(ctx.ball_speeds.get(tid, 0.0), 2),
                }
                for tid, state in ctx.ball_tracks.items()
                if state.active and state.positions
            },
            "players": [
                {
                    "track_id": p.track_id, "team": p.team,
                    "bbox": [round(v, 2) for v in p.box.ltrb],
                }
                for p in ctx.people.players.values()
            ],
            "referees": [
                {
                    "track_id": r.track_id,
                    "bbox": [round(v, 2) for v in r.box.ltrb],
                }
                for r in ctx.people.referees.values()
            ],
        }
        assert self._fh is not None, "exporter already closed"
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # FrameExporter Protocol: the pipeline's ExportStage passes a frame, but
    # this sink needs the whole context, so it is wired via ExportContextStage.
    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
