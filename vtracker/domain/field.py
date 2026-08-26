"""Volleyball court/net domain constants and the field value object.

The real-world reference points used for homography were previously hard-coded as
numpy arrays inside ``AppConfig.__post_init__``. They are court geometry, not
configuration, so they live here in the domain layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Real court points (metres), ordered to match the image `court` points in
# field_config.json. 9m x 18m court with 3/6/9/12 m lines.
REAL_COURT_POINTS = np.array([
    [0.0, 18.0], [9.0, 18.0], [9.0, 0.0], [0.0, 0.0],
    [0.0, 12.0], [9.0, 12.0], [9.0, 6.0], [0.0, 6.0],
    [0.0, 9.0], [9.0, 9.0],
], dtype=np.float32)

# Net plane points (X across court, Z height); top of net at 2.43 m.
REAL_NET_POINTS = np.array([
    [0.0, 2.43], [9.0, 2.43], [9.0, 0.0], [0.0, 0.0],
], dtype=np.float32)

COURT_WIDTH_M = 9.0
COURT_LENGTH_M = 18.0
NET_HEIGHT_M = 2.43
# Vertical extent shown on the net side-view minimap (net height + headroom
# for a ball crossing well above the net).
NET_MINIMAP_HEIGHT_M = 8.43


@dataclass
class VolleyballField:
    """Image-space field geometry loaded from config (court/net polygons)."""

    court: list                 # >= 4 boundary points (image px)
    net: list                   # 4 net points (image px)
    other: list | None = None

    def validate(self) -> None:
        assert len(self.court) >= 4, "Court needs >= 4 boundary points"
        assert len(self.net) == 4, "Net needs exactly 4 points"
