"""Interactive tool to mark court/net reference points on a video frame and
save them as the ``field_points`` JSON consumed by ``HomographyProjector``
(see ``domain/field.py`` and ``configs/match.example.yaml:field_points_path``).

Native port of the old ``field_point/field_marker.FieldMarker``. Fixes the
original's hard-coded video path and hard-coded Windows config path
(``C:\\work_space\\ww_project\\...``, ``C:\\Users\\Ivan\\Downloads\\...``) —
everything now comes from CLI arguments, so it runs on any machine.

Usage:
    python -m vtracker.tools.field_marker --video match.mp4 \
        --frame 195 --width 1280 --height 720 --out data/field_config.json

Controls:
    1/2/3   select category: court / net / other
    click   add a point to the current category
    z       undo last point
    c       clear current category
    s       save to --out
    l       reload from --out
    q       quit (prints the collected points before exiting)
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

_CATEGORIES = ("court", "net", "other")
_COLORS = {"court": (0, 255, 0), "net": (0, 0, 255), "other": (128, 128, 128)}
_KEYS = {ord("1"): "court", ord("2"): "net", ord("3"): "other"}


class FieldMarker:
    def __init__(self, video_path: str, out_path: str, frame_number: int = 1,
                 target_size: tuple[int, int] | None = None) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames and frame_number >= total_frames:
            cap.release()
            raise ValueError(f"Video only has {total_frames} frames")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, self.image = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Could not read the requested frame")

        if target_size:
            h, w = self.image.shape[:2]
            if (w, h) != target_size:
                self.image = cv2.resize(self.image, target_size)

        self.out_path = out_path
        self.points: dict[str, list[list[int]]] = {c: [] for c in _CATEGORIES}
        self.current_category = "court"

        cv2.namedWindow("Field Marker")
        cv2.setMouseCallback("Field Marker", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.current_category].append([x, y])

    def _draw(self) -> np.ndarray:
        img = self.image.copy()
        for category, pts in self.points.items():
            color = _COLORS[category]
            for idx, (x, y) in enumerate(pts):
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.putText(img, f"{category[0]}{idx}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"Current: {self.current_category}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        help_lines = ["1/2/3: category", "click: add point", "z: undo",
                      "c: clear category", "s: save", "l: load", "q: quit"]
        y = 60
        for line in help_lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            y += 28
        return img

    def run(self) -> dict:
        while True:
            cv2.imshow("Field Marker", self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key in _KEYS:
                self.current_category = _KEYS[key]
            elif key == ord("z") and self.points[self.current_category]:
                self.points[self.current_category].pop()
            elif key == ord("c"):
                self.points[self.current_category] = []
            elif key == ord("s"):
                self.save()
            elif key == ord("l"):
                self.load()
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        self._print_summary()
        return self.points

    def save(self) -> None:
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(self.points, f, indent=2)
        print(f"Saved field points -> {self.out_path}")

    def load(self) -> None:
        try:
            with open(self.out_path, "r", encoding="utf-8") as f:
                self.points = json.load(f)
            print(f"Loaded field points <- {self.out_path}")
        except FileNotFoundError:
            print(f"No config found at {self.out_path}")

    def _print_summary(self) -> None:
        print("\nCollected field points:")
        for category, pts in self.points.items():
            print(f"  {category}: {pts}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mark court/net field points on a video frame")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--frame", type=int, default=1, help="Frame index to mark on")
    parser.add_argument("--width", type=int, default=None, help="Resize frame width")
    parser.add_argument("--height", type=int, default=None, help="Resize frame height")
    parser.add_argument("--out", required=True, help="Output field_points JSON path")
    args = parser.parse_args(argv)

    size = (args.width, args.height) if args.width and args.height else None
    marker = FieldMarker(args.video, args.out, frame_number=args.frame, target_size=size)
    marker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
