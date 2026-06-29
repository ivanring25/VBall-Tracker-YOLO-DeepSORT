"""Volleyball detector: background-motion mask -> contour shape/colour filter
-> YOLO ROI classification. Native port of the old
``detectors/yolo_ball_detector.YOLOBallDetector``; returns typed
``BallDetection`` and implements the ``BallDetector`` Protocol.

Perf note (audit §7.1 #3): HSV conversion is done per-ROI, not over the whole
frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from vtracker.core.config import DetectorConfig
from vtracker.core.logging import get_logger
from vtracker.core.types import BBox, FrameBGR
from vtracker.domain.entities import BallDetection

_log = get_logger("vtracker.detector")
_LOWER_HSV = np.array([10, 50, 115])
_UPPER_HSV = np.array([70, 255, 255])


class YoloBallDetector:
    def __init__(self, cfg: DetectorConfig, device: str) -> None:
        self._model = YOLO(cfg.model_path)
        self._conf = cfg.confidence_threshold
        self._min_area = cfg.min_area
        self._max_area = cfg.max_area
        self._device = device
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=5, varThreshold=70, detectShadows=False)
        self._prev_gray: np.ndarray | None = None
        _log.info("ball detector ready on %s", device)

    # --- motion preprocessing ---------------------------------------------
    def _motion_mask(self, frame: FrameBGR) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros_like(gray)
        delta = cv2.absdiff(self._prev_gray, gray)
        mask = self._fgbg.apply(gray)
        mask = cv2.addWeighted(mask, 0.7, delta, 0.3, 0)
        mask = cv2.medianBlur(mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]
        self._prev_gray = gray.copy()
        return mask

    # --- contour candidate filtering --------------------------------------
    def _candidate_boxes(self, frame: FrameBGR, mask: np.ndarray) -> list[tuple]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self._min_area < area < self._max_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not (0.4 <= w / float(h) <= 2.5):
                continue
            circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)
            fill = area / (w * h)
            if not (0.5 < circularity <= 1.4 and 0.5 < fill <= 1.1):
                continue
            # Colour check on the ROI only (cheap vs whole-frame HSV).
            roi_hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
            colour_ratio = cv2.countNonZero(
                cv2.inRange(roi_hsv, _LOWER_HSV, _UPPER_HSV)) / (w * h)
            if colour_ratio < 0.15:
                continue
            base = max(w, h)
            pad = int(max(1.0, 40 / base) * base * 0.2)
            x1, y1 = max(x - pad, 0), max(y - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            y2 = min(y + h + pad, frame.shape[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return _merge_boxes(boxes)

    # --- public API --------------------------------------------------------
    def detect(self, frame: FrameBGR) -> list[BallDetection]:
        mask = self._motion_mask(frame)
        boxes = self._candidate_boxes(frame, mask)
        rois, kept = [], []
        for (x, y, w, h) in boxes:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            rois.append(cv2.resize(roi, (64, 64)))
            kept.append((x, y, w, h))
        if not rois:
            return []
        results = self._model(rois, device=self._device, verbose=False)
        out: list[BallDetection] = []
        for (x, y, w, h), r in zip(kept, results):
            if r.probs is None:
                continue
            conf = r.probs.top1conf.item()
            if r.probs.top1 == 0 and conf >= self._conf:  # class 0 == ball
                out.append(BallDetection(box=BBox(x, y, w, h), confidence=conf))
        return out


def _merge_boxes(boxes: list[tuple]) -> list[tuple]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for cur in boxes[1:]:
        last = merged[-1]
        xo = max(0, min(last[0] + last[2], cur[0] + cur[2]) - max(last[0], cur[0]))
        yo = max(0, min(last[1] + last[3], cur[1] + cur[3]) - max(last[1], cur[1]))
        if xo * yo > 0.2 * (last[2] * last[3] + cur[2] * cur[3]):
            last[0] = min(last[0], cur[0])
            last[1] = min(last[1], cur[1])
            last[2] = max(last[0] + last[2], cur[0] + cur[2]) - last[0]
            last[3] = max(last[1] + last[3], cur[1] + cur[3]) - last[1]
        else:
            merged.append(list(cur))
    return [tuple(b) for b in merged]
