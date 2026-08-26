"""DirectYoloBallDetector — one full-frame YOLO *detection* pass, no motion
pre-processing.

The default ``YoloBallDetector`` runs a CPU pipeline first (MOG2 background
subtraction, morphology, contour shape/colour filtering) and then asks a YOLO
**classifier** whether each surviving ROI is a ball. That pre-processing is
only worth its cost if the model cannot find the ball itself.

If you have detection weights (boxes, not class probabilities), this detector
skips the whole CPU stage. Which one wins is an empirical question about your
weights — recall on small fast balls versus frames per second — so both are
available and the choice lives in the config (``detector.mode``).

Note the two are not interchangeable at the weights level: this needs a
detection model, the other needs a classifier.
"""

from __future__ import annotations

from ultralytics import YOLO

from vtracker.core.config import DetectorConfig
from vtracker.core.logging import get_logger
from vtracker.core.types import BBox, FrameBGR
from vtracker.domain.entities import BallDetection
from vtracker.infrastructure.detectors._runtime import inference_context, use_half, warmup

_log = get_logger("vtracker.detector")


class DirectYoloBallDetector:
    def __init__(self, cfg: DetectorConfig, device: str,
                 frame_size: tuple[int, int] = (1280, 720)) -> None:
        self._model = YOLO(cfg.model_path)
        self._conf = cfg.confidence_threshold
        self._device = device
        self._half = use_half(device)
        self._min_area = cfg.min_area
        self._max_area = cfg.max_area
        warmup(self._model, device, frame_size)
        _log.info("direct ball detector ready on %s (fp16=%s)", device, self._half)

    def detect(self, frame: FrameBGR) -> list[BallDetection]:
        with inference_context():
            result = self._model.predict(source=frame, conf=self._conf,
                                         device=self._device, half=self._half,
                                         verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            _log.warning("model returned no boxes — are these classifier "
                         "weights? use detector.mode: motion_roi instead")
            return []
        out: list[BallDetection] = []
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            # Same size sanity filter as the motion path, so config carries over.
            if not (self._min_area < w * h < self._max_area):
                continue
            out.append(BallDetection(box=BBox(x1, y1, w, h),
                                     confidence=float(box.conf[0])))
        return out
