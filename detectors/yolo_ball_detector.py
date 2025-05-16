import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
from typing import List, Tuple
from .base_detector import BaseDetector

class YOLOBallDetector(BaseDetector):
    """YOLO-based volleyball detector with motion analysis"""
    
    def __init__(self, config):
        self.model = YOLO(config.model_path)
        self.conf_thresh = config.confidence_threshold
        self.min_area = 100
        self.max_area = 10000
        self.device = config.device
        self.frame_size = config.frame_size
        
        # Motion detection setup
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=70, detectShadows=False)
        self.prev_frame = None
        self.motion_level = 0
        
        # HSV color range for ball detection
        self.lower_color = np.array([10, 50, 115])
        self.upper_color = np.array([70, 255, 255])
        
        logging.info(f"Initialized YOLO detector on {self.device}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        
        # Motion analysis
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        fgmask = self.fgbg.apply(gray)
        fgmask = cv2.addWeighted(fgmask, 0.7, frame_delta, 0.3, 0)
        fgmask = cv2.medianBlur(fgmask, 5)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        fgmask = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)[1]
        
        self.prev_frame = gray.copy()
        return fgmask

    def _merge_boxes(self, boxes: List[Tuple]) -> List[Tuple]:
        """Merge overlapping bounding boxes"""
        if not boxes:
            return []
            
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [list(boxes[0])]
        
        for current in boxes[1:]:
            last = merged[-1]
            x_overlap = max(0, min(last[0]+last[2], current[0]+current[2]) - max(last[0], current[0]))
            y_overlap = max(0, min(last[1]+last[3], current[1]+current[3]) - max(last[1], current[1]))
            
            if x_overlap * y_overlap > 0.2 * (last[2]*last[3] + current[2]*current[3]):
                # Merge boxes
                last[0] = min(last[0], current[0])
                last[1] = min(last[1], current[1])
                last[2] = max(last[0]+last[2], current[0]+current[2]) - last[0]
                last[3] = max(last[1]+last[3], current[1]+current[3]) - last[1]
            else:
                merged.append(list(current))
        return merged

    def _filter_contours(self, frame: np.ndarray, fgmask: np.ndarray) -> List[Tuple]:
        """Find and filter contours based on shape characteristics"""
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Shape validation
            if 0.4 <= aspect_ratio <= 2.5:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)
                rect_fill_ratio = area / (w * h)
                
                if 0.5 < circularity <= 1.4 and 0.5 < rect_fill_ratio <= 1.1:
                    # Color validation
                    ball_roi = hsv[y:y+h, x:x+w]
                    mask = cv2.inRange(ball_roi, self.lower_color, self.upper_color)
                    color_ratio = cv2.countNonZero(mask) / (w * h)
                    
                    if color_ratio >= 0.15:
                        # Adaptive ROI expansion
                        base_size = max(w, h)
                        scale_factor = max(1.0, 40 / base_size)
                        padding = int(scale_factor * base_size * 0.2)
                        
                        x1 = max(x - padding, 0)
                        y1 = max(y - padding, 0)
                        x2 = min(x + w + padding, frame.shape[1])
                        y2 = min(y + h + padding, frame.shape[0])
                            
                        valid_boxes.append((x1, y1, x2-x1, y2-y1))
        
        return self._merge_boxes(valid_boxes)

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Main detection pipeline"""
        fgmask = self.preprocess(frame)
        boxes = self._filter_contours(frame, fgmask)
        
        # Prepare ROIs for classification
        rois = []
        for (x, y, w, h) in boxes:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            rois.append(cv2.resize(roi, (64, 64)))
        
        # YOLO classification
        detections = []
        if rois:
            results = self.model(rois, device=self.device, verbose=False)
            
            for i, r in enumerate(results):
                if r.probs is None:
                    continue
                
                conf = r.probs.top1conf.item()
                cls_id = r.probs.top1
                
                if cls_id == 0 and conf >= self.conf_thresh:  # 0 - ball class
                    x, y, w, h = boxes[i]
                    detections.append((x, y, w, h, conf))
        
        logging.debug(f"Detected {len(detections)} ball candidates")
        return detections

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()