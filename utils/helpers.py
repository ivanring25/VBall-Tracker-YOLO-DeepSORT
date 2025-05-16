import cv2
import numpy as np
from typing import List, Tuple

class VideoUtils:
    @staticmethod
    def get_video_properties(video_path: str) -> Tuple[int, int, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return width, height, fps

class GeometryUtils:
    @staticmethod
    def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return inter_area / float(boxA_area + boxB_area - inter_area)

    @staticmethod
    def calculate_speed(pos_history: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(pos_history) < 2:
            return (0.0, 0.0)
        dx = pos_history[-1][0] - pos_history[-2][0]
        dy = pos_history[-1][1] - pos_history[-2][1]
        return (dx, dy)
    
    @staticmethod
    def get_center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def get_bbox_width(bbox):
        x1, _, x2, _ = bbox
        return int(abs(x2 - x1) / 2)

    @staticmethod
    def get_foot_position(bbox):
        x_center, _ = GeometryUtils.get_center_of_bbox(bbox)
        y_bottom = int(bbox[3])
        return (x_center, y_bottom)