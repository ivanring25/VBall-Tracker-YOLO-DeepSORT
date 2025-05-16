from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from .base_tracker import BaseTracker
import time

class DeepSortBallTracker(BaseTracker):
    def __init__(self, config):
        self.max_age = config.tracker_params["max_age"]
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=config.tracker_params["n_init"],
            max_cosine_distance=config.tracker_params["max_cosine_distance"],
            nn_budget=config.tracker_params["nn_budget"], 
            embedder_gpu=True
        )
        
        self.track_history = defaultdict(lambda: {
            'positions': deque(maxlen=config.track_history_length),
            'timestamps': deque(maxlen=config.track_history_length),
            'speeds': deque(maxlen=5),  
            'last_seen': 0,
            'active': False
        })
        self.frame_count = 0
        self.active_tracks = set()

    def update(self, detections: List[Tuple], frame: np.ndarray) -> List:
        self.frame_count += 1
        current_time = time.time()
        
        # Конвертация детекций в формат DeepSORT
        ds_detections = [
            ([float(x), float(y), float(w), float(h)], float(conf), 'ball')
            for x, y, w, h, conf in detections
        ]
        
        # Обновление трекера с учетом экстраполяции
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        # Обновление истории треков
        current_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            current_ids.add(track_id)
            ltrb = track.to_ltrb()
            cx = (ltrb[0] + ltrb[2]) / 2
            cy = (ltrb[1] + ltrb[3]) / 2

            history = self.track_history[track_id]
            
            # Расчет скорости с учетом временного интервала
            if len(history['positions']) > 0:
                prev_pos = history['positions'][-1]
                dt = current_time - history['timestamps'][-1]
                
                if dt > 1e-3:  # Защита от деления на ноль
                    dx = cx - prev_pos[0]
                    dy = cy - prev_pos[1]
                    speed_px = np.sqrt(dx**2 + dy**2) / dt
                    history['speeds'].append(speed_px)
                else:
                    history['speeds'].append(0.0)

            # Обновление данных трека
            history['positions'].append((cx, cy))
            history['timestamps'].append(current_time)
            history['last_seen'] = self.frame_count
            history['active'] = True

        # Удаление устаревших треков с учетом max_age
        to_delete = [tid for tid, data in self.track_history.items()
                    if self.frame_count - data['last_seen'] > self.max_age]
        
        for tid in to_delete:
            del self.track_history[tid]
            if tid in self.active_tracks:
                self.active_tracks.remove(tid)

        self.active_tracks = current_ids
        return tracks

    def get_tracks(self) -> Dict[int, Dict[str, Any]]:
        return self.track_history

    def _remove_expired_tracks(self) -> None:
        """Автоматическая очистка с использованием настроенного max_age"""
        to_remove = [
            tid for tid, data in self.track_history.items()
            if self.frame_count - data['last_seen'] > self.max_age
        ]
        for tid in to_remove:
            del self.track_history[tid]

    def get_speed(self, track_id: int) -> float:
        """Возвращает сглаженную скорость с использованием медианы"""
        history = self.track_history.get(track_id)
        if not history or not history['speeds']:
            return 0.0
        return float(np.median(history['speeds']))

    def get_track_speed(self, track_id: int) -> Tuple[float, float]:
        """Рассчитывает мгновенную скорость по последним двум позициям"""
        positions = self.track_history[track_id]['positions']
        if len(positions) < 2:
            return (0.0, 0.0)
        
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        return (dx, dy)

    def __del__(self):
        if hasattr(self, 'tracker'):
            del self.tracker