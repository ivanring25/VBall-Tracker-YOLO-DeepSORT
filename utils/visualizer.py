import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict, deque
from .config import AppConfig
from utils.helpers import GeometryUtils

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.colors = config.track_colors
        self.speed_colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255), # Yellow
            'high': (0, 0, 255)      # Red
        }
        self.speed_thresholds = (50, 150)  # px/s
        self.geometry_utils = GeometryUtils()

    def _draw_text_with_background(self, frame, text, position, font_scale, color, thickness, bg_color, padding=5):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = position
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            bg_color,
            -1
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        for (x, y, w, h, conf) in detections:
            cv2.rectangle(
                frame,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                self.colors['track'],
                self.config.box_thickness
            )
            self._draw_text_with_background(
                frame,
                f"{conf:.2f}",
                (int(x), int(y) - 5),
                self.config.text_scale,
                self.colors['text'],
                self.config.text_thickness,
                (40, 40, 40)
            )
        return frame

    def draw_tracks(self, frame: np.ndarray, tracks: Dict, tracker: Any) -> np.ndarray:
        for track_id, data in tracks.items():
            if not data['active'] or len(data['positions']) < 2:
                continue

            self._draw_track_history(frame, data['positions'])
            last_pos = data['positions'][-1]
            speed = tracker.get_speed(track_id)
            self._draw_track_info(frame, track_id, last_pos, speed)
            self._draw_movement_direction(frame, data['positions'])

        return frame

    def _draw_track_history(self, frame: np.ndarray, positions: deque) -> None:
        for i in range(1, len(positions)):
            alpha = i / len(positions)
            base_color = np.array(self.colors['history'], dtype=np.uint8)
            faded_color = tuple((base_color * alpha).astype(int).tolist())

            cv2.line(
                frame,
                tuple(map(int, positions[i - 1])),
                tuple(map(int, positions[i])),
                faded_color,
                2,
                cv2.LINE_AA
            )

    def _draw_track_info(self, frame: np.ndarray, track_id: int, position: Tuple[float, float], speed: float) -> None:
        x, y = position
        speed_color = self._get_speed_color(speed)
        text = f"ID: {track_id} | {speed:.1f} px/s"
        self._draw_text_with_background(
            frame,
            text,
            (int(x), int(y) - 10),
            self.config.text_scale,
            speed_color,
            self.config.text_thickness,
            (30, 30, 30)
        )

    def _draw_movement_direction(self, frame: np.ndarray, positions: List) -> None:
        if len(positions) < 2:
            return

        pos_list = list(positions)
        start = tuple(map(int, pos_list[-2]))
        end = tuple(map(int, pos_list[-1]))

        cv2.arrowedLine(
            frame,
            start,
            end,
            self.colors['prediction'],
            2,
            tipLength=0.3
        )

    def _get_speed_color(self, speed: float) -> Tuple[int, int, int]:
        if speed < self.speed_thresholds[0]:
            return self.speed_colors['low']
        elif speed < self.speed_thresholds[1]:
            return self.speed_colors['medium']
        return self.speed_colors['high']

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        self._draw_text_with_background(
            frame,
            text,
            (10, 25),
            0.6,
            (0, 255, 0),
            1,
            (20, 20, 20)
        )
        return frame

    def draw_system_info(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        y_offset = 50
        for key, value in info.items():
            text = f"{key}: {value}"
            self._draw_text_with_background(
                frame,
                text,
                (10, y_offset),
                0.5,
                (255, 255, 255),
                1,
                (50, 50, 50)
            )
            y_offset += 25
        return frame

    def draw_field(self, frame):
        boundary = np.array(self.config.field_points['court'][:4], np.int32)
        cv2.polylines(frame, [boundary], True,
                      self.config.field_colors['court'],
                      self.config.field_line_thickness)

        for point in self.config.field_points['court']:
            cv2.circle(frame, point, 5,
                       self.config.field_colors['markers'], -1)

        net_rect = self.config.field_points['net']
        cv2.rectangle(frame, net_rect[0], net_rect[2],
                      self.config.field_colors['net'],
                      self.config.field_line_thickness)

        for point in self.config.field_points['net']:
            cv2.circle(frame, point, 2,
                       self.config.field_colors['markers'], -1)
            
        cv2.line(frame, self.config.field_points['court'][4], self.config.field_points['court'][5], 
                 self.config.field_colors['court'], self.config.field_line_thickness)
        
        cv2.line(frame, self.config.field_points['court'][6], self.config.field_points['court'][7], 
                self.config.field_colors['court'], self.config.field_line_thickness)
        
        cv2.line(frame, self.config.field_points['court'][8], self.config.field_points['court'][9], 
                self.config.field_colors['court'], self.config.field_line_thickness)
        
        return frame

    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        y_start = frame.shape[0] - 170
        box_width = 300
        box_height = 140

        # Создаем прозрачный overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_start), (10 + box_width, y_start + box_height), (30, 30, 30), -1)

        # Альфа-смешивание
        alpha = 0.5  # прозрачность: 0 — полностью прозрачный, 1 — непрозрачный
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Обводка рамки (непрозрачная)
        cv2.rectangle(frame, (10, y_start), (10 + box_width, y_start + box_height), (70, 70, 70), 1)

        # Текст
        cv2.putText(
            frame,
            "Speed Legend",
            (20, y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y = y_start + 55
        labels = {
            'low': f"Low (<{self.speed_thresholds[0]} px/s)",
            'medium': f"Medium (<{self.speed_thresholds[1]} px/s)",
            'high': f"High (>{self.speed_thresholds[1]} px/s)"
        }
        for level, label in labels.items():
            cv2.circle(frame, (25, y), 6, self.speed_colors[level], -1)
            cv2.putText(
                frame,
                label,
                (40, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA
            )
            y += 30

        return frame
    
    def create_net_minimap_sideview(self, minimap_width=200, minimap_height=300, padding=20) -> np.ndarray:
        field_width = 9.0        # ширина по X
        total_height = 8.43      # общая высота (Z): до 6 м над сеткой
        net_height = 2.43        # высота сетки

        canvas_width = minimap_width + 2 * padding
        canvas_height = minimap_height + 2 * padding

        minimap = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)  # 4 канала (RGBA)
        minimap[:] = (30, 30, 30, 255)  # фон с альфа-каналом

        # --- Координаты нижней и верхней рамки (только до 2.43 м) ---
        z0_y = padding + minimap_height
        z_net_y = padding + int((1 - net_height / total_height) * minimap_height)

        # --- Рамка только до высоты сетки (по Z = 2.43 м) ---
        cv2.rectangle(minimap, (padding, z_net_y), (padding + minimap_width, z0_y), (255, 255, 255, 255), 1)

        # --- Линия сетки (Z = 2.43 м) ---
        cv2.line(minimap, (padding, z_net_y), (padding + minimap_width, z_net_y), (200, 200, 0, 255), 1)

        # --- Горизонтальные линии через каждый метр ---
        for z_m in range(1, int(total_height) + 1):
            y_px = padding + int((1 - z_m / total_height) * minimap_height)
            cv2.line(minimap, (padding, y_px), (padding + minimap_width, y_px), (60, 60, 60, 255), 1)
            # Подписи высот слева
            cv2.putText(minimap, f"{z_m}m", (5, y_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return minimap


    def draw_minimap_net(self, frame: np.ndarray, tracks: dict, projector, logger) -> np.ndarray:
        minimap_width = 150
        minimap_height = 175
        padding = 20

        # Создаем миникарту с альфа-каналом
        minimap = self.create_net_minimap_sideview(minimap_width, minimap_height, padding)

        # Отрисовываем траектории на миникарте
        field_width = 9.0        # по X
        total_height = 8.43      # по Z

        for track_id, data in tracks.items():
            if not data['active'] or len(data['positions']) < 2:
                continue

            projected_points = []
            for px, py in data['positions']:
                try:
                    x, z = projector.project_point((px, py), plane='net')
                except Exception:
                    continue

                mx = padding + int((x / field_width) * minimap_width)
                mz = padding + int((1 - z / total_height) * minimap_height)
                projected_points.append((mx, mz))

            for i in range(1, len(projected_points)):
                cv2.line(minimap, projected_points[i - 1], projected_points[i], (0, 255, 0, 255), 1)

            if projected_points:
                cv2.circle(minimap, projected_points[-1], 3, (0, 0, 255, 255), -1)

        # Установка прозрачности
        alpha = 0.5  # прозрачность: 0 — полностью прозрачный, 1 — непрозрачный

        # Наложение миникарты на кадр с использованием прозрачности
        h, w = frame.shape[:2]
        h_minimap, w_minimap = minimap.shape[:2]
        top_left_y = 20
        top_left_x = w - w_minimap - 20

        # Создаем временное изображение для наложения
        overlay = frame.copy()

        # Накладываем миникарту на изображение с заданной прозрачностью
        overlay[top_left_y:top_left_y + h_minimap, top_left_x:top_left_x + w_minimap] = minimap[:, :, :3]  # без альфа-канала

        # Объединяем кадр и миникарту с прозрачностью
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame
    
    def create_field_minimap_base(self, minimap_size=200, padding=20) -> np.ndarray:
        field_width = 9.0    # по оси X
        field_length = 18.0  # по оси Y

        canvas_size = minimap_size + 2 * padding
        minimap = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        minimap[:] = (30, 30, 30)  # тёмный фон

        # --- Отрисовка границы поля (белый прямоугольник) ---
        top_left = (padding, padding)
        bottom_right = (padding + minimap_size, padding + minimap_size)
        cv2.rectangle(minimap, top_left, bottom_right, (255, 255, 255), 1)

        # --- Горизонтальные линии на 6, 9 и 12 метрах ---
        for y_meter in [6.0, 9.0, 12.0]:
            y_px = padding + int(((field_length - y_meter) / field_length) * minimap_size)
            cv2.line(minimap, (padding, y_px), (padding + minimap_size, y_px), (80, 80, 80), 1)

        return minimap


    def draw_minimap_court(self, frame: np.ndarray, tracks, projector, logger) -> np.ndarray:
        minimap_width = 150
        minimap_height = 150
        padding = 15

        # Создаём базу миникарты
        minimap = self.create_field_minimap_base(minimap_width, padding)

        players, referees = tracks
        field_width = 9.0    # по X
        field_length = 18.0  # по Y

        def draw_point_on_minimap(x_field, z_field, color, track_id=None):
            mx = padding + int((x_field / field_width) * minimap_width)
            mz = padding + int((1 - z_field / field_length) * minimap_height)
            cv2.circle(minimap, [mx, mz], 4, color, -1)
            if track_id is not None:
                cv2.putText(minimap, str(track_id), (mx + 5, mz - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Игроки
        for track_id, player in players.items():
            y2 = int(player["bbox"][3])
            x_center, _ = self.geometry_utils.get_center_of_bbox(player["bbox"])
            try:
                x, z = projector.project_point((x_center, y2), plane='field')
            except Exception:
                continue

            # Определяем цвет на основе команды игрока
            team_id = player.get("team")
            if team_id == 1:
                team_color = (0, 0, 255)  # Красный для первой команды
            elif team_id == 2:
                team_color = (255, 255, 0)  # Синий для второй команды
            else:
                team_color = (0, 255, 0)  # Зеленый для неопределенных игроков

            draw_point_on_minimap(x, z, team_color, track_id)

        # Судьи
        for track_id, referee in referees.items():
            y2 = int(referee["bbox"][3])
            x_center, _ = self.geometry_utils.get_center_of_bbox(referee["bbox"])
            try:
                x, z = projector.project_point((x_center, y2), plane='field')
            except Exception:
                continue
            draw_point_on_minimap(x, z, (0, 255, 255), track_id)  # Судьи остаются желтыми

        # Наложение миникарты
        alpha = 0.5
        h, w = frame.shape[:2]
        h_m, w_m = minimap.shape[:2]
        top_left_x = w - w_m - 20
        top_left_y = h - h_m - 20
        overlay = frame.copy()

        overlay[top_left_y:top_left_y + h_m, top_left_x:top_left_x + w_m] = minimap
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame


    
    def draw_ellipse(self, frame, bbox, color, track_id=None, label=None):
        y2 = int(bbox[3])
        x_center, _ = self.geometry_utils.get_center_of_bbox(bbox)
        width = self.geometry_utils.get_bbox_width(bbox)

        # Рисуем эллипс под игроком
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rect_w, rect_h = 80, 25
            x1_rect = x_center - rect_w // 2
            y1_rect = y2 + 15 - rect_h // 2
            y2_rect = y2 + 15 + rect_h // 2

            # Рисуем фон под текст
            cv2.rectangle(frame, (x1_rect, y1_rect), (x1_rect + rect_w, y2_rect), color, cv2.FILLED)

            # Подписываем ID и роль
            text = f"{track_id}"
            if label:
                text += f" {label}"

            cv2.putText(frame, text, (x1_rect + 5, y2_rect - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame
        

    def draw_annotations(self, frame, tracks):
        players, referees = tracks

        # Цвета для команд (BGR format)
        team_colors = {
            1: (0, 0, 255),    # Красный для первой команды
            2: (255, 0, 0),    # Синий для второй команды
            None: (0, 255, 0)  # Зеленый для неопределенных
        }

        # Рисуем игроков
        for track_id, player in players.items():
            # Получаем назначенный team_id
            team_id = player.get("team")
           
            # Выбираем цвет на основе команды
            color = team_colors.get(team_id)
           

            # Формируем метку с номером команды и ID трека
            label = f"T{team_id}-{track_id}" if team_id is not None else f"U-{track_id}"
            # Отрисовка эллипса с меткой
            frame = self.draw_ellipse(
                frame, 
                player["bbox"],
                color=color,
                track_id=track_id,
                label=label
            )

        # Рисуем судей (единый цвет)
        for track_id, referee in referees.items():
            frame = self.draw_ellipse(
                frame,
                referee["bbox"],
                color=(0, 255, 255),  # Желтый для судей
                track_id=track_id,
                label=f"Ref-{track_id}"
            )

        return frame
    
    