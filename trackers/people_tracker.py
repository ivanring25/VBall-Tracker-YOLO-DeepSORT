from ultralytics import YOLO
import supervision as sv
from utils.helpers import GeometryUtils
from .base_tracker import BaseTracker
from team_detector.team_assigner import TeamAssigner  # Импортируем ваш TeamAssigner

class PeopleTracker(BaseTracker):
    def __init__(self, config):
        self.model = YOLO(config.MODEL_PEOPLE)
        self.tracker = sv.ByteTrack()
        self.conf = config.CONFIDENCE_THRESHOLD

        # История треков для игроков и судей
        self.tracks_history = {"Players": {}, "Referee": {}}

        self.team_assigner = TeamAssigner( )
        self.frame_count = 0
        self.initialized_teams = False

    def process_frame(self, frame):
        result = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]
        detection = sv.Detections.from_ultralytics(result)

        cls_names = result.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        tracked = self.tracker.update_with_detections(detection)

        players, referees = {}, {}
        player_detections = {}

        # Сбор bbox игроков
        for det in tracked:
            bbox, _, _, cls_id, track_id, extra_info = det

            if isinstance(track_id, (int, float)):
                track_id = str(track_id)

            if cls_id == cls_names_inv['Players']:
                bbox = bbox.tolist()
                player_detections[track_id] = {"bbox": bbox}

        # Если еще не инициализировали команды — делаем на первом кадре с игроками
        if not self.initialized_teams and len(player_detections) >= 12:
            print(f"len(player_detections): {len(player_detections)}")
            self.team_assigner.assign_team_color(frame, player_detections)
            self.initialized_teams = True
            print(f"Teams initialized. Colors: {self.team_assigner.team_colors}")

        # Определяем команду и либеро для каждого игрока
        for track_id, pdata in player_detections.items():
            bbox = pdata["bbox"]
            position = GeometryUtils.get_foot_position(bbox)

            team_id = self.team_assigner.get_player_team(frame, bbox, track_id) if self.initialized_teams else None

            player_info = {
                "bbox": bbox,
                "position": position,
                "team": team_id,
                "track_id": track_id,
                "is_libero": False
            }

            players[track_id] = player_info
            self.tracks_history["Players"][track_id] = player_info

        # Обработка судей
        for det in tracked:
            bbox, _, _, cls_id, track_id, extra_info = det
            if isinstance(track_id, (int, float)):
                track_id = str(track_id)

            if cls_id == cls_names_inv['Referee']:
                bbox = bbox.tolist()
                position = GeometryUtils.get_foot_position(bbox)
                referee_info = {
                    "bbox": bbox,
                    "position": position
                }
                referees[track_id] = referee_info
                self.tracks_history["Referee"][track_id] = referee_info

        self.frame_count += 1
        return players, referees

    def get_tracks(self):
        return self.tracks_history

    def update(self):
        # Можно добавить очистку старых треков
        pass
