from dataclasses import dataclass
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import os 

@dataclass
class AppConfig:
    # Paths
    model_path: str = "ball_for_cv.pt"
    video_path: str = "video.mp4"
    output_path: str = "output.mp4"
    field_config_path: str = r"C:\work_space\ww_project\project\data\field_config.json"
    
    # Processing parameters
    confidence_threshold: float = 0.85
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_size: tuple = (1280, 720)
    skip_frames: int = 1  # Добавлен новый параметр
    save_output: bool = True
    show_output: bool = True
    
    # Tracking parameters
    track_history_length: int = 10
    tracker_params: dict = None

    MODEL_PEOPLE: str =  "path/to/your/model.pt"
    STUB_PATH: str  = "C:\work_space\ww_project\tracks.pkl"
    CONFIDENCE_THRESHOLD: float = 0.2
    output_dir: str = r"C:\work_space\ww_project\project\data"
    # Visualization
    box_thickness: int = 2
    text_scale: float = 0.7
    text_thickness: int = 1
    track_colors: dict = None
    mini_map: bool = True 

    # Field dimensions (used for minimap scaling)
    field_width = 450
    field_height = 900 
    
    # Field configuration
    field_points: Dict[str, List[Tuple[int, int]]] = None
    field_colors: Dict[str, Tuple[int, int, int]] = None
    field_line_thickness: int = 1

    def __post_init__(self):
        self.track_colors = {
            'track': (255, 0, 0),
            'prediction': (0, 0, 255),
            'history': (0, 255, 255),
            'text': (0, 255, 0)
        }
        
        self.tracker_params = {
            "max_age": 15,
            "n_init": 2,
            "max_cosine_distance": 0.4,
            "nn_budget": 50
        }
         # Загрузка конфигурации поля
        if os.path.exists(self.field_config_path):
            with open(self.field_config_path, 'r') as f:
                tmp = json.load(f)
            self.field_points = tmp
        else:
            raise FileNotFoundError(f"Field config file not found: {self.field_config_path}")
        
        self.field_colors = {
            'court': (0, 255, 0),
            'net': (255, 255, 255),
            'markers': (0, 0, 255)
        }


        self.real_field_points = np.array([
      [0.0, 18.0],  # X=0м, Y=18м
      [9.0, 18.0],  # X=9м, Y=18м
      [9.0, 0.0],   # X=9м, Y=0м
      [0.0, 0.0],   # X=0м, Y=0м
      [0.0, 12.0],
      [9.0, 12.0],
      [9.0, 6.0],
      [0.0, 6.0],
      [0.0, 9.0],
      [9.0, 9.0]
  ], dtype=np.float32)
        
        self.net_real_points = np.array([
    [0.0, 2.43],   # X=0м, Y=9м, Z=2.43м (верхняя часть)
    [9.0, 2.43],   # X=9м, Y=9м, Z=2.43м (верхняя часть)
    [9.0, 0.0],    # X=9м, Y=9м, Z=0м (нижняя часть)
    [0.0, 0.0],    # X=0м, Y=9м, Z=0м (нижняя часть)
], dtype=np.float32)
