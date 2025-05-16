# utils/field_marker.py
import cv2
import json
import numpy as np

class FieldMarker:
    def __init__(self, video_path, frame_number=1, config_path=r'C:\work_space\ww_project\project\data\field_config.json',target_width=None, target_height=None):
        # Загружаем указанный кадр из видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео файл")
        # Устанавливаем целевое разрешение

        self.target_size = (target_width, target_height)

        if target_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        if target_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        # Проверяем общее количество кадров
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number >= total_frames:
            cap.release()
            raise ValueError(f"Видео содержит только {total_frames} кадров")
        
        # Устанавливаем и читаем нужный кадр
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, self.image = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Не удалось прочитать указанный кадр из видео")
        

        # Принудительно меняем размер если нужно
        if self.target_size != (None, None):
            h, w = self.image.shape[:2]
            if (w, h) != self.target_size:
                self.image = cv2.resize(self.image, self.target_size)


        self.config_path = config_path
        self.points = {
            'court': [],
            'net': [],
            'other': []
        }
        self.current_category = 'court'
        self.colors = {
            'court': (0, 255, 0),
            'net': (0, 0, 255),
            'other': (128, 128, 128)
        }
        
        cv2.namedWindow('Field Marker')
        cv2.setMouseCallback('Field Marker', self.mouse_callback)
    
    # Остальные методы остаются без изменений
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.current_category].append((x, y))
            self.draw()

    def draw(self):
        img = self.image.copy()
        
        # Рисуем все точки
        for category, points in self.points.items():
            for idx, (x, y) in enumerate(points):
                color = self.colors[category]
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.putText(img, f"{category[:1]}{idx}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Показываем текущую категорию
        cv2.putText(img, f"Current: {self.current_category}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Инструкция
        help_text = [
            "1-3: Select category",
            "S: Save config",
            "L: Load config",
            "Z: Undo last point",
            "C: Clear category",
            "Q: Quit"
        ]
        y_start = 60
        for line in help_text:
            cv2.putText(img, line, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1)
            y_start += 30
            
        cv2.imshow('Field Marker', img)

    def run(self):
        while True:
            self.draw()
            key = cv2.waitKey(1) & 0xFF
            
            # Выбор категории
            if key == ord('1'): self.current_category = 'court'
            if key == ord('2'): self.current_category = 'net'
            if key == ord('3'): self.current_category = 'other'
            
            # Управление
            if key == ord('z'):
                if self.points[self.current_category]:
                    self.points[self.current_category].pop()
            if key == ord('c'):
                self.points[self.current_category] = []
            if key == ord('s'):
                self.save_config()
            if key == ord('l'):
                self.load_config()
            if key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        self.print_config_code()
        
    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.points, f, indent=2)
        print(f"Config saved to {self.config_path}")

    def load_config(self):
        try:
            with open(self.config_path) as f:
                self.points = json.load(f)
            print(f"Config loaded from {self.config_path}")
        except FileNotFoundError:
            print("Config file not found!")

    def print_config_code(self):
        print("\nField configuration for your code:")
        print("VOLLEYBALL_FIELD_CONFIG = {")
        for category, points in self.points.items():
            print(f"    '{category}': {points},")
        print("}")

if __name__ == "__main__":
    # Укажите путь к вашему видео файлу
    video_path = r"C:\Users\Ivan\Downloads\Отыгрались с 12-19, взяли партию уступая 17-23 СРБ - СУЭК.mp4"
    marker = FieldMarker(video_path, frame_number=195, target_width=1280,target_height=720)
    marker.run()