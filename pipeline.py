import cv2
import time
import logging
import numpy as np
from typing import Any, Dict, List
from utils.config import AppConfig
from detectors.yolo_ball_detector import YOLOBallDetector
from trackers.deepsort_tracker import DeepSortBallTracker
from trackers.people_tracker import PeopleTracker
from homography.homography import HomographyProjector
from utils.visualizer import Visualizer
from utils.logger import AppLogger
from utils.field import VolleyballField  # Добавлен импорт

class BallTrackingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = AppLogger(config).get_logger()
        self.frame_count = 0
        self.processed_frames = 0
        self.start_time = time.time()

        # Инициализация компонентов
        self.detector = YOLOBallDetector(config)
        self.tracker = DeepSortBallTracker(config)
        self.visualizer = Visualizer(config)
        self.people_tracker = PeopleTracker(config)
        self.homog = self._init_homog()
        self.field = self._init_field()  # Инициализация поля
        
        # Инициализация видео потоков
        self.video_processor = self._init_video_processor()
        self.fps_history = []

    def _init_homog(self) -> HomographyProjector:
        net_image_points = np.array(self.config.field_points['net'][:2] +self.config.field_points['court'][9:] + self.config.field_points['court'][8:9], dtype=np.float32)
        return HomographyProjector(court_image_points=np.array(self.config.field_points['court'], dtype=np.float32), court_real_points=self.config.real_field_points, net_image_points=net_image_points, net_real_points=self.config.net_real_points)
    

    def _init_field(self) -> VolleyballField:
        """Инициализация конфигурации поля"""
        return VolleyballField(
            court=self.config.field_points['court'],
            net=self.config.field_points['net']
        )
    
    def _init_video_processor(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.config.video_path}")

        writer = None
        if self.config.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                self.config.output_path,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                self.config.frame_size
            )

        return {'cap': cap, 'writer': writer}

    def run(self) -> None:
        cap = self.video_processor['cap']
        writer = self.video_processor['writer']
        
        try:
            while True:
                start_time = time.time()
                
                # Чтение кадра
                ret, frame = cap.read()
                if not ret:
                    break

                # Пропуск кадров для ускорения
                self.frame_count += 1
                if self.frame_count % self.config.skip_frames != 0:
                    continue

                # Предобработка
                frame = cv2.resize(frame, self.config.frame_size)
                display_frame = frame.copy()

                # Детекция
                detections = self._process_detections(frame)

                # Трекинг
                tracks = self.tracker.update(detections, frame)
                tracks_people = self.people_tracker.process_frame(frame)

                # Визуализация
                display_frame = self._visualize_frame(display_frame, None, tracks_people)

                # Запись и отображение
                self._handle_output(writer, display_frame)

                # Обновление метрик
                self._update_metrics(start_time)

        finally:
            self._release_resources()
            self._log_final_stats()

    def _process_detections(self, frame: np.ndarray) -> List:
        try:
            return self.detector.detect(frame)
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return []

    def _visualize_frame(self, frame: np.ndarray, tracks: List, tracks_people) -> np.ndarray:
        # Отрисовка треков и информации
        frame = self.visualizer.draw_tracks(frame, self.tracker.track_history, self.tracker)
        frame = self.visualizer.draw_annotations(frame, tracks_people)
        frame = self.visualizer.draw_fps(frame, self._calculate_fps())

        frame = self.visualizer.draw_system_info(frame, {
            'Device': self.config.device,
            'Frame': self.frame_count,
            'Tracks': len(self.tracker.active_tracks)
        })
        frame = self.visualizer.draw_field(frame)

        if self.config.mini_map:
            frame = self.visualizer.draw_minimap_net(frame, self.tracker.track_history, self.homog, self.logger)
            frame = self.visualizer.draw_minimap_court(frame, tracks_people, self.homog, self.logger)
        frame = self.visualizer.draw_legend(frame)
        return frame

    def _handle_output(self, writer: cv2.VideoWriter, frame: np.ndarray) -> None:
        if writer is not None:
            writer.write(frame)
            
        if self.config.show_output:
            cv2.imshow("Ball Tracking", frame)
            if cv2.waitKey(1) == 27:  # ESC для выхода
                
                raise KeyboardInterrupt

    def _update_metrics(self, start_time: float) -> None:
        self.processed_frames += 1
        self.fps_history.append(1.0 / (time.time() - start_time))
        
        if self.processed_frames % 10 == 0:
            self.logger.info(
                f"Processed {self.processed_frames} frames | "
                f"Current FPS: {self.fps_history[-1]:.1f}"
            )

    def _calculate_fps(self) -> float:
        if len(self.fps_history) == 0:
            return 0.0
        return np.mean(self.fps_history[-10:])

    def _release_resources(self) -> None:
        self.video_processor['cap'].release()
        if self.video_processor['writer'] is not None:
            self.video_processor['writer'].release()
        cv2.destroyAllWindows()

        #self.detector.export_detections(r"data_output\detections_log.json")

    def _log_final_stats(self) -> None:
        total_time = time.time() - self.start_time
        avg_fps = self.processed_frames / total_time if total_time > 0 else 0

        self.logger.info("\n=== Final Statistics ===")
        self.logger.info(f"Total frames processed: {self.frame_count}")
        self.logger.info(f"Processed frames: {self.processed_frames}")
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info(f"Average FPS: {avg_fps:.1f}")
        self.logger.info(f"Max FPS: {max(self.fps_history) if self.fps_history else 0:.1f}")
        self.logger.info(f"Output saved to: {self.config.output_path}")

if __name__ == "__main__":
    config = AppConfig(
        model_path=r"C:\work_space\ww_project\project\models\ball_cv_new_data.pt",
        video_path=r"C:\Users\Ivan\Downloads\Отыгрались с 12-19, взяли партию уступая 17-23 СРБ - СУЭК.mp4",
        output_path=r"C:\work_space\ww_project\project\video_output\output.mp4",
        MODEL_PEOPLE = r"C:\work_space\ww_project\people_detector\people_detector.pt",

        confidence_threshold=0.96,
        frame_size=(1280,720),
        skip_frames=1,
        save_output=True,
        show_output= True,
        mini_map = True,

        tracker_params={
            "max_age": 15,
            "n_init": 2,
            "max_cosine_distance": 0.4,
            "nn_budget": 10,
        }

    )
    
    try:
        pipeline = BallTrackingPipeline(config)
        pipeline.run()
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        raise