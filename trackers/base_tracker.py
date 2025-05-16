from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections, frame):
        pass
    
    @abstractmethod
    def get_tracks(self):
        pass