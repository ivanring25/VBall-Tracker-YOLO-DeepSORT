from abc import ABC, abstractmethod
import cv2

class BaseDetector(ABC):
    @abstractmethod
    def preprocess(self, frame):
        pass
    
    @abstractmethod
    def detect(self, frame):
        pass