import logging
import sys
from typing import Dict, Any


class AppLogger:
    def __init__(self, config: Any):
        self.logger = logging.getLogger("BallTracker")
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler("tracker.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Enable/disable debug logging
        if config.device == "cuda":
            self.logger.propagate = False

    def get_logger(self) -> logging.Logger:
        return self.logger

    @staticmethod
    def log_config(config: Dict[str, Any]):
        logger = logging.getLogger("BallTracker")
        logger.info("Application Configuration:")
        for key, value in config.items():
            logger.info(f"{key:20}: {str(value):<}")