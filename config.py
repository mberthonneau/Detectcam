import json
from pathlib import Path
import logging

class ConfigManager:
    DEFAULT_CONFIG = {
        'yolo': {
            'config_path': "yolov4-tiny.cfg",
            'weights_path': "yolov4-tiny.weights",
            'names_path': "coco.names",
            'confidence_threshold': 0.65,
            'target_classes': ["person", "car", "truck"],
            'batch_size': 4,
            'input_size': (416, 416)
        },
        'save': {
            'output_dir': "detections",
            'format': "jpg",
            'save_detections_only': True,
            'create_subfolders': True
        },
        'display': {
            'window_width_ratio': 0.8,
            'window_height_ratio': 0.8,
            'canvas_width_ratio': 0.6,
            'canvas_height_ratio': 0.6
        }
    }

    @classmethod
    def load_config(cls):
        config_path = Path("config.json")
        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    return json.load(f)
            return cls.DEFAULT_CONFIG.copy()
        except Exception as e:
            logging.error(f"Erreur de chargement config: {e}")
            return cls.DEFAULT_CONFIG.copy()

    @classmethod
    def save_config(cls, config):
        try:
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logging.error(f"Erreur de sauvegarde config: {e}")
