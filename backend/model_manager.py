from typing import Dict
from base_model import BaseModel
from yolo_model import YOLOModel
from fastercnn_model import FasterRCNNModel


class ModelManager:    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self._load_models()
    
    def _load_models(self):
        for model_name, model_path in self.config.items():
            if model_name == "YOLO":
                self.models[model_name] = YOLOModel(model_path)
            elif model_name == "FasterRCNN":
                self.models[model_name] = FasterRCNNModel(model_path)
            else:
                print(f"Non-existing model: {model_name}")
    
    def get_model(self, model_name: str) -> BaseModel:
        return self.models.get(model_name)