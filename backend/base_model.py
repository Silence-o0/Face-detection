from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BoundingBox:
    """Class for representing a bounding box"""

    def __init__(self, x: float, y: float, w: float, h: float,
                 confidence: float = 1.0, class_id: int = 0, class_name: str = "object"):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for JSON"""
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }


class BaseModel(ABC):
    """Base class for all detection models"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.original_shape = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Loads model from disk"""
        pass

    @abstractmethod
    def preprocess(self, image_bytes: bytes) -> Any:
        """Image preprocessing
        Returns: preprocessed_input (can be any type)
        Also must set self.original_shape
        """
        pass

    @abstractmethod
    def inference(self, preprocessed_input: Any) -> Any:
        """Runs model inference"""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, original_shape: tuple) -> List[BoundingBox]:
        """Postprocessing of inference results (model-specific)"""
        pass

    def predict(self, image_bytes: bytes) -> List[BoundingBox]:
        """Full pipeline: preprocess -> inference -> postprocess"""
        preprocessed = self.preprocess(image_bytes)
        raw_output = self.inference(preprocessed)
        return self.postprocess(raw_output, self.original_shape)