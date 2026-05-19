from typing import List, Any
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

from base_model import BaseModel, BoundingBox


class YOLOModel(BaseModel):
    """YOLO model implementation for face detection using ultralytics"""

    def load_model(self):
        """Loads YOLO model"""
        print(f"Loading YOLO model: {self.model_path}")

        self.model = YOLO(self.model_path)

        self.device = 'cpu'
        self.model.to(self.device)

        self.conf_threshold = 0.5

        self.class_names = {0: 'face'}

        print(f"Model loaded on {self.device}")

    def preprocess(self, image_bytes: bytes) -> Any:
        """Preprocessing for YOLO"""
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        self.original_shape = (image.width, image.height)
        return image

    def inference(self, preprocessed_input: Any) -> Any:
        """Runs YOLO inference"""
        results = self.model(
            preprocessed_input,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )
        return results[0]

    def postprocess(self, raw_output: Any, original_shape: tuple) -> List[BoundingBox]:
        """Postprocessing YOLO results"""
        boxes = []

        if raw_output.boxes is None or len(raw_output.boxes) == 0:
            return boxes

        boxes_data = raw_output.boxes.xyxy.cpu().numpy()
        confidences = raw_output.boxes.conf.cpu().numpy()
        class_ids = raw_output.boxes.cls.cpu().numpy()

        for i in range(len(boxes_data)):
            box_data = boxes_data[i]
            confidence = confidences[i]
            class_id = int(class_ids[i])

            x1, y1, x2, y2 = box_data

            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)

            boxes.append(BoundingBox(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=float(confidence),
                class_id=class_id,
                class_name=self.class_names.get(class_id, 'face')
            ))

        return boxes
