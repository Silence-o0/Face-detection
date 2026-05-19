from typing import List, Any
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from base_model import BaseModel, BoundingBox


IMAGE_SIZE = (800, 800)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES = 2


def _build_model(num_classes: int) -> FasterRCNN:
    """Creates Faster R-CNN architecture identical to the one used
    during training (see fastercnn_train.ipynb)."""
    backbone = torchvision.models.vgg16(weights=None).features
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((65, 140, 185, 260, 450),),
        aspect_ratios=((0.86,),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )


class FasterRCNNModel(BaseModel):
    """Faster R-CNN model implementation for face detection"""

    def load_model(self):
        print(f"Loading Faster R-CNN model: {self.model_path}")

        self.device = torch.device('cpu')

        self.model = _build_model(num_classes=NUM_CLASSES)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.conf_threshold = 0.5
        self.class_names = {1: 'face'}

        print(f"Model loaded on {self.device}")

    def preprocess(self, image_bytes: bytes) -> Any:
        """Resize -> Normalize (ImageNet) -> CHW tensor.
        Reproduces albumentations pipeline from fastercnn_train.ipynb."""
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        self.original_shape = (image.width, image.height)

        image_resized = image.resize(IMAGE_SIZE, Image.BILINEAR)
        img_arr = np.asarray(image_resized, dtype=np.float32) / 255.0
        img_arr = (img_arr - MEAN) / STD
        img_arr = img_arr.transpose(2, 0, 1)  # HWC -> CHW

        return torch.from_numpy(img_arr).to(self.device)

    def inference(self, preprocessed_input: Any) -> Any:
        with torch.no_grad():
            predictions = self.model([preprocessed_input])

        return predictions[0]

    def postprocess(self, raw_output: Any, original_shape: tuple) -> List[BoundingBox]:
        boxes: List[BoundingBox] = []

        width, height = original_shape

        scale_x = width / IMAGE_SIZE[0]
        scale_y = height / IMAGE_SIZE[1]

        pred_boxes = raw_output['boxes'].cpu().numpy()
        pred_scores = raw_output['scores'].cpu().numpy()
        pred_labels = raw_output['labels'].cpu().numpy()

        for i in range(len(pred_boxes)):
            score = float(pred_scores[i])

            if score < self.conf_threshold:
                continue

            class_id = int(pred_labels[i])

            x1, y1, x2, y2 = pred_boxes[i]

            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            boxes.append(BoundingBox(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                confidence=score,
                class_id=class_id,
                class_name=self.class_names.get(class_id, 'face')
            ))

        return boxes