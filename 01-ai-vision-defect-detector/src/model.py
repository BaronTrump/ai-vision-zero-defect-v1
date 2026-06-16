import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from config import config


class DefectClassifier(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LightweightDefectDetector(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.class_head(features)
        bbox = torch.sigmoid(self.bbox_head(features))
        return class_logits, bbox


def get_model(model_type: str = None, num_classes: int = 8, device: str = None):
    if model_type is None:
        model_type = config.model.model_type
    if device is None:
        device = config.model.device

    if model_type.startswith("yolo"):
        try:
            from ultralytics import YOLO
            model = YOLO(f"{model_type}.pt")
            if device == "cpu":
                model.to("cpu")
            return model
        except ImportError:
            print("ultralytics not installed, using custom CNN")
            return LightweightDefectDetector(num_classes)
    elif model_type == "cnn":
        return LightweightDefectDetector(num_classes)
    elif model_type == "classifier":
        return DefectClassifier(num_classes)
    else:
        return LightweightDefectDetector(num_classes)


def preprocess_image(img: np.ndarray, img_size: int = 640):
    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    if scale != 1:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    return canvas


def postprocess_detections(results, conf_threshold: float = None):
    if conf_threshold is None:
        conf_threshold = config.model.confidence_threshold

    detections = []
    if hasattr(results, 'boxes'):
        boxes = results.boxes
        for i in range(len(boxes)):
            det = {
                "bbox": boxes.xyxy[i].tolist(),
                "confidence": float(boxes.conf[i]),
                "class_id": int(boxes.cls[i]),
                "class_name": results.names[int(boxes.cls[i])],
            }
            if det["confidence"] >= conf_threshold:
                detections.append(det)
    return detections


def load_model_weights(model, path: str):
    path = Path(path)
    if path.exists():
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        print(f"Loaded weights from {path}")
    return model
