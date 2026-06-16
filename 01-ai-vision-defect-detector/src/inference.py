import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque
from config import config
from model import get_model, preprocess_image, postprocess_detections


class InferenceEngine:
    def __init__(self, model_path: str = None):
        self.cfg = config
        self.model = get_model(model_type=self.cfg.model.model_type)
        self.classes = ["none", "scratch", "dent", "crack",
                        "discoloration", "missing_component",
                        "deformation", "contamination"]

        if model_path and Path(model_path).exists():
            try:
                if hasattr(self.model, 'load'):
                    self.model.load(model_path)
                else:
                    state = torch.load(model_path, map_location=self.cfg.model.device)
                    self.model.load_state_dict(state)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")

        self.stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "defect_counts": {d: 0 for d in self.classes},
            "recent_results": deque(maxlen=self.cfg.dashboard.max_history),
        }

    def detect(self, img: np.ndarray):
        img_resized = preprocess_image(img, self.cfg.data.img_size)
        results = self.model(img_resized)

        detections = postprocess_detections(
            results, conf_threshold=self.cfg.model.confidence_threshold
        )

        for det in detections:
            self.stats["total"] += 1
            cls_name = det["class_name"]
            if cls_name == "none" or cls_name == 0:
                self.stats["passed"] += 1
            else:
                self.stats["failed"] += 1
                if cls_name in self.stats["defect_counts"]:
                    self.stats["defect_counts"][cls_name] += 1

            self.stats["recent_results"].append(det)

        return detections

    def draw_detections(self, img: np.ndarray, detections: list):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_name = det["class_name"]
            conf = det["confidence"]

            is_defect = cls_name != "none" and cls_name != 0
            color = (0, 255, 0) if not is_defect else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def run_on_video(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error opening video source: {source}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = self.detect(frame)
            frame = self.draw_detections(frame, dets)

            cv2.imshow("AI Defect Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def run_inference(source: str = None):
    if source is None:
        source = config.inference.source
    engine = InferenceEngine()
    print(f"Running inference on: {source}")
    engine.run_on_video(source)


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    run_inference(source)
