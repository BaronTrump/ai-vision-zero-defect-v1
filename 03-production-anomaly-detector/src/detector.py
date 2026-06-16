import torch
import numpy as np
from pathlib import Path
from collections import deque
from model import ConvAutoencoder, LSTMAutoencoder, AnomalyScorer


class AnomalyDetector:
    def __init__(self, model_path: str = None, model_type: str = "conv_ae"):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "conv_ae":
            self.model = ConvAutoencoder(latent_dim=128, img_channels=3)
        elif model_type == "lstm_ae":
            self.model = LSTMAutoencoder(input_dim=8, hidden_dim=64, latent_dim=32)
        else:
            self.model = ConvAutoencoder(latent_dim=128, img_channels=3)

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and "model_state" in state:
                self.model.load_state_dict(state["model_state"])
            else:
                self.model.load_state_dict(state)
            print(f"Loaded model from {model_path}")

        self.scorer = AnomalyScorer(self.model, self.device)
        self.history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)

    def analyze_image(self, img: np.ndarray) -> dict:
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        score = self.scorer.score(img_tensor)
        is_anomaly = score.item() > (self.scorer.threshold or 0.1)

        result = {
            "score": float(score),
            "is_anomaly": is_anomaly,
            "threshold": float(self.scorer.threshold or 0.1),
        }

        self.history.append(result)
        if is_anomaly:
            self.alerts.append(result)
            result["alert"] = "⚠️ ANOMALY DETECTED"

        return result

    def analyze_sensor(self, data: np.ndarray) -> dict:
        seq_tensor = torch.from_numpy(data).float().unsqueeze(0)

        score = self.scorer.score(seq_tensor)
        is_anomaly = score.item() > (self.scorer.threshold or 0.1)

        result = {
            "score": float(score),
            "is_anomaly": is_anomaly,
            "threshold": float(self.scorer.threshold or 0.1),
        }

        self.history.append(result)
        if is_anomaly:
            self.alerts.append(result)
            result["alert"] = "⚠️ SENSOR ANOMALY DETECTED"

        return result

    def get_stats(self) -> dict:
        scores = [h["score"] for h in self.history] if self.history else [0]
        return {
            "total_analyzed": len(self.history),
            "total_alerts": len(self.alerts),
            "current_score": scores[-1] if scores else 0,
            "mean_score": float(np.mean(scores)) if scores else 0,
            "max_score": float(np.max(scores)) if scores else 0,
            "alert_rate": len(self.alerts) / max(len(self.history), 1),
            "threshold": self.scorer.threshold or 0,
        }


class RealTimeMonitor:
    def __init__(self):
        self.detector = AnomalyDetector()
        self.sensor_buffer = deque(maxlen=32)

    def process_sensor_reading(self, reading: dict) -> dict:
        self.sensor_buffer.append(list(reading.values()))
        result = {"is_anomaly": False, "score": 0}

        if len(self.sensor_buffer) >= 32:
            seq = np.array(list(self.sensor_buffer))
            result = self.detector.analyze_sensor(seq)

        return result

    def process_camera_frame(self, frame: np.ndarray) -> dict:
        img = cv2.resize(frame, (64, 64))
        return self.detector.analyze_image(img)


def run_detection(source: str = None):
    import cv2
    detector = AnomalyDetector()
    cap = cv2.VideoCapture(source or 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.analyze_image(img_rgb)

        if result["is_anomaly"]:
            cv2.putText(frame, "⚠️ ANOMALY", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Score: {result['score']:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
