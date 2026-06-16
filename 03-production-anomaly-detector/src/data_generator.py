import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import random


class ProductionDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)

    def generate_sensor_timeseries(self, n_samples: int = 10000,
                                    n_sensors: int = 8) -> pd.DataFrame:
        data = {}
        timestamps = pd.date_range(start="2025-01-01", periods=n_samples, freq="1s")

        data["timestamp"] = timestamps

        base_temp = 75.0
        data["temperature"] = base_temp + self.rng.randn(n_samples) * 2.0 + \
                              np.sin(np.linspace(0, 20 * np.pi, n_samples)) * 3.0

        data["pressure"] = 100.0 + self.rng.randn(n_samples) * 5.0 + \
                           np.sin(np.linspace(0, 10 * np.pi, n_samples)) * 2.0

        data["vibration_x"] = self.rng.randn(n_samples) * 0.1
        data["vibration_y"] = self.rng.randn(n_samples) * 0.1
        data["vibration_z"] = self.rng.randn(n_samples) * 0.1

        data["rpm"] = 1500 + self.rng.randn(n_samples) * 50 + \
                      np.sin(np.linspace(0, 5 * np.pi, n_samples)) * 100

        data["current"] = 5.0 + self.rng.randn(n_samples) * 0.3
        data["voltage"] = 480.0 + self.rng.randn(n_samples) * 2.0

        df = pd.DataFrame(data)
        return df

    def add_anomalies(self, df: pd.DataFrame,
                       anomaly_ratio: float = 0.05) -> pd.DataFrame:
        df = df.copy()
        n_anomalies = int(len(df) * anomaly_ratio)
        anomaly_indices = self.rng.choice(len(df), n_anomalies, replace=False)
        df["is_anomaly"] = False
        df["anomaly_type"] = "none"

        for idx in anomaly_indices:
            atype = self.py_rng.choice([
                "spike", "drift", "noise", "stuck", "trend_change"
            ])
            df.loc[idx, "is_anomaly"] = True
            df.loc[idx, "anomaly_type"] = atype

            if atype == "spike":
                col = self.py_rng.choice(["temperature", "pressure", "current"])
                df.loc[idx, col] += self.rng.uniform(15, 50)

            elif atype == "drift":
                window = slice(max(0, idx - 20), min(len(df), idx + 20))
                col = self.py_rng.choice(["temperature", "pressure", "rpm"])
                drift = np.linspace(0, self.rng.uniform(10, 30),
                                    window.stop - window.start)
                df.loc[df.index[window], col] += drift

            elif atype == "noise":
                window = slice(max(0, idx - 10), min(len(df), idx + 10))
                col = self.py_rng.choice(["vibration_x", "vibration_y", "vibration_z"])
                df.loc[df.index[window], col] += self.rng.uniform(0.5, 2.0, window.stop - window.start)

            elif atype == "stuck":
                window = slice(max(0, idx - 15), min(len(df), idx + 15))
                col = self.py_rng.choice(["temperature", "pressure"])
                stuck_val = df.loc[idx, col]
                df.loc[df.index[window], col] = stuck_val + self.rng.randn(window.stop - window.start) * 0.1

            elif atype == "trend_change":
                window = slice(idx, min(len(df), idx + 50))
                col = self.py_rng.choice(["rpm", "current"])
                slope = self.rng.uniform(-5, 5)
                trend = np.arange(window.stop - window.start) * slope
                df.loc[df.index[window], col] += trend

        return df

    def generate_image_anomalies(self, img_size: int = 64,
                                  n_normal: int = 1000,
                                  n_anomaly: int = 200) -> tuple:
        normal_imgs = []
        anomaly_imgs = []

        for _ in tqdm(range(n_normal), desc="Generating normal images"):
            img = self._generate_normal_part(img_size)
            normal_imgs.append(img)

        for _ in tqdm(range(n_anomaly), desc="Generating anomaly images"):
            img = self._generate_normal_part(img_size)
            img = self._apply_image_anomaly(img)
            anomaly_imgs.append(img)

        return np.array(normal_imgs), np.array(anomaly_imgs)

    def _generate_normal_part(self, img_size: int) -> np.ndarray:
        img = np.ones((img_size, img_size, 3), dtype=np.float32) * 0.9
        cx, cy = img_size // 2, img_size // 2
        rw, rh = img_size // 3, img_size // 4
        color = self.rng.uniform(0.3, 0.5)
        noise = self.rng.randn(img_size, img_size, 3) * 0.02

        y1, y2 = cy - rh, cy + rh
        x1, x2 = cx - rw, cx + rw
        img[y1:y2, x1:x2] = color + noise[y1:y2, x1:x2]

        img += noise
        img = np.clip(img, 0, 1)
        return img

    def _apply_image_anomaly(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        atype = self.py_rng.choice(["scratch", "blob", "missing", "bright"])

        if atype == "scratch":
            x1 = self.rng.randint(0, w)
            y1 = self.rng.randint(0, h)
            x2 = self.rng.randint(0, w)
            y2 = self.rng.randint(0, h)
            from skimage.draw import line
            rr, cc = line(y1, x1, y2, x2)
            rr = np.clip(rr, 0, h - 1)
            cc = np.clip(cc, 0, w - 1)
            img[rr, cc] = 1.0

        elif atype == "blob":
            from skimage.draw import disk
            cx = self.rng.randint(10, w - 10)
            cy = self.rng.randint(10, h - 10)
            r = self.rng.randint(5, 15)
            rr, cc = disk((cy, cx), r)
            rr = np.clip(rr, 0, h - 1)
            cc = np.clip(cc, 0, w - 1)
            img[rr, cc] = 0.0

        elif atype == "missing":
            mw = self.rng.randint(8, 20)
            mh = self.rng.randint(8, 20)
            mx = self.rng.randint(0, w - mw)
            my = self.rng.randint(0, h - mh)
            img[my:my + mh, mx:mx + mw] = 0.9

        elif atype == "bright":
            bx = self.rng.randint(0, w - 20)
            by = self.rng.randint(0, h - 20)
            bw = self.rng.randint(10, 30)
            bh = self.rng.randint(10, 30)
            img[by:by + bh, bx:bx + bw] += self.rng.uniform(0.3, 0.6)

        img = np.clip(img, 0, 1)
        return img

    def generate_dataset(self, output_dir: str = "data",
                          n_normal: int = 4000, n_anomaly: int = 1000):
        output_dir = Path(output_dir)
        (output_dir / "sensor").mkdir(parents=True, exist_ok=True)
        (output_dir / "images/normal").mkdir(parents=True, exist_ok=True)
        (output_dir / "images/anomaly").mkdir(parents=True, exist_ok=True)

        print("Generating sensor data...")
        df = self.generate_sensor_timeseries(n_samples=n_normal + n_anomaly)
        df = self.add_anomalies(df, anomaly_ratio=n_anomaly / (n_normal + n_anomaly))
        df.to_csv(output_dir / "sensor" / "production_data.csv", index=False)
        print(f"Sensor data: {len(df)} samples, {df['is_anomaly'].sum()} anomalies")

        print("Generating image data...")
        normal_imgs, anomaly_imgs = self.generate_image_anomalies(
            n_normal=n_normal // 10, n_anomaly=n_anomaly // 10
        )

        import cv2
        for i, img in enumerate(normal_imgs):
            cv2.imwrite(str(output_dir / "images/normal" / f"normal_{i:06d}.png"),
                        (img * 255).astype(np.uint8))

        for i, img in enumerate(anomaly_imgs):
            cv2.imwrite(str(output_dir / "images/anomaly" / f"anomaly_{i:06d}.png"),
                        (img * 255).astype(np.uint8))

        normal_imgs = normal_imgs.transpose(0, 3, 1, 2)
        anomaly_imgs = anomaly_imgs.transpose(0, 3, 1, 2)
        np.save(output_dir / "images" / "normal_imgs.npy", normal_imgs)
        np.save(output_dir / "images" / "anomaly_imgs.npy", anomaly_imgs)

        print(f"Normal images: {len(normal_imgs)}, Anomaly images: {len(anomaly_imgs)}")
        print(f"Dataset saved to {output_dir}")
        return output_dir


if __name__ == "__main__":
    gen = ProductionDataGenerator()
    gen.generate_dataset()
