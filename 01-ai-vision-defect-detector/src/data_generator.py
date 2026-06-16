import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm
from config import config
import json


class SyntheticDefectGenerator:
    def __init__(self):
        self.cfg = config
        self.rng = random.Random(42)
        self.np_rng = np.random.RandomState(42)

    def _random_part_shape(self, img_size):
        w = self.rng.randint(*self.cfg.data.part_size_range)
        h = self.rng.randint(*self.cfg.data.part_size_range)
        x = self.rng.randint(50, img_size - w - 50)
        y = self.rng.randint(50, img_size - h - 50)
        return x, y, w, h

    def _draw_base_part(self, img, x, y, w, h):
        color = (
            self.rng.randint(180, 220),
            self.rng.randint(180, 220),
            self.rng.randint(180, 220),
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 2)
        for _ in range(self.rng.randint(3, 6)):
            sx = x + self.rng.randint(10, w - 10)
            sy = y + self.rng.randint(10, h - 10)
            sw = self.rng.randint(10, 30)
            sh = self.rng.randint(10, 30)
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh),
                          (150, 150, 150), -1)
        return img

    def _apply_scratch(self, img, x, y, w, h):
        num_scratch = self.rng.randint(1, 5)
        for _ in range(num_scratch):
            x1 = x + self.rng.randint(0, w)
            y1 = y + self.rng.randint(0, h)
            angle = self.rng.uniform(0, np.pi)
            length = self.rng.randint(20, 80)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            color = (self.rng.randint(180, 220),
                     self.rng.randint(180, 220),
                     self.rng.randint(180, 220))
            thickness = self.rng.randint(1, 3)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _apply_dent(self, img, x, y, w, h):
        cx = x + self.rng.randint(10, w - 10)
        cy = y + self.rng.randint(10, h - 10)
        radius = self.rng.randint(8, 25)
        color = (self.rng.randint(60, 100),
                 self.rng.randint(60, 100),
                 self.rng.randint(60, 100))
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius, (40, 40, 40), 1)
        return img

    def _apply_crack(self, img, x, y, w, h):
        x1 = x + self.rng.randint(0, w)
        y1 = y + self.rng.randint(0, h)
        points = [(x1, y1)]
        for _ in range(self.rng.randint(5, 15)):
            px, py = points[-1]
            dx = self.rng.randint(-15, 15)
            dy = self.rng.randint(-15, 15)
            points.append((px + dx, py + dy))
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1],
                     (30, 30, 30), self.rng.randint(1, 3))
        return img

    def _apply_discoloration(self, img, x, y, w, h):
        roi = img[y:y + h, x:x + w]
        tint = self.rng.choice([(0, 0, 200), (200, 0, 0),
                                (0, 200, 0), (200, 200, 0)])
        intensity = self.rng.uniform(0.3, 0.6)
        mask = np.zeros_like(roi)
        mask[:] = tint
        cv2.addWeighted(roi, 1 - intensity, mask, intensity, 0, roi)
        img[y:y + h, x:x + w] = roi
        return img

    def _apply_missing_component(self, img, x, y, w, h):
        roi = img[y:y + h, x:x + w]
        sub_w = self.rng.randint(w // 4, w // 2)
        sub_h = self.rng.randint(h // 4, h // 2)
        sub_x = self.rng.randint(0, w - sub_w)
        sub_y = self.rng.randint(0, h - sub_h)
        bg = np.full((sub_h, sub_w, 3), self.cfg.data.background_color,
                     dtype=np.uint8)
        roi[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w] = bg
        img[y:y + h, x:x + w] = roi
        return img

    def _apply_deformation(self, img, x, y, w, h):
        roi = img[y:y + h, x:x + w]
        rows, cols = roi.shape[:2]
        src_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1],
                              [cols - 1, rows - 1]])
        offset = self.rng.randint(5, 20)
        dst_pts = np.float32([
            [self.rng.randint(-offset, offset),
             self.rng.randint(-offset, offset)],
            [cols - 1 + self.rng.randint(-offset, offset),
             self.rng.randint(-offset, offset)],
            [self.rng.randint(-offset, offset),
             rows - 1 + self.rng.randint(-offset, offset)],
            [cols - 1 + self.rng.randint(-offset, offset),
             rows - 1 + self.rng.randint(-offset, offset)],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        deformed = cv2.warpPerspective(roi, M, (cols, rows))
        img[y:y + h, x:x + w] = deformed
        return img

    def _apply_contamination(self, img, x, y, w, h):
        num_spots = self.rng.randint(3, 15)
        for _ in range(num_spots):
            sx = x + self.rng.randint(0, w)
            sy = y + self.rng.randint(0, h)
            color = (self.rng.randint(0, 60),
                     self.rng.randint(0, 60),
                     self.rng.randint(0, 60))
            radius = self.rng.randint(2, 8)
            cv2.circle(img, (sx, sy), radius, color, -1)
        return img

    def apply_defect(self, img, x, y, w, h, defect_type):
        fx = {
            "scratch": self._apply_scratch,
            "dent": self._apply_dent,
            "crack": self._apply_crack,
            "discoloration": self._apply_discoloration,
            "missing_component": self._apply_missing_component,
            "deformation": self._apply_deformation,
            "contamination": self._apply_contamination,
        }
        if defect_type in fx:
            img = fx[defect_type](img, x, y, w, h)
        return img

    def _add_noise(self, img):
        noise = self.np_rng.randn(*img.shape) * self.rng.randint(2, 8)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def _adjust_lighting(self, img):
        alpha = self.rng.uniform(0.7, 1.3)
        beta = self.rng.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def generate_sample(self, defect_type: str = None, has_defect: bool = None):
        img_size = self.cfg.data.img_size
        img = np.full((img_size, img_size, 3),
                      self.cfg.data.background_color, dtype=np.uint8)

        x, y, w, h = self._random_part_shape(img_size)
        img = self._draw_base_part(img, x, y, w, h)

        if has_defect is None:
            has_defect = self.rng.random() < self.cfg.defect.defect_probability

        if has_defect:
            if defect_type is None:
                defect_type = self.rng.choice(self.cfg.defect.defect_types)
            img = self.apply_defect(img, x, y, w, h, defect_type)
        else:
            defect_type = "none"

        img = self._adjust_lighting(img)
        img = self._add_noise(img)

        bbox = [x, y, x + w, y + h]
        return img, defect_type, bbox

    def generate_dataset(self, output_dir: str = None, samples: int = None):
        if output_dir is None:
            output_dir = self.cfg.data.output_dir
        if samples is None:
            samples = self.cfg.data.samples_per_class * len(self.cfg.defect.defect_types)

        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"

        for d in [images_dir, labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        classes = ["none"] + self.cfg.defect.defect_types
        class_map = {c: i for i, c in enumerate(classes)}

        per_class = samples // len(classes)
        annotations = []

        for cls_name in classes:
            print(f"Generating {cls_name} samples...")
            for i in tqdm(range(per_class)):
                img, defect_type, bbox = self.generate_sample(
                    defect_type=None if cls_name == "none" else cls_name,
                    has_defect=cls_name != "none"
                )
                fname = f"{cls_name}_{i:06d}"
                cv2.imwrite(str(images_dir / f"{fname}.jpg"), img)

                x1, y1, x2, y2 = bbox
                x_c = ((x1 + x2) / 2) / self.cfg.data.img_size
                y_c = ((y1 + y2) / 2) / self.cfg.data.img_size
                bw = (x2 - x1) / self.cfg.data.img_size
                bh = (y2 - y1) / self.cfg.data.img_size
                cls_id = class_map[cls_name]

                with open(labels_dir / f"{fname}.txt", "w") as f:
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

                annotations.append({
                    "file": f"{fname}.jpg",
                    "class": cls_name,
                    "class_id": cls_id,
                    "bbox": bbox,
                })

        with open(output_dir / "annotations.json", "w") as f:
            json.dump(annotations, f, indent=2)

        with open(output_dir / "classes.txt", "w") as f:
            for cls_name in classes:
                f.write(f"{cls_name}\n")

        print(f"Dataset generated: {len(annotations)} samples")
        print(f"Classes: {classes}")
        return output_dir


if __name__ == "__main__":
    gen = SyntheticDefectGenerator()
    gen.generate_dataset(samples=200)
