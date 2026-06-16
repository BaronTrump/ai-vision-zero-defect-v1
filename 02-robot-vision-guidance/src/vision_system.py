import numpy as np
import cv2
import random
from pathlib import Path
from typing import Tuple, List, Optional


class CameraCalibration:
    def __init__(self, img_size: Tuple[int, int] = (640, 480)):
        self.img_size = img_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self._setup_default()

    def _setup_default(self):
        fx = fy = 600
        cx = self.img_size[0] / 2
        cy = self.img_size[1] / 2
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def generate_calibration_board(self, rows: int = 7, cols: int = 9,
                                    square_size: float = 0.025) -> np.ndarray:
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        return objp

    def estimate_pose(self, object_points: np.ndarray,
                      image_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        if success:
            return rvec, tvec
        return None, None

    def pixel_to_world(self, pixel_coords: np.ndarray,
                       depth: float = 1.0) -> np.ndarray:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (pixel_coords[0] - cx) * depth / fx
        y = (pixel_coords[1] - cy) * depth / fy
        return np.array([x, y, depth])

    def world_to_pixel(self, world_coords: np.ndarray) -> np.ndarray:
        pts, _ = cv2.projectPoints(
            world_coords.reshape(1, 1, 3),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.camera_matrix,
            self.dist_coeffs
        )
        return pts[0, 0]


class VisionSystem:
    def __init__(self):
        self.calibration = CameraCalibration()
        self.detector = cv2.SimpleBlobDetector_create()

    def detect_part(self, img: np.ndarray) -> Optional[dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 500:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        box = cv2.boxPoints(rect)

        return {
            "center": (cx, cy),
            "contour": largest,
            "bbox": box,
            "angle": angle,
            "area": area,
        }

    def estimate_grasp_pose(self, img: np.ndarray, part_info: dict) -> Optional[np.ndarray]:
        if part_info is None:
            return None

        cx, cy = part_info["center"]
        depth_estimate = 0.5

        world_pos = self.calibration.pixel_to_world(
            np.array([cx, cy]), depth_estimate
        )

        angle_rad = np.deg2rad(part_info["angle"])
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ])

        T = np.eye(4)
        T[:3, 3] = world_pos
        T[:3, :3] = R
        return T

    def draw_detection(self, img: np.ndarray, part_info: dict,
                       grasp_pose: np.ndarray = None) -> np.ndarray:
        if part_info is None:
            return img

        box = part_info["bbox"]
        cv2.drawContours(img, [np.int0(box)], 0, (0, 255, 0), 2)

        cx, cy = part_info["center"]
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        angle = part_info["angle"]
        length = 40
        rad = np.deg2rad(angle)
        end_x = int(cx + length * np.cos(rad))
        end_y = int(cy + length * np.sin(rad))
        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (255, 0, 0), 2)

        cv2.putText(img, f"Part ({cx}, {cy})", (cx - 60, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if grasp_pose is not None:
            pos = grasp_pose[:3, 3]
            cv2.putText(img, f"Grasp: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        return img


class SyntheticPartGenerator:
    def __init__(self, img_size: Tuple[int, int] = (640, 480)):
        self.img_size = img_size
        self.rng = random.Random(42)

    def generate_scene(self, num_parts: int = 1) -> Tuple[np.ndarray, List[dict]]:
        img = np.full((*self.img_size[::-1], 3), 200, dtype=np.uint8)
        parts = []

        for _ in range(num_parts):
            w = self.rng.randint(40, 80)
            h = self.rng.randint(30, 60)
            x = self.rng.randint(50, self.img_size[0] - w - 50)
            y = self.rng.randint(50, self.img_size[1] - h - 50)
            angle = self.rng.uniform(0, 180)

            color = (
                self.rng.randint(100, 180),
                self.rng.randint(100, 180),
                self.rng.randint(100, 180),
            )

            center = (x + w // 2, y + h // 2)
            rect = ((center[0], center[1]),
                    (w, h),
                    angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img, [box], 0, color, -1)
            cv2.drawContours(img, [box], 0, (50, 50, 50), 1)

            parts.append({
                "center": center,
                "bbox": box,
                "angle": angle,
                "width": w,
                "height": h,
            })

        return img, parts
