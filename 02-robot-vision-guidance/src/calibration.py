import numpy as np
import cv2
import json
from pathlib import Path
from vision_system import CameraCalibration
from typing import List, Tuple


def generate_checkerboard(rows: int = 6, cols: int = 8,
                           square_size_mm: float = 30,
                           img_size: Tuple[int, int] = (640, 480)):
    img = np.ones((*img_size[::-1], 3), dtype=np.uint8) * 255
    square_h = img_size[1] // rows
    square_w = img_size[0] // cols

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                x1 = j * square_w
                y1 = i * square_h
                x2 = (j + 1) * square_w
                y2 = (i + 1) * square_h
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return img


def calibrate_from_images(image_paths: List[str],
                          board_size: Tuple[int, int] = (9, 6),
                          square_size: float = 0.025) -> Tuple[np.ndarray, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            cv2.drawChessboardCorners(img, board_size, corners2, ret)

    if len(obj_points) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        print(f"Calibration RMS error: {ret}")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")
        return mtx, dist

    return None, None


def save_calibration(mtx: np.ndarray, dist: np.ndarray, path: str = "calibration.json"):
    data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {path}")


def load_calibration(path: str = "calibration.json") -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    return mtx, dist


def test_calibration():
    cal = CameraCalibration()
    print("Camera matrix:")
    print(cal.camera_matrix)

    pixel = np.array([320, 240])
    world = cal.pixel_to_world(pixel, depth=0.5)
    print(f"Pixel {pixel} -> World {world}")

    pixel_back = cal.world_to_pixel(world)
    print(f"World {world} -> Pixel {pixel_back}")


if __name__ == "__main__":
    test_calibration()
