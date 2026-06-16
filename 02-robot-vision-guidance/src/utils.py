import numpy as np
from kinematics import RobotKinematics


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    cx, cy, cz = np.cos(euler)
    sx, sy, sz = np.sin(euler)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def pose_to_transform(position: np.ndarray, euler: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = euler_to_rotation_matrix(euler)
    return T


def transform_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    position = T[:3, 3]
    euler = rotation_matrix_to_euler(T[:3, :3])
    return position, euler


def distance_between_poses(T1: np.ndarray, T2: np.ndarray) -> float:
    pos_diff = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    R_diff = T1[:3, :3].T @ T2[:3, :3]
    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    return pos_diff + 0.1 * angle_diff


def interpolate_poses(T_start: np.ndarray, T_end: np.ndarray,
                       steps: int = 50) -> List[np.ndarray]:
    poses = []
    for i in range(steps + 1):
        t = i / steps
        pos = T_start[:3, 3] + t * (T_end[:3, 3] - T_start[:3, 3])

        R_start = T_start[:3, :3]
        R_end = T_end[:3, :3]
        R = R_start @ (np.eye(3) + t * (R_start.T @ R_end - np.eye(3)))

        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = R
        poses.append(T)

    return poses


if __name__ == "__main__":
    kin = RobotKinematics()
    q = np.array([0.2, 0.5, -0.3, 0.1, 0.8, 0.0])
    T, transforms = kin.forward_kinematics(q)
    pos, euler = transform_to_pose(T)
    print(f"End effector position: {pos}")
    print(f"End effector euler: {euler}")

    T_recovered = pose_to_transform(pos, euler)
    print(f"Recovery error: {np.linalg.norm(T - T_recovered):.6f}")
