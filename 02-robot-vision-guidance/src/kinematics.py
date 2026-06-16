import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DHParam:
    theta: float
    d: float
    a: float
    alpha: float


@dataclass
class RobotConfig:
    dh_params: List[DHParam] = field(default_factory=lambda: [
        DHParam(0, 0.4, 0.0, np.pi / 2),
        DHParam(0, 0.0, 0.4, 0.0),
        DHParam(0, 0.0, 0.3, 0.0),
        DHParam(0, 0.35, 0.0, np.pi / 2),
        DHParam(0, 0.0, 0.0, -np.pi / 2),
        DHParam(0, 0.1, 0.0, 0.0),
    ])
    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-np.pi, np.pi) for _ in range(6)
    ])
    base_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


class RobotKinematics:
    def __init__(self, config: RobotConfig = None):
        self.config = config or RobotConfig()

    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ])

    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        T = np.eye(4)
        T[:3, 3] = self.config.base_position
        transforms = [T.copy()]

        for i, dh in enumerate(self.config.dh_params):
            theta = joint_angles[i] + dh.theta
            Ti = self.dh_transform(theta, dh.d, dh.a, dh.alpha)
            T = T @ Ti
            transforms.append(T.copy())

        return T, transforms

    def inverse_kinematics(self, target_pose: np.ndarray,
                           initial_guess: np.ndarray = None) -> np.ndarray:
        if initial_guess is None:
            initial_guess = np.zeros(6)

        from scipy.optimize import minimize

        def objective(q):
            T, _ = self.forward_kinematics(q)
            pos_err = np.linalg.norm(T[:3, 3] - target_pose[:3, 3])
            rot_err = 1 - (np.trace(T[:3, :3].T @ target_pose[:3, :3]) - 1) / 2
            return pos_err * 10 + rot_err * 5

        bounds = self.config.joint_limits
        result = minimize(objective, initial_guess, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 1000})

        if result.fun < 0.01:
            return result.x % (2 * np.pi)
        else:
            raise ValueError(f"IK did not converge: error={result.fun:.4f}")

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        epsilon = 1e-6
        T0, _ = self.forward_kinematics(joint_angles)
        p0 = T0[:3, 3]
        J = np.zeros((6, 6))

        for i in range(6):
            q_plus = joint_angles.copy()
            q_plus[i] += epsilon
            T_plus, _ = self.forward_kinematics(q_plus)
            p_plus = T_plus[:3, 3]
            J[:3, i] = (p_plus - p0) / epsilon

            R_plus = T_plus[:3, :3]
            R0 = T0[:3, :3]
            dR = (R_plus @ R0.T - np.eye(3)) / epsilon
            omega = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])
            J[3:, i] = omega

        return J

    def get_end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        T, _ = self.forward_kinematics(joint_angles)
        return T

    def interpolate_trajectory(self, q_start: np.ndarray, q_end: np.ndarray,
                                steps: int = 50) -> List[np.ndarray]:
        traj = []
        for i in range(steps + 1):
            t = i / steps
            q = q_start + t * (q_end - q_start)
            traj.append(q.copy())
        return traj
