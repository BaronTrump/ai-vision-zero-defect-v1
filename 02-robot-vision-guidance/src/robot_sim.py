import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from kinematics import RobotKinematics
from typing import List, Optional
import matplotlib.animation as animation


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class RobotVisualizer:
    def __init__(self, kinematics: RobotKinematics):
        self.kinematics = kinematics
        self.fig = None
        self.ax = None

    def _setup_axes(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 1.5)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('6-DOF Robot Arm')

    def draw_robot(self, joint_angles: np.ndarray, ax: plt.Axes = None):
        if ax is None:
            if self.fig is None:
                self._setup_axes()
            ax = self.ax

        T_end, transforms = self.kinematics.forward_kinematics(joint_angles)
        positions = [T[:3, 3] for T in transforms]

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        ax.cla()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('6-DOF Robot Arm')

        ax.plot(xs, ys, zs, 'b-', linewidth=3, label='Robot Arm')
        ax.scatter(xs, ys, zs, color='red', s=50, zorder=5)

        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, f'J{i}', fontsize=8)

        T_ee = transforms[-1]
        R = T_ee[:3, :3]
        p = T_ee[:3, 3]

        for i, color in enumerate(['r', 'g', 'b']):
            axis = R[:, i] * 0.1
            ax.quiver(p[0], p[1], p[2],
                      axis[0], axis[1], axis[2],
                      color=color, linewidth=2)

        ax.legend()
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        return ax

    def animate_trajectory(self, trajectory: List[np.ndarray],
                           interval: int = 50, save_path: str = None):
        self._setup_axes()

        def update(frame):
            self.draw_robot(trajectory[frame], self.ax)
            self.ax.set_title(f'Trajectory Frame {frame + 1}/{len(trajectory)}')

        ani = animation.FuncAnimation(
            self.fig, update, frames=len(trajectory),
            interval=interval, repeat=True
        )

        if save_path:
            ani.save(save_path, writer='pillow', fps=20)

        plt.show()
        return ani

    def show_grasp_pose(self, joint_angles: np.ndarray,
                        target_pose: np.ndarray, ax: plt.Axes = None):
        if ax is None:
            self._setup_axes()
            ax = self.ax

        self.draw_robot(joint_angles, ax)

        p = target_pose[:3, 3]
        ax.scatter([p[0]], [p[1]], [p[2]],
                   color='green', s=100, marker='*', label='Target')
        ax.legend()
        return ax


class Simulation2D:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.robot_x = width // 2
        self.robot_y = height // 2
        self.arm_length = 150
        self.joint_angle = 0
        self.gripper_state = False

    def update(self, target_x: float, target_y: float):
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        self.joint_angle = np.arctan2(dy, dx)

    def get_end_effector_pos(self) -> Tuple[float, float]:
        x = self.robot_x + self.arm_length * np.cos(self.joint_angle)
        y = self.robot_y + self.arm_length * np.sin(self.joint_angle)
        return x, y


if __name__ == "__main__":
    kin = RobotKinematics()
    viz = RobotVisualizer(kin)
    q = np.array([0.0, 0.5, -0.3, 0.0, 0.8, 0.0])
    viz.draw_robot(q)
    plt.show()
