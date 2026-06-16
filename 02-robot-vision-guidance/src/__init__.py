from typing import List, Tuple, Optional
import numpy as np
from kinematics import RobotKinematics, RobotConfig, DHParam
from vision_system import VisionSystem, SyntheticPartGenerator
from robot_sim import RobotVisualizer
from pick_and_place import PickAndPlaceController, SimulatedProductionLine
import time
import threading


class VisionGuidedRobotDemo:
    def __init__(self):
        self.kinematics = RobotKinematics()
        self.vision = VisionSystem()
        self.visualizer = RobotVisualizer(self.kinematics)
        self.part_gen = SyntheticPartGenerator()
        self.controller = PickAndPlaceController()
        self.running = False
        self.current_joints = np.zeros(6)
        self.target_joints = np.zeros(6)
        self.trajectory_progress = 0.0

    def generate_scene(self) -> Tuple[np.ndarray, List[dict]]:
        return self.part_gen.generate_scene(num_parts=2)

    def detect_and_grasp(self, img: np.ndarray) -> Optional[dict]:
        part = self.vision.detect_part(img)
        if part is None:
            return None

        grasp_pose = self.vision.estimate_grasp_pose(img, part)
        if grasp_pose is None:
            return None

        result = self.controller.execute_pick_and_place(
            grasp_pose,
            np.eye(4)
        )
        return {
            "part": part,
            "grasp_pose": grasp_pose,
            "result": result,
        }

    def get_robot_state(self) -> dict:
        T, _ = self.kinematics.forward_kinematics(self.current_joints)
        return {
            "joints": self.current_joints.tolist(),
            "end_effector": T[:3, 3].tolist(),
            "status": "running" if self.running else "stopped",
        }

    def run_demo_cycle(self):
        self.running = True
        for _ in range(10):
            if not self.running:
                break
            img, parts = self.generate_scene()
            for part in parts:
                grasp_pose = np.eye(4)
                cx, cy = part["center"]
                grasp_pose[:3, 3] = np.array([cx * 0.001, cy * 0.001, 0.1])

                try:
                    target_q = self.kinematics.inverse_kinematics(
                        grasp_pose, self.current_joints
                    )
                    for t in np.linspace(0, 1, 20):
                        self.current_joints = self.current_joints + t * (target_q - self.current_joints) / 20
                        time.sleep(0.02)
                except ValueError:
                    pass

            time.sleep(0.5)
        self.running = False


class WebDemoState:
    def __init__(self):
        self.kinematics = RobotKinematics()
        self.vision = VisionSystem()
        self.part_gen = SyntheticPartGenerator()
        self.current_joints = np.zeros(6)
        self.joint_sliders = [0.0] * 6
        self.conveyor_position = 0.0
        self.parts_detected = 0
        self.parts_picked = 0
        self.gripper_open = True
        self.demo_mode = "auto"

    def update_conveyor(self):
        self.conveyor_position += 0.01
        if self.conveyor_position > 1.0:
            self.conveyor_position = 0.0

    def randomize_joints(self):
        self.current_joints = np.random.uniform(-1, 1, 6)

    def get_end_effector_pose(self) -> np.ndarray:
        T, _ = self.kinematics.forward_kinematics(self.current_joints)
        return T

    def get_scene(self) -> Tuple[np.ndarray, List[dict]]:
        return self.part_gen.generate_scene(num_parts=self.rng.randint(1, 3))

    def __init__(self):
        self.rng = np.random.RandomState()
        self.part_gen = SyntheticPartGenerator()


if __name__ == "__main__":
    demo = VisionGuidedRobotDemo()
    print("Starting robot vision demo...")
    print(f"Initial state: {demo.get_robot_state()}")

    img, parts = demo.generate_scene()
    print(f"Generated scene with {len(parts)} parts")

    result = demo.detect_and_grasp(img)
    if result:
        print(f"Grasp result: {result['result']['success']}")
    else:
        print("No parts detected")
