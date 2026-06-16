import numpy as np
import cv2
from kinematics import RobotKinematics
from vision_system import VisionSystem, SyntheticPartGenerator
from typing import Optional


class PickAndPlaceController:
    def __init__(self):
        self.kinematics = RobotKinematics()
        self.vision = VisionSystem()
        self.current_joints = np.zeros(6)
        self.gripper_open = True
        self.status = "idle"

    def plan_grasp(self, target_pose: np.ndarray) -> Optional[np.ndarray]:
        try:
            pre_grasp_pose = target_pose.copy()
            pre_grasp_pose[2, 3] += 0.1

            pre_grasp_q = self.kinematics.inverse_kinematics(
                pre_grasp_pose, self.current_joints
            )
            grasp_q = self.kinematics.inverse_kinematics(
                target_pose, pre_grasp_q
            )

            return grasp_q
        except ValueError as e:
            print(f"Grasp planning failed: {e}")
            return None

    def plan_place(self, place_pose: np.ndarray) -> Optional[np.ndarray]:
        try:
            place_q = self.kinematics.inverse_kinematics(
                place_pose, self.current_joints
            )
            return place_q
        except ValueError as e:
            print(f"Place planning failed: {e}")
            return None

    def execute_pick_and_place(self, part_pose: np.ndarray,
                                place_pose: np.ndarray) -> dict:
        log = {"steps": [], "success": True}

        grasp_q = self.plan_grasp(part_pose)
        if grasp_q is None:
            log["success"] = False
            log["error"] = "Cannot plan grasp"
            return log

        trajectory = self.kinematics.interpolate_trajectory(
            self.current_joints, grasp_q
        )
        self.current_joints = grasp_q
        log["steps"].append({"action": "move_to_grasp", "joints": grasp_q.tolist()})
        log["steps"].append({"action": "close_gripper"})
        self.gripper_open = False

        lift_pose = part_pose.copy()
        lift_pose[2, 3] += 0.15
        lift_q = self.kinematics.inverse_kinematics(lift_pose, self.current_joints)
        self.current_joints = lift_q
        log["steps"].append({"action": "lift_part", "joints": lift_q.tolist()})

        place_q = self.plan_place(place_pose)
        if place_q is None:
            log["success"] = False
            log["error"] = "Cannot plan place"
            return log

        self.current_joints = place_q
        log["steps"].append({"action": "move_to_place", "joints": place_q.tolist()})
        log["steps"].append({"action": "open_gripper"})
        self.gripper_open = True

        retract_pose = place_pose.copy()
        retract_pose[2, 3] += 0.15
        retract_q = self.kinematics.inverse_kinematics(retract_pose, self.current_joints)
        self.current_joints = retract_q
        log["steps"].append({"action": "retract", "joints": retract_q.tolist()})

        return log

    def process_camera_frame(self, img: np.ndarray) -> np.ndarray:
        part = self.vision.detect_part(img)
        if part:
            grasp_pose = self.vision.estimate_grasp_pose(img, part)
            img = self.vision.draw_detection(img, part, grasp_pose)
            self.status = "part_detected"
        else:
            self.status = "searching"
        return img

    def get_state(self) -> dict:
        T, _ = self.kinematics.forward_kinematics(self.current_joints)
        return {
            "joints": self.current_joints.tolist(),
            "end_effector": T[:3, 3].tolist(),
            "gripper_open": self.gripper_open,
            "status": self.status,
        }


class SimulatedProductionLine:
    def __init__(self):
        self.controller = PickAndPlaceController()
        self.part_gen = SyntheticPartGenerator()
        self.conveyor_position = 0.0
        self.conveyor_speed = 0.02
        self.parts_on_belt = []
        self.picked_parts = []
        self.place_position = np.eye(4)
        self.place_position[:3, 3] = np.array([0.5, 0.3, 0.1])

    def update(self):
        self.conveyor_position += self.conveyor_speed

        if len(self.parts_on_belt) < 3 and np.random.random() < 0.01:
            img, parts = self.part_gen.generate_scene(1)
            if parts:
                self.parts_on_belt.append({
                    "info": parts[0],
                    "position": self.conveyor_position,
                    "detected": False,
                })

        for part in self.parts_on_belt[:]:
            if not part["detected"] and part["position"] > 0.3:
                part_pos = np.eye(4)
                part_pos[:3, 3] = np.array([
                    part["info"]["center"][0] * 0.001,
                    part["info"]["center"][1] * 0.001,
                    0.05,
                ])
                result = self.controller.execute_pick_and_place(
                    part_pos, self.place_position
                )
                if result["success"]:
                    self.picked_parts.append(part)
                    self.parts_on_belt.remove(part)

        return self.get_state()

    def get_state(self) -> dict:
        return {
            "conveyor_position": self.conveyor_position,
            "parts_on_belt": len(self.parts_on_belt),
            "parts_picked": len(self.picked_parts),
            "robot": self.controller.get_state(),
        }


if __name__ == "__main__":
    line = SimulatedProductionLine()
    for _ in range(100):
        state = line.update()
    print(f"Production line state: {state}")
