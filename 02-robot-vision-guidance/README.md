# Robot Vision Guidance System

AI-powered visual servoing for robotic pick-and-place operations — camera-guided robot arm simulation for automated production lines.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Camera         │    │  Object Detection │    │  Coordinate     │
│  (Part Feed)    │───▶│  (YOLO/OpenCV)    │───▶│  Transformation  │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Robot Arm      │◀───│  Path Planning   │◀───│  Pose Estimation │
│  Simulation     │    │  (IK Solver)     │    │  (PnP)           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

- **6-DOF Robot Arm Simulation** — realistic kinematics with PyGame/Matplotlib visualization
- **Camera Calibration** — intrinsic/extrinsic parameter estimation
- **Object Detection & Pose Estimation** — detect parts and compute 6-DOF pose
- **Inverse Kinematics Solver** — compute joint angles for target positions
- **Pick-and-Place Pipeline** — complete workflow from detection to placement
- **Trajectory Planning** — smooth motion with collision avoidance
- **Streamlit Dashboard** — real-time visualization of robot state and camera feed
- **Synthetic Data Generation** — generate training data for pose estimation

## Quick Start

```bash
pip install -r requirements.txt

# Run robot simulation with vision guidance
python src/web_demo.py

# Run headless simulation
python src/robot_sim.py

# Test camera calibration
python src/calibration.py
```

## Project Structure

```
├── src/
│   ├── robot_sim.py        # 6-DOF robot arm simulation
│   ├── vision_system.py    # Object detection & pose estimation
│   ├── calibration.py      # Camera calibration utilities
│   ├── kinematics.py       # Forward/Inverse kinematics
│   ├── pick_and_place.py   # Full pick-and-place pipeline
│   ├── web_demo.py         # Streamlit dashboard
│   └── utils.py            # Visualization & math utilities
├── requirements.txt
└── README.md
```

## Vision-Guided Robotic Workflow

1. **Part Detection**: Camera detects part on conveyor belt
2. **Pose Estimation**: Compute 6-DOF pose (position + orientation)
3. **Coordinate Transform**: Map camera coordinates to robot base frame
4. **IK Solution**: Compute joint angles for grasp pose
5. **Trajectory Planning**: Plan smooth path with via points
6. **Execution**: Robot moves to pick position, grasps, moves to place
7. **Verification**: Camera confirms successful placement

## [Company Name] Integration

- **Conveyor-based**: Parts detected while moving on belt, robot tracks and picks
- **Bin picking**: Random bin picking with 3D pose estimation
- **Assembly**: Vision-guided alignment for precision assembly
- **Quality sorting**: Defect detection integration for automated sorting
