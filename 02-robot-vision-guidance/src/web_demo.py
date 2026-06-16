import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from kinematics import RobotKinematics
from vision_system import VisionSystem, SyntheticPartGenerator

st.set_page_config(page_title="Robot Vision Guidance", page_icon="🤖", layout="wide")

st.title("🤖 Robot Vision Guidance System")
st.markdown("**Camera-guided pick-and-place simulation** — [Company Name]")
st.markdown("---")

if "kinematics" not in st.session_state:
    st.session_state.kinematics = RobotKinematics()
    st.session_state.vision = VisionSystem()
    st.session_state.part_gen = SyntheticPartGenerator()
    st.session_state.joints = np.zeros(6)
    st.session_state.target_joints = np.zeros(6)
    st.session_state.moving = False

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/robot-3.png", width=80)
    st.subheader("Robot Control")

    with st.expander("Joint Angles", expanded=True):
        joint_sliders = []
        for i in range(6):
            val = st.slider(f"Joint {i + 1}", -180, 180, 0, 5,
                            key=f"j{i}")
            joint_sliders.append(np.deg2rad(val))

    st.subheader("Action")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 Home", use_container_width=True):
            st.session_state.target_joints = np.zeros(6)
            st.session_state.moving = True
    with col2:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.target_joints = np.random.uniform(-1, 1, 6)
            st.session_state.moving = True

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📐 Robot Arm Visualization")
    viz_placeholder = st.empty()

with col2:
    st.subheader("🔍 Camera View")
    cam_placeholder = st.empty()

st.markdown("---")
col3, col4, col5 = st.columns(3)
with col3:
    end_effector_ph = st.empty()
with col4:
    joint_angles_ph = st.empty()
with col5:
    status_ph = st.empty()

joints = np.array([np.deg2rad(st.session_state[f"j{i}"]) for i in range(6)])

T, transforms = st.session_state.kinematics.forward_kinematics(joints)
ee_pos = T[:3, 3]

fig = go.Figure()
positions = [T[:3, 3] for T in transforms]
xs, ys, zs = zip(*positions)

fig.add_trace(go.Scatter3d(
    x=xs, y=ys, z=zs,
    mode='lines+markers',
    line=dict(color='blue', width=6),
    marker=dict(size=8, color='red'),
    name='Robot Arm'
))

target_T, _ = st.session_state.kinematics.forward_kinematics(
    st.session_state.target_joints
)
target_pos = target_T[:3, 3]
fig.add_trace(go.Scatter3d(
    x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
    mode='markers',
    marker=dict(size=12, color='green', symbol='diamond'),
    name='Target'
))

fig.update_layout(
    height=500,
    scene=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[0, 1.5]),
        aspectmode='cube',
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)
viz_placeholder.plotly_chart(fig, use_container_width=True)

img, parts = st.session_state.part_gen.generate_scene(num_parts=2)
cam_placeholder.image(img, channels="RGB", use_container_width=True)

end_effector_ph.metric("End Effector",
                       f"({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})")
joint_angles_ph.metric("Joint Angles",
                       f"{np.rad2deg(joints).round(1).tolist()}")
status_ph.metric("Status", "Running" if st.session_state.moving else "Idle")

if st.session_state.moving:
    step = 0.05
    diff = st.session_state.target_joints - joints
    if np.linalg.norm(diff) < 0.01:
        st.session_state.moving = False
    else:
        new_joints = joints + step * diff / np.linalg.norm(diff)
        for i in range(6):
            st.session_state[f"j{i}"] = np.rad2deg(new_joints[i])
        time.sleep(0.05)
        st.rerun()
