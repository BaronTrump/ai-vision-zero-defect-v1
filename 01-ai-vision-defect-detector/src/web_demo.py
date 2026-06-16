import streamlit as st
import cv2
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque
from config import config
from data_generator import SyntheticDefectGenerator
from model import get_model, preprocess_image, postprocess_detections
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


st.set_page_config(
    page_title="AI Vision Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    if "engine" not in st.session_state:
        st.session_state.engine = None
        st.session_state.running = False
        st.session_state.results_history = deque(maxlen=1000)
        st.session_state.start_time = time.time()
        st.session_state.total_inspected = 0
        st.session_state.passed = 0
        st.session_state.failed = 0
        st.session_state.defect_counts = {}


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/robot-3.png", width=80)
        st.title("AI Vision QC")
        st.markdown("---")

        st.subheader("Camera Settings")
        source_type = st.selectbox("Source Type", ["Webcam", "Video File", "Image"])
        source_input = st.text_input("Source (0 for webcam)", "0")

        st.markdown("---")
        st.subheader("Detection Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        iou_thresh = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

        st.markdown("---")
        st.subheader("Dashboard")
        st.metric("Uptime", f"{int(time.time() - st.session_state.start_time)}s")
        st.metric("Total Inspected", st.session_state.total_inspected)

        if st.button("Reset Stats", use_container_width=True):
            st.session_state.total_inspected = 0
            st.session_state.passed = 0
            st.session_state.failed = 0
            st.session_state.defect_counts = {}
            st.session_state.results_history.clear()
            st.rerun()

        return source_type, source_input, conf_thresh, iou_thresh


def render_live_feed():
    col = st.columns([3, 1])[0]
    with col:
        st.subheader("📷 Live Inspection Feed")
        placeholder = st.empty()

        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            pass_metric = st.empty()
        with metrics_cols[1]:
            fail_metric = st.empty()
        with metrics_cols[2]:
            rate_metric = st.empty()
        with metrics_cols[3]:
            acc_metric = st.empty()

    return placeholder, pass_metric, fail_metric, rate_metric, acc_metric


def render_stats_panel():
    col = st.columns([1, 3])[0]
    with col:
        st.subheader("📊 Defect Distribution")
        defect_chart = st.empty()

        st.subheader("📋 Recent Results")
        recent_table = st.empty()

    return defect_chart, recent_table


def render_defect_breakdown():
    st.subheader("🔬 Defect Type Breakdown")
    cols = st.columns(4)
    defect_icons = {
        "scratch": "〰️", "dent": "🔘", "crack": "⚡",
        "discoloration": "🎨", "missing_component": "⬜",
        "deformation": "🌀", "contamination": "🦠",
    }
    displays = []
    for col_idx, (defect_type, icon) in enumerate(defect_icons.items()):
        with cols[col_idx % 4]:
            count = st.session_state.defect_counts.get(defect_type, 0)
            d = st.empty()
            displays.append(d)
    return displays


def update_metrics(pass_metric, fail_metric, rate_metric, acc_metric):
    total = st.session_state.total_inspected
    passed = st.session_state.passed
    failed = st.session_state.failed
    elapsed = time.time() - st.session_state.start_time

    rate = total / elapsed if elapsed > 0 else 0
    accuracy = (passed / total * 100) if total > 0 else 100

    pass_metric.metric("✅ Passed", passed)
    fail_metric.metric("❌ Failed", failed)
    rate_metric.metric("⚡ Rate (unit/s)", f"{rate:.1f}")
    acc_metric.metric("🎯 Accuracy", f"{accuracy:.1f}%")


def update_defect_chart(defect_chart):
    counts = st.session_state.defect_counts
    if counts:
        df = pd.DataFrame({
            "Defect Type": list(counts.keys()),
            "Count": list(counts.values()),
        })
        fig = px.bar(df, x="Defect Type", y="Count",
                     color="Count", color_continuous_scale="Reds",
                     title="Defect Distribution")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        defect_chart.plotly_chart(fig, use_container_width=True)
    else:
        defect_chart.info("No defects detected yet")


def update_recent_table(recent_table):
    history = list(st.session_state.results_history)[-20:]
    if history:
        rows = []
        for r in history:
            cls_name = r.get("class_name", "unknown")
            conf = r.get("confidence", 0)
            status = "✅ PASS" if cls_name in ["none", 0] else "❌ FAIL"
            rows.append({
                "Status": status,
                "Defect": cls_name if status == "❌ FAIL" else "—",
                "Confidence": f"{conf:.2%}",
                "Time": r.get("timestamp", ""),
            })
        df = pd.DataFrame(rows)
        recent_table.dataframe(df, use_container_width=True, hide_index=True)
    else:
        recent_table.info("Waiting for inspections...")


def process_frame(frame, conf_thresh):
    from model import get_model, preprocess_image, postprocess_detections
    engine = st.session_state.engine
    if engine is None:
        config.model.device = "cpu"
        engine = get_model(model_type=config.model.model_type)
        st.session_state.engine = engine

    img_resized = preprocess_image(frame, config.data.img_size)
    results = engine(img_resized)
    dets = postprocess_detections(results, conf_threshold=conf_thresh)

    for det in dets:
        cls_name = det.get("class_name", "unknown")
        det["timestamp"] = datetime.now().strftime("%H:%M:%S")
        st.session_state.results_history.append(det)
        st.session_state.total_inspected += 1

        if cls_name in ["none", 0]:
            st.session_state.passed += 1
        else:
            st.session_state.failed += 1
            if cls_name not in st.session_state.defect_counts:
                st.session_state.defect_counts[cls_name] = 0
            st.session_state.defect_counts[cls_name] += 1

    for det in dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det.get("class_name", "unknown")
        conf = det.get("confidence", 0)
        is_defect = cls_name not in ["none", 0]
        color = (0, 255, 0) if not is_defect else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    st.title("🔍 AI Vision Quality Control System")
    st.markdown("**Real-time Defect Detection for Production Lines** — [Company Name]")
    st.markdown("---")

    init_session_state()

    source_type, source_input, conf_thresh, iou_thresh = render_sidebar()
    placeholder, pass_metric, fail_metric, rate_metric, acc_metric = render_live_feed()
    defect_chart, recent_table = render_stats_panel()

    source_map = {"Webcam": 0, "Video File": source_input, "Image": source_input}
    try:
        src = int(source_input) if source_input.isdigit() else source_input
    except ValueError:
        src = source_input

    cap = cv2.VideoCapture(src if source_type == "Webcam" else source_input)

    if not cap.isOpened():
        st.error(f"Cannot open source: {source_input}")
        st.info("Using demo mode with synthetic data generator...")
        gen = SyntheticDefectGenerator()
        demo_mode = True
    else:
        demo_mode = False

    st.session_state.running = True

    while st.session_state.running:
        if demo_mode:
            frame, defect_type, bbox = gen.generate_sample()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video stream")
                break

        frame = process_frame(frame, conf_thresh)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        update_metrics(pass_metric, fail_metric, rate_metric, acc_metric)
        update_defect_chart(defect_chart)
        update_recent_table(recent_table)

        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
            break

    cap.release() if not demo_mode else None


if __name__ == "__main__":
    main()
