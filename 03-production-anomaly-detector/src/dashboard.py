import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import deque
from datetime import datetime, timedelta
import time
from pathlib import Path
from model import AnomalyScorer, ConvAutoencoder
from data_generator import ProductionDataGenerator
from detector import AnomalyDetector
import cv2


st.set_page_config(
    page_title="Production Anomaly Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state():
    if "detector" not in st.session_state:
        st.session_state.detector = AnomalyDetector()
        st.session_state.scores = deque(maxlen=500)
        st.session_state.alerts = deque(maxlen=100)
        st.session_state.start_time = time.time()
        st.session_state.data_gen = ProductionDataGenerator()
        st.session_state.sim_data = st.session_state.data_gen.generate_sensor_timeseries(5000)
        st.session_state.data_idx = 0


def render_sidebar():
    with st.sidebar:
        st.title("🔬 Anomaly Monitor")
        st.markdown("---")

        st.subheader("Detection Settings")
        threshold_pct = st.slider("Threshold Percentile", 80, 99, 95, 1)
        update_rate = st.slider("Update Rate (Hz)", 1, 30, 10, 1)

        st.markdown("---")
        st.subheader("Data Source")
        source = st.selectbox("Source", ["Simulated Sensor", "Camera Feed", "Upload File"])

        st.markdown("---")
        st.subheader("Live Stats")
        stats_placeholder = st.empty()

        if st.button("Reset Monitor", use_container_width=True):
            st.session_state.scores.clear()
            st.session_state.alerts.clear()
            st.rerun()

        return threshold_pct, update_rate, source, stats_placeholder


def render_main_dashboard():
    st.title("🏭 Production Line Anomaly Detection")
    st.markdown("**Real-time monitoring with unsupervised deep learning**")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("📈 Live Anomaly Score")
        score_chart = st.empty()

    with col2:
        st.subheader("⚠️ Alerts")
        alerts_panel = st.empty()

    with col3:
        st.subheader("📊 Stats")
        stats_panel = st.empty()

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("📉 Score Distribution")
        dist_chart = st.empty()

    with col5:
        st.subheader("🕐 Score Timeline")
        timeline_chart = st.empty()

    return score_chart, alerts_panel, stats_panel, dist_chart, timeline_chart


def render_sensor_data_explorer():
    st.subheader("🔍 Sensor Data Explorer")
    tab1, tab2, tab3 = st.tabs(["Temperature", "Pressure", "Vibration"])

    with tab1:
        temp_chart = st.empty()
    with tab2:
        press_chart = st.empty()
    with tab3:
        vib_chart = st.empty()

    return temp_chart, press_chart, vib_chart


def update_simulated_data(detector, data_gen, idx, threshold_pct):
    df = st.session_state.sim_data
    if idx >= len(df):
        idx = 0

    row = df.iloc[idx]
    sensor_cols = ["temperature", "pressure", "vibration_x", "vibration_y",
                   "vibration_z", "rpm", "current", "voltage"]
    values = row[sensor_cols].values.astype(float)

    result = detector.analyze_sensor(values[np.newaxis, :])

    st.session_state.scores.append({
        "score": result["score"],
        "timestamp": datetime.now(),
        "is_anomaly": result["is_anomaly"],
    })

    if result["is_anomaly"]:
        st.session_state.alerts.append({
            "timestamp": datetime.now(),
            "score": result["score"],
            "index": idx,
        })

    return idx + 1, result


def update_score_chart(score_chart):
    scores = list(st.session_state.scores)
    if not scores:
        return

    df = pd.DataFrame(scores)
    fig = go.Figure()
    colors = ["red" if s["is_anomaly"] else "blue" for s in scores]

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["score"],
        mode="lines+markers",
        marker=dict(color=colors, size=4),
        line=dict(color="rgba(100, 100, 255, 0.3)"),
        name="Anomaly Score",
    ))

    threshold = st.session_state.detector.detector.scorer.threshold or 0.1
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: {threshold:.4f}")

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Reconstruction Error",
        xaxis_title="Time",
        showlegend=True,
    )
    score_chart.plotly_chart(fig, use_container_width=True)


def update_alerts(alerts_panel):
    alerts = list(st.session_state.alerts)
    if alerts:
        recent = alerts[-5:]
        alert_text = ""
        for a in reversed(recent):
            ts = a["timestamp"].strftime("%H:%M:%S")
            alert_text += f"⚠️ **{ts}** — Score: {a['score']:.4f}\n\n"
        alerts_panel.markdown(alert_text)
    else:
        alerts_panel.info("No anomalies detected yet")


def update_stats(stats_panel):
    scores = [s["score"] for s in st.session_state.scores]
    if scores:
        stats_panel.metric("Mean Score", f"{np.mean(scores):.4f}")
        stats_panel.metric("Max Score", f"{np.max(scores):.4f}")
        stats_panel.metric("Anomalies", sum(1 for s in st.session_state.scores if s["is_anomaly"]))
        stats_panel.metric("Alert Rate", f"{sum(1 for s in st.session_state.scores if s['is_anomaly']) / max(len(scores), 1) * 100:.1f}%")
    else:
        stats_panel.info("Waiting for data...")


def update_dist_chart(dist_chart):
    scores = [s["score"] for s in st.session_state.scores]
    if scores:
        fig = px.histogram(
            x=scores, nbins=30,
            labels={"x": "Anomaly Score"},
            color_discrete_sequence=["rgba(100, 100, 255, 0.7)"],
        )
        threshold = st.session_state.detector.detector.scorer.threshold or 0.1
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        dist_chart.plotly_chart(fig, use_container_width=True)


def update_timeline_chart(timeline_chart):
    scores = list(st.session_state.scores)
    if len(scores) < 2:
        return

    df = pd.DataFrame(scores)
    df["rolling_mean"] = df["score"].rolling(window=10).mean()
    df["rolling_std"] = df["score"].rolling(window=10).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["score"],
        mode="lines",
        name="Raw Score",
        line=dict(color="rgba(100, 100, 255, 0.3)"),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["rolling_mean"],
        mode="lines",
        name="Rolling Mean (10)",
        line=dict(color="orange", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["rolling_mean"] + 2 * df["rolling_std"],
        mode="lines",
        name="+2σ",
        line=dict(color="red", dash="dot"),
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
    timeline_chart.plotly_chart(fig, use_container_width=True)


def update_sensor_charts(temp_chart, press_chart, vib_chart):
    df = st.session_state.sim_data
    recent = df.tail(200)

    for chart, col, title, color in [
        (temp_chart, "temperature", "Temperature (°C)", "red"),
        (press_chart, "pressure", "Pressure (kPa)", "blue"),
        (vib_chart, "vibration_x", "Vibration X (mm/s)", "green"),
    ]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=recent[col].values,
            mode="lines",
            name=title,
            line=dict(color=color),
        ))
        anomalies = recent[recent["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies[col],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=8, symbol="x"),
            ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20),
                          title=title)
        chart.plotly_chart(fig, use_container_width=True)


def main():
    init_state()
    threshold_pct, update_rate, source, stats_placeholder = render_sidebar()
    score_chart, alerts_panel, stats_panel, dist_chart, timeline_chart = render_main_dashboard()
    temp_chart, press_chart, vib_chart = render_sensor_data_explorer()

    while True:
        st.session_state.data_idx, result = update_simulated_data(
            st.session_state.detector,
            st.session_state.data_gen,
            st.session_state.data_idx,
            threshold_pct,
        )

        update_score_chart(score_chart)
        update_alerts(alerts_panel)
        update_stats(stats_panel)
        update_dist_chart(dist_chart)
        update_timeline_chart(timeline_chart)
        update_sensor_charts(temp_chart, press_chart, vib_chart)

        time.sleep(1.0 / update_rate)


if __name__ == "__main__":
    main()
