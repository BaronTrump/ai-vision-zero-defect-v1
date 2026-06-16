import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from collections import deque

from data_simulator import ProductionSimulator, LineState
from metrics import MetricsCalculator, format_metric, get_metric_color
from alerts import AlertManager


st.set_page_config(
    page_title="Production Monitor Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.simulator = ProductionSimulator()
        st.session_state.metrics = MetricsCalculator()
        st.session_state.alerts = AlertManager()
        st.session_state.lines = {}
        st.session_state.history = {f"Line {i + 1}": deque(maxlen=1000)
                                      for i in range(3)}
        st.session_state.start_time = datetime.now()
        st.session_state.total_produced = 0
        st.session_state.total_defects = 0
        st.session_state.dark_mode = False
        st.session_state.auto_refresh = True

        for i in range(3):
            products = ["Widget A", "Widget B", "Gadget X"]
            rates = [100, 80, 120]
            st.session_state.lines[f"Line {i + 1}"] = LineState(
                f"Line {i + 1}", products[i], rates[i]
            )

        st.session_state.initialized = True


def render_sidebar():
    with st.sidebar:
        st.title("🏭 Production Control")
        st.markdown("---")

        tab1, tab2 = st.tabs(["Monitor", "Settings"])

        with tab1:
            st.subheader("Live Overview")
            overall_oee = st.empty()
            total_produced = st.empty()
            total_defects = st.empty()
            uptime = st.empty()

            st.markdown("---")
            st.subheader("⚠️ Active Alerts")
            alert_panel = st.empty()

        with tab2:
            st.subheader("Display")
            st.session_state.dark_mode = st.toggle("Dark Mode")

            st.subheader("Simulation Speed")
            speed = st.select_slider("Speed", [1, 2, 5, 10, 50], 5)

            st.subheader("Alert Thresholds")
            defect_warn = st.slider("Defect Rate Warning", 0.01, 0.20, 0.05, 0.01)
            oee_crit = st.slider("OEE Critical", 0.3, 0.9, 0.6, 0.05)

            if st.button("Reset Simulation", use_container_width=True):
                st.session_state.initialized = False
                st.rerun()

        return speed, alert_panel, overall_oee, total_produced, total_defects, uptime


def render_oee_dashboard():
    st.title("🏭 Production Line Monitor & Control")
    st.markdown("**AI-Integrated Production Management System** — [Company Name]")
    st.markdown("---")

    kpi_cols = st.columns(6)
    kpi_placeholders = []
    titles = ["OEE", "Availability", "Performance", "Quality", "Throughput", "Defect Rate"]
    for i, (col, title) in enumerate(zip(kpi_cols, titles)):
        with col:
            st.markdown(f"**{title}**")
            ph = st.empty()
            kpi_placeholders.append(ph)

    st.markdown("---")
    tabs = st.tabs(["📊 Overview", "📈 Trends", "⚠️ Alerts", "📋 Report"])

    return tabs, kpi_placeholders


def render_line_cards():
    st.subheader("Production Lines")
    line_cols = st.columns(3)
    line_placeholders = []
    for i, (col, (line_id, line_state)) in enumerate(zip(line_cols, st.session_state.lines.items())):
        with col:
            ph = st.empty()
            line_placeholders.append(ph)
    return line_placeholders


def render_trends_tab():
    col1, col2 = st.columns(2)
    with col1:
        oee_trend = st.empty()
        throughput_trend = st.empty()
    with col2:
        defect_trend = st.empty()
        cycle_trend = st.empty()
    return oee_trend, throughput_trend, defect_trend, cycle_trend


def render_alerts_tab():
    alert_history = st.empty()
    alert_stats = st.empty()
    return alert_history, alert_stats


def render_report_tab():
    report_params = st.columns(3)
    with report_params[0]:
        report_type = st.selectbox("Report Type", ["Shift Summary", "Daily Report", "Custom Range"])
    with report_params[1]:
        report_line = st.selectbox("Line", ["All Lines"] + list(st.session_state.lines.keys()))
    with report_params[2]:
        if st.button("Generate Report", use_container_width=True):
            st.success("Report generated!")
    report_output = st.empty()
    return report_output


def update_simulation(speed: int):
    for line_id, line_state in st.session_state.lines.items():
        for _ in range(speed):
            row = st.session_state.simulator.generate_realtime_row(
                line_id=line_id,
                product=line_state.product,
                target_rate=line_state.target_rate,
            )
            line_state.update(row)
            st.session_state.history[line_id].append(row)

            st.session_state.total_produced += row.get("units_produced", 0)
            st.session_state.total_defects += row.get("units_defective", 0)

            if "units_defective" in row and row["units_defective"] > 0:
                alert = st.session_state.alerts.evaluate_all({
                    "defect_rate": row["units_defective"] / max(row["units_produced"], 1),
                    "temperature": row.get("temperature", 0),
                    "vibration": row.get("vibration", 0),
                })


def update_kpis(kpi_placeholders):
    now = datetime.now()
    uptime_delta = now - st.session_state.start_time

    total_produced = st.session_state.total_produced
    total_defects = st.session_state.total_defects

    avg_oee = 0
    avg_avail = 0
    avg_perf = 0
    avg_qual = 0
    total_throughput = 0
    total_defect_rate = 0
    n_lines = len(st.session_state.lines)

    for line_id, line_state in st.session_state.lines.items():
        oee = line_state.get_oee()
        avg_oee += oee["oee"]
        avg_avail += oee["availability"]
        avg_perf += oee["performance"]
        avg_qual += oee["quality"]

    avg_oee /= n_lines
    avg_avail /= n_lines
    avg_perf /= n_lines
    avg_qual /= n_lines

    recent_history = []
    for h in st.session_state.history.values():
        recent_history.extend(list(h)[-60:])

    if recent_history:
        recent_df = pd.DataFrame(recent_history)
        running = recent_df[recent_df["status"] == "running"]
        total_throughput = running["units_produced"].sum()
        total_defect_rate = running["units_defective"].sum() / max(running["units_produced"].sum(), 1)

    metrics = [avg_oee, avg_avail, avg_perf, avg_qual, total_throughput, total_defect_rate]
    formats = ["pct", "pct", "pct", "pct", "rate", "pct"]

    for ph, value, fmt in zip(kpi_placeholders, metrics, formats):
        ph.metric("", format_metric(value, fmt))

    return avg_oee, avg_avail, avg_perf, avg_qual


def update_line_cards(line_placeholders):
    for ph, (line_id, line_state) in zip(line_placeholders, st.session_state.lines.items()):
        oee = line_state.get_oee()
        color = get_metric_color(oee["oee"])

        ph.markdown(f"""
        **{line_id}** — {line_state.product}
        - Status: **{line_state.status}**
        - OEE: **{format_metric(oee['oee'], 'pct')}**
        - Produced: **{line_state.daily_total}**
        - Defects: **{line_state.daily_defects}**
        - Uptime: **{format_metric(1 - line_state.daily_downtime / 480, 'pct')}**
        """)


def update_sidebar_stats(overall_oee, total_produced, total_defects, uptime):
    uptime_delta = datetime.now() - st.session_state.start_time
    hours = uptime_delta.total_seconds() / 3600

    overall_oee.metric("Avg OEE", format_metric(
        np.mean([line.get_oee()["oee"] for line in st.session_state.lines.values()]), "pct"
    ))
    total_produced.metric("Total Produced", st.session_state.total_produced)
    total_defects.metric("Total Defects", st.session_state.total_defects)
    uptime.metric("Session", f"{hours:.1f}h")


def update_alert_panel(alert_panel):
    recent = st.session_state.alerts.get_recent_alerts(5)
    if recent:
        alert_text = ""
        for a in recent:
            icon = "🔴" if a["severity"] == "critical" else "🟡"
            alert_text += f"{icon} **{a['rule']}**: {a['message']}\n\n"
        alert_panel.markdown(alert_text)
    else:
        alert_panel.success("✅ All systems nominal")


def update_trends(oee_trend, throughput_trend, defect_trend, cycle_trend):
    for line_id in st.session_state.lines:
        history = list(st.session_state.history[line_id])
        if not history:
            continue

    oee_trend.info("OEE trend chart")
    throughput_trend.info("Throughput trend chart")
    defect_trend.info("Defect rate trend chart")
    cycle_trend.info("Cycle time trend chart")


def main():
    with st.spinner("Initializing production dashboard..."):
        init_session_state()

    speed, alert_panel, overall_oee, total_produced, total_defects, uptime = render_sidebar()
    tabs, kpi_placeholders = render_oee_dashboard()
    line_placeholders = render_line_cards()
    oee_trend, throughput_trend, defect_trend, cycle_trend = render_trends_tab()
    alert_history, alert_stats = render_alerts_tab()
    report_output = render_report_tab()

    update_simulation(speed)
    avg_oee, avg_avail, avg_perf, avg_qual = update_kpis(kpi_placeholders)
    update_line_cards(line_placeholders)
    update_sidebar_stats(overall_oee, total_produced, total_defects, uptime)
    update_alert_panel(alert_panel)
    update_trends(oee_trend, throughput_trend, defect_trend, cycle_trend)

    time.sleep(1)


if __name__ == "__main__":
    main()
