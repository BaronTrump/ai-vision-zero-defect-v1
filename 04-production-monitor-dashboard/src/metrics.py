import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


class MetricsCalculator:
    def __init__(self):
        self.defect_type_map = {
            "scratch": "cosmetic",
            "dent": "cosmetic",
            "crack": "structural",
            "discoloration": "cosmetic",
            "missing_component": "assembly",
            "deformation": "structural",
            "contamination": "process",
        }

    def calculate_oee(self, df: pd.DataFrame,
                       target_rate: int = 100) -> dict:
        if df.empty:
            return {"availability": 0, "performance": 0, "quality": 0, "oee": 0, "takt_time": 0}

        total_time = len(df)
        downtime = len(df[df["status"] == "downtime"])
        running = df[df["status"] == "running"]

        availability = 1 - (downtime / total_time) if total_time > 0 else 0

        actual_units = running["units_produced"].sum()
        max_time = (total_time - downtime) / 60
        theoretical_units = target_rate * max_time
        performance = actual_units / max(theoretical_units, 1)

        total_defects = running["units_defective"].sum()
        quality = 1 - (total_defects / max(actual_units, 1))

        oee = availability * performance * quality

        avg_cycle = running["cycle_time_ms"].mean()
        takt_time = 60000 / max(target_rate, 1)

        return {
            "availability": availability,
            "performance": performance,
            "quality": quality,
            "oee": oee,
            "takt_time": takt_time,
            "actual_takt": avg_cycle if not np.isnan(avg_cycle) else 0,
        }

    def calculate_yield(self, df: pd.DataFrame) -> float:
        running = df[df["status"] == "running"]
        total = running["units_produced"].sum()
        defects = running["units_defective"].sum()
        return 1 - (defects / max(total, 1))

    def calculate_throughput(self, df: pd.DataFrame, window_minutes: int = 5) -> pd.Series:
        df = df[df["status"] == "running"].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        throughput = df["units_produced"].rolling(
            window=f"{window_minutes}min", min_periods=1
        ).sum()
        return throughput * (60 / window_minutes)

    def calculate_defect_rate(self, df: pd.DataFrame) -> dict:
        running = df[df["status"] == "running"]
        total = running["units_produced"].sum()
        if total == 0:
            return {"overall": 0, "by_type": {}}

        defects = running["units_defective"].sum()
        return {
            "overall": defects / total,
            "total_defects": int(defects),
            "total_units": int(total),
        }

    def calculate_mtbf_mttr(self, df: pd.DataFrame) -> dict:
        downtime_events = df[df["status"].isin(["downtime", "downtime_start"])]
        num_events = len(downtime_events[downtime_events["status"] == "downtime_start"])

        total_runtime = len(df[df["status"] == "running"])
        total_downtime = len(df[df["status"] == "downtime"])

        mtbf = (total_runtime / max(num_events, 1)) * 60
        mttr = (total_downtime / max(num_events, 1)) * 60

        return {
            "mtbf_minutes": mtbf,
            "mttr_minutes": mttr,
            "num_failures": num_events,
        }

    def calculate_trend(self, df: pd.DataFrame, metric: str = "oee",
                         window: int = 30) -> float:
        if len(df) < window:
            return 0

        recent = df.tail(window)
        older = df.tail(window * 2).head(window)

        if metric == "oee":
            recent_val = recent["units_produced"].sum() / max(len(recent), 1)
            older_val = older["units_produced"].sum() / max(len(older), 1)
        else:
            recent_val = recent[metric].mean() if metric in recent.columns else 0
            older_val = older[metric].mean() if metric in older.columns else 0

        if older_val == 0:
            return 0
        return (recent_val - older_val) / older_val * 100

    def generate_report(self, df: pd.DataFrame, line_name: str,
                         target_rate: int = 100) -> dict:
        oee_metrics = self.calculate_oee(df, target_rate)
        yield_val = self.calculate_yield(df)
        defect_rate = self.calculate_defect_rate(df)
        mtbf_mttr = self.calculate_mtbf_mttr(df)
        trend = self.calculate_trend(df)

        running_df = df[df["status"] == "running"]
        total_units = int(running_df["units_produced"].sum())
        total_defects = int(running_df["units_defective"].sum())

        report = {
            "line": line_name,
            "generated_at": datetime.now().isoformat(),
            "total_units": total_units,
            "good_units": total_units - total_defects,
            "defective_units": total_defects,
            **oee_metrics,
            "yield": yield_val,
            "defect_rate": defect_rate["overall"],
            **mtbf_mttr,
            "trend_pct": trend,
        }
        return report


def format_metric(value: float, metric_type: str = "pct") -> str:
    if metric_type == "pct":
        return f"{value * 100:.1f}%"
    elif metric_type == "time":
        return f"{value:.1f} min"
    elif metric_type == "rate":
        return f"{value:.1f}/hr"
    return f"{value:.3f}"


def get_metric_color(value: float, thresholds: tuple = (0.6, 0.8)) -> str:
    if value < thresholds[0]:
        return "red"
    elif value < thresholds[1]:
        return "orange"
    return "green"
