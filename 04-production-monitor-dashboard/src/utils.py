import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional


def export_to_csv(df, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Exported to {path}")
    return path


def export_to_json(data, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Exported to {path}")
    return path


def generate_report_filename(prefix: str = "production_report",
                              extension: str = "csv") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_shift_report(self, metrics: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"shift_report_{timestamp}.json"

        report = {
            "type": "shift_report",
            "generated_at": datetime.now().isoformat(),
            **metrics,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return str(path)

    def generate_daily_summary(self, line_data: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"daily_summary_{timestamp}.json"

        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "lines": line_data,
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return str(path)
