import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Optional


class ProductionSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.start_time = datetime.now() - timedelta(hours=8)
        self.current_time = self.start_time

    def generate_shift_data(self, hours: int = 8,
                             product_name: str = "Widget A",
                             target_rate: int = 100) -> pd.DataFrame:
        records = []
        current_time = self.current_time
        operational = True
        downtime_remaining = 0

        for minute in range(hours * 60):
            timestamp = current_time + timedelta(minutes=minute)

            if downtime_remaining > 0:
                downtime_remaining -= 1
                if downtime_remaining == 0:
                    operational = True
                records.append({
                    "timestamp": timestamp,
                    "product": product_name,
                    "status": "downtime",
                    "units_produced": 0,
                    "units_defective": 0,
                    "cycle_time_ms": 0,
                    "temperature": np.nan,
                    "vibration": np.nan,
                    "operator": self._get_operator(timestamp),
                })
                continue

            if operational and self.py_rng.random() < 0.002:
                downtime_remaining = self.py_rng.randint(5, 30)
                operational = False
                records.append({
                    "timestamp": timestamp,
                    "product": product_name,
                    "status": "downtime_start",
                    "units_produced": 0,
                    "units_defective": 0,
                    "cycle_time_ms": 0,
                    "temperature": np.nan,
                    "vibration": np.nan,
                    "operator": self._get_operator(timestamp),
                })
                continue

            base_cycle = 60000 / target_rate
            cycle_jitter = self.rng.randn() * 5
            cycle_time = max(base_cycle + cycle_jitter, 20)

            units = max(0, int(60 / (cycle_time / 1000)))

            defect_rate = 0.01 + 0.005 * np.sin(2 * np.pi * minute / 120)
            defect_rate += self.rng.randn() * 0.005
            defect_rate = max(0.001, min(0.15, defect_rate))

            if self.py_rng.random() < 0.001:
                defect_rate = self.rng.uniform(0.2, 0.5)

            defective = self.rng.binomial(units, defect_rate)

            temp = 75 + 5 * np.sin(2 * np.pi * minute / 240) + self.rng.randn() * 2
            vib = 0.5 + 0.3 * np.sin(2 * np.pi * minute / 180) + self.rng.randn() * 0.1

            records.append({
                "timestamp": timestamp,
                "product": product_name,
                "status": "running",
                "units_produced": units,
                "units_defective": defective,
                "cycle_time_ms": cycle_time,
                "temperature": temp,
                "vibration": vib,
                "operator": self._get_operator(timestamp),
            })

        self.current_time = current_time + timedelta(hours=hours)
        df = pd.DataFrame(records)
        return df

    def _get_operator(self, timestamp: datetime) -> str:
        operators = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        idx = (timestamp.hour * 60 + timestamp.minute) // 120 % len(operators)
        return operators[idx]

    def generate_multiline_data(self, lines: int = 3,
                                 hours: int = 8) -> dict:
        products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Component Z"]
        rates = [100, 80, 120, 90, 60]

        line_data = {}
        for i in range(lines):
            product = products[i % len(products)]
            rate = rates[i % len(rates)]
            df = self.generate_shift_data(
                hours=hours,
                product_name=product,
                target_rate=rate,
            )
            line_data[f"Line {i + 1}"] = {
                "product": product,
                "target_rate": rate,
                "data": df,
            }

        return line_data

    def generate_realtime_row(self, line_id: str = "Line 1",
                               product: str = "Widget A",
                               target_rate: int = 100) -> dict:
        is_running = self.py_rng.random() > 0.05
        row = {
            "timestamp": datetime.now(),
            "line": line_id,
            "product": product,
            "status": "running" if is_running else "idle",
        }

        if is_running:
            cycle_time = 60000 / target_rate + self.rng.randn() * 10
            cycle_time = max(20, cycle_time)
            units = max(0, int(60 / (cycle_time / 1000)))
            defect_rate = max(0, min(0.2, 0.02 + self.rng.randn() * 0.01))
            defective = self.rng.binomial(units, defect_rate)

            row.update({
                "units_produced": units,
                "units_defective": defective,
                "cycle_time_ms": cycle_time,
                "temperature": 75 + self.rng.randn() * 3,
                "vibration": 0.5 + abs(self.rng.randn()) * 0.2,
            })
        else:
            row.update({
                "units_produced": 0,
                "units_defective": 0,
                "cycle_time_ms": 0,
                "temperature": 70 + self.rng.randn() * 2,
                "vibration": 0.1 + abs(self.rng.randn()) * 0.05,
            })

        return row


class LineState:
    def __init__(self, line_id: str, product: str, target_rate: int):
        self.line_id = line_id
        self.product = product
        self.target_rate = target_rate
        self.history = []
        self.status = "idle"
        self.daily_total = 0
        self.daily_defects = 0
        self.daily_downtime = 0
        self.shift_start = datetime.now()

    def update(self, row: dict):
        self.history.append(row)
        self.status = row.get("status", "idle")
        self.daily_total += row.get("units_produced", 0)
        self.daily_defects += row.get("units_defective", 0)
        if row.get("status") == "downtime":
            self.daily_downtime += 1

    def get_oee(self) -> dict:
        total_time = (datetime.now() - self.shift_start).total_seconds() / 60
        if total_time == 0:
            return {"availability": 0, "performance": 0, "quality": 0, "oee": 0}

        availability = max(0, 1 - self.daily_downtime / total_time)
        theoretical_units = self.target_rate * (total_time / 60)
        performance = self.daily_total / max(theoretical_units, 1)
        quality = 1 - self.daily_defects / max(self.daily_total, 1)
        oee = availability * performance * quality

        return {
            "availability": availability,
            "performance": performance,
            "quality": quality,
            "oee": oee,
        }


if __name__ == "__main__":
    sim = ProductionSimulator()
    df = sim.generate_shift_data()
    print(f"Generated {len(df)} records")
    print(f"Total produced: {df['units_produced'].sum()}")
    print(f"Total defects: {df['units_defective'].sum()}")
