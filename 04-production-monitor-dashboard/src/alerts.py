from collections import deque
from datetime import datetime
from typing import Callable, Optional
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertRule:
    def __init__(self, name: str, metric: str, operator: str,
                 threshold: float, severity: str = "warning",
                 message: str = ""):
        self.name = name
        self.metric = metric
        self.operator = operator
        self.threshold = threshold
        self.severity = severity
        self.message = message
        self.triggered = False
        self.last_triggered = None

    def evaluate(self, value: float) -> bool:
        if self.operator == ">":
            triggered = value > self.threshold
        elif self.operator == "<":
            triggered = value < self.threshold
        elif self.operator == ">=":
            triggered = value >= self.threshold
        elif self.operator == "<=":
            triggered = value <= self.threshold
        elif self.operator == "==":
            triggered = value == self.threshold
        else:
            triggered = False

        if triggered and not self.triggered:
            self.triggered = True
            self.last_triggered = datetime.now()
            return True
        elif not triggered:
            self.triggered = False

        return False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
        }


class AlertManager:
    def __init__(self):
        self.rules: list[AlertRule] = []
        self.alerts: deque = deque(maxlen=500)
        self.notifiers: list[Callable] = []

        self._setup_default_rules()

    def _setup_default_rules(self):
        self.add_rule(AlertRule(
            "High Defect Rate", "defect_rate", ">", 0.05,
            "critical", "Defect rate exceeded 5%"
        ))
        self.add_rule(AlertRule(
            "Low OEE", "oee", "<", 0.6,
            "critical", "OEE dropped below 60%"
        ))
        self.add_rule(AlertRule(
            "High Temperature", "temperature", ">", 85.0,
            "warning", "Temperature exceeded 85°C"
        ))
        self.add_rule(AlertRule(
            "High Vibration", "vibration", ">", 1.0,
            "warning", "Vibration level exceeded 1.0 mm/s"
        ))
        self.add_rule(AlertRule(
            "Low Performance", "performance", "<", 0.7,
            "critical", "Performance below 70%"
        ))
        self.add_rule(AlertRule(
            "Low Quality", "quality", "<", 0.95,
            "warning", "Quality below 95%"
        ))

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def remove_rule(self, name: str):
        self.rules = [r for r in self.rules if r.name != name]

    def evaluate_all(self, metrics: dict) -> list[dict]:
        triggered = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is not None and rule.evaluate(value):
                alert = {
                    "rule": rule.name,
                    "severity": rule.severity,
                    "message": rule.message,
                    "value": value,
                    "threshold": rule.threshold,
                    "timestamp": datetime.now().isoformat(),
                }
                self.alerts.append(alert)
                triggered.append(alert)

                for notifier in self.notifiers:
                    try:
                        notifier(alert)
                    except Exception as e:
                        print(f"Notifier failed: {e}")

        return triggered

    def get_recent_alerts(self, n: int = 50) -> list:
        return list(self.alerts)[-n:]

    def get_alert_count_by_severity(self) -> dict:
        counts = {"critical": 0, "warning": 0, "info": 0}
        for alert in self.alerts:
            sev = alert.get("severity", "info")
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def add_email_notifier(self, smtp_server: str, smtp_port: int,
                            username: str, password: str,
                            recipients: list[str]):
        def email_notifier(alert: dict):
            msg = MIMEMultipart()
            msg["From"] = username
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[{alert['severity'].upper()}] {alert['rule']}"

            body = f"""
Alert: {alert['rule']}
Severity: {alert['severity']}
Message: {alert['message']}
Value: {alert['value']:.4f}
Threshold: {alert['threshold']}
Time: {alert['timestamp']}
            """
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

        self.notifiers.append(email_notifier)

    def add_webhook_notifier(self, url: str):
        import requests

        def webhook_notifier(alert: dict):
            try:
                requests.post(url, json=alert, timeout=5)
            except Exception as e:
                print(f"Webhook failed: {e}")

        self.notifiers.append(webhook_notifier)

    def to_config(self) -> dict:
        return {
            "rules": [r.to_dict() for r in self.rules],
        }


class AlertHistory:
    def __init__(self):
        self.alerts = deque(maxlen=10000)

    def add(self, alert: dict):
        self.alerts.append(alert)

    def get_summary(self, hours: int = 24) -> dict:
        cutoff = datetime.now().timestamp() - hours * 3600
        recent = [a for a in self.alerts
                  if datetime.fromisoformat(a["timestamp"]).timestamp() > cutoff]

        summary = {"total": len(recent), "critical": 0, "warning": 0, "info": 0}
        for a in recent:
            summary[a.get("severity", "info")] += 1

        return summary
