# Real-time Production Monitor Dashboard

Centralized AI-powered monitoring dashboard for production lines — integrates defect detection, robot vision, and anomaly detection into a single operations interface.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Project 1      │    │  Project 2       │    │  Project 3      │
│  Defect Detector│───▶│  Robot Vision    │───▶│  Anomaly        │
└─────────────────┘    └──────────────────┘    │  Detector       │
                                                └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Alert System   │◀───│  MAIN            │◀───│  Data Pipeline  │
│  (Email/Slack)  │    │  CONTROL DASHBOARD│    │  (Aggregation)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

- **Unified Dashboard** — single interface for all production line AI systems
- **Real-time OEE Tracking** — Overall Equipment Effectiveness metrics
- **Production KPIs** — throughput, yield, defect rate, downtime
- **Multi-line Support** — monitor multiple production lines simultaneously
- **Alert Configuration** — customizable alert rules (email, Slack, webhook)
- **Historical Analytics** — trend analysis with configurable date ranges
- **Report Generation** — auto-generated shift/daily/weekly reports
- **User Roles** — operator, supervisor, admin views
- **Dark/Light Mode** — theme support for control room environments

## Quick Start

```bash
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard.py

# Run data simulator separately
python src/data_simulator.py
```

## Project Structure

```
├── src/
│   ├── dashboard.py        # Main Streamlit dashboard
│   ├── data_simulator.py   # Production data simulation
│   ├── metrics.py          # KPI calculation engine
│   ├── alerts.py           # Alert management system
│   └── utils.py            # Database, export utilities
├── config.yaml             # Dashboard configuration
├── requirements.txt
└── README.md
```

## KPI Dashboard

| Metric | Description |
|--------|-------------|
| OEE | Overall Equipment Effectiveness |
| Throughput | Units per hour |
| Yield | Good units / Total units |
| Defect Rate | Defective units / Total units |
| MTBF | Mean Time Between Failures |
| MTTR | Mean Time To Repair |
| Availability | Uptime / Total time |
| Performance | Actual speed / Ideal speed |

## [Company Name] Integration

- **Control Room Display**: Full-screen dashboard for 24/7 monitoring
- **Shift Reports**: Automated end-of-shift performance summaries
- **Alert Routing**: Critical alerts to supervisor mobile devices
- **Data Export**: CSV/JSON export for external analytics
- **PLC Integration**: Connect to existing PLCs via Modbus/OPC-UA
