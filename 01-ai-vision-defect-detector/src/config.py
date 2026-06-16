from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DefectConfig:
    defect_types: List[str] = field(default_factory=lambda: [
        "scratch", "dent", "crack", "discoloration",
        "missing_component", "deformation", "contamination"
    ])
    defect_probability: float = 0.3
    severity_range: tuple = (0.3, 1.0)


@dataclass
class DataConfig:
    img_size: int = 640
    output_dir: str = "data"
    train_ratio: float = 0.8
    samples_per_class: int = 500
    background_color: tuple = (200, 200, 200)
    part_size_range: tuple = (100, 250)


@dataclass
class ModelConfig:
    model_type: str = "yolov8n"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    weights_dir: str = "models"


@dataclass
class InferenceConfig:
    source: str = "0"
    show_conf: bool = True
    show_labels: bool = True
    line_width: int = 2
    save_output: bool = True
    output_dir: str = "runs/detect"


@dataclass
class DashboardConfig:
    port: int = 8501
    update_interval: float = 0.1
    max_history: int = 1000
    alert_email: str = ""
    alert_webhook: str = ""


@dataclass
class AppConfig:
    defect: DefectConfig = field(default_factory=DefectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "AppConfig":
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg) if cfg else cls()


config = AppConfig()
