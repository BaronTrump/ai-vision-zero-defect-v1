import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from config import config


def visualize_detection(img: np.ndarray, detections: list, save_path: str = None):
    img_copy = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det["class_name"]
        conf = det["confidence"]
        is_defect = cls_name != "none" and cls_name != 0
        color = (0, 255, 0) if not is_defect else (0, 0, 255)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, img_copy)
    return img_copy


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def compute_metrics(detections: list, ground_truth: list):
    tp = fp = fn = 0
    for pred, gt in zip(detections, ground_truth):
        if pred["class_id"] == gt["class_id"] and pred["confidence"] >= 0.5:
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou >= 0.5:
                tp += 1
            else:
                fp += 1
                fn += 1
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def generate_report(stats: dict, output_dir: str = "reports"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("AI VISION DEFECT DETECTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Inspected: {stats.get('total', 0)}\n")
        f.write(f"Passed: {stats.get('passed', 0)}\n")
        f.write(f"Failed: {stats.get('failed', 0)}\n")
        f.write(f"Defect Rate: {stats.get('failed', 0) / max(stats.get('total', 1), 1) * 100:.2f}%\n\n")
        f.write("Defect Breakdown:\n")
        for defect, count in stats.get("defect_counts", {}).items():
            f.write(f"  - {defect}: {count}\n")

    print(f"Report saved to {report_path}")
    return report_path
