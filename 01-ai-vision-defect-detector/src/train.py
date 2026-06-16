import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import config
from model import LightweightDefectDetector, DefectClassifier


class DefectDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", img_size: int = 224):
        self.img_size = img_size
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []

        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        class_file = self.data_dir / "classes.txt"
        self.classes = []
        if class_file.exists():
            with open(class_file) as f:
                self.classes = [line.strip() for line in f]

        all_files = sorted(images_dir.glob("*.jpg"))
        split_idx = int(len(all_files) * 0.8)
        files = all_files[:split_idx] if split == "train" else all_files[split_idx:]

        for fpath in files:
            label_path = labels_dir / f"{fpath.stem}.txt"
            if label_path.exists():
                self.samples.append((fpath, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        with open(label_path) as f:
            parts = f.read().strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))

        return img, class_id, torch.tensor(bbox)


class Trainer:
    def __init__(self, model, device: str = None):
        self.model = model
        self.device = device or config.model.device
        self.model.to(self.device)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_bbox = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=0.0005
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.model.epochs
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels, bboxes in tqdm(loader, desc="Training"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            bboxes = bboxes.to(self.device)

            self.optimizer.zero_grad()
            cls_out, bbox_out = self.model(imgs)
            loss_cls = self.criterion_cls(cls_out, labels)
            loss_bbox = self.criterion_bbox(bbox_out, bboxes)
            loss = loss_cls + loss_bbox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(cls_out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(loader)
        return avg_loss, acc

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels, bboxes in tqdm(loader, desc="Validating"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                cls_out, bbox_out = self.model(imgs)
                loss_cls = self.criterion_cls(cls_out, labels)
                loss_bbox = self.criterion_bbox(bbox_out, bboxes)
                loss = loss_cls + loss_bbox

                total_loss += loss.item()
                _, preds = torch.max(cls_out, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(loader)
        return avg_loss, acc

    def save_checkpoint(self, path: str, epoch: int, val_acc: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }, str(path))
        print(f"Checkpoint saved: {path}")


def train():
    print(f"Device: {config.model.device}")
    print(f"Epochs: {config.model.epochs}")

    train_dataset = DefectDataset("data", split="train")
    val_dataset = DefectDataset("data", split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=config.model.batch_size,
        shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.model.batch_size,
        shuffle=False, num_workers=2
    )

    num_classes = len(train_dataset.classes) if train_dataset.classes else 8
    model = LightweightDefectDetector(num_classes=num_classes)
    trainer = Trainer(model)

    best_acc = 0.0
    for epoch in range(config.model.epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        trainer.scheduler.step()

        print(f"Epoch {epoch + 1}/{config.model.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            trainer.save_checkpoint(
                f"{config.model.weights_dir}/best_model.pt",
                epoch, val_acc
            )

    print(f"Training complete! Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    train()
