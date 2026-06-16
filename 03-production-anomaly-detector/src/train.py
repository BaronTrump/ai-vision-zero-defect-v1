import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import ConvAutoencoder, VariationalAutoencoder, LSTMAutoencoder


class Trainer:
    def __init__(self, model, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0

        for batch in tqdm(loader, desc="Training"):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            self.optimizer.zero_grad()

            if isinstance(self.model, VariationalAutoencoder):
                recon, mu, logvar, _ = self.model(x)
                recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + 0.1 * kl_loss) / x.size(0)
                kl_loss_total += kl_loss.item()

            elif isinstance(self.model, LSTMAutoencoder):
                recon, _ = self.model(x)
                loss = nn.functional.mse_loss(recon, x)

            else:
                recon, _ = self.model(x)
                loss = nn.functional.mse_loss(recon, x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            recon_loss_total += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)

                if isinstance(self.model, VariationalAutoencoder):
                    recon, mu, logvar, _ = self.model(x)
                    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = (recon_loss + 0.1 * kl_loss) / x.size(0)
                elif isinstance(self.model, LSTMAutoencoder):
                    recon, _ = self.model(x)
                    loss = nn.functional.mse_loss(recon, x)
                else:
                    recon, _ = self.model(x)
                    loss = nn.functional.mse_loss(recon, x)

                total_loss += loss.item()

        return total_loss / len(loader)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, str(path))
        print(f"Model saved to {path}")


def train_image_model():
    print("Loading image data...")
    data_dir = Path("data")
    normal_path = data_dir / "images" / "normal_imgs.npy"

    if not normal_path.exists():
        print("No pre-generated data found. Generating...")
        from data_generator import ProductionDataGenerator
        gen = ProductionDataGenerator()
        gen.generate_dataset()

    normal_imgs = np.load(str(normal_path))
    tensor_imgs = torch.from_numpy(normal_imgs).float()

    dataset = TensorDataset(tensor_imgs)
    split = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ConvAutoencoder(latent_dim=128, img_channels=3)
    trainer = Trainer(model)

    print("Training ConvAutoencoder...")
    for epoch in range(50):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        trainer.scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/50 | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    trainer.save("models/conv_ae.pt")


def train_sensor_model():
    import pandas as pd
    data_dir = Path("data")
    sensor_path = data_dir / "sensor" / "production_data.csv"

    if not sensor_path.exists():
        print("No sensor data found. Generating...")
        from data_generator import ProductionDataGenerator
        gen = ProductionDataGenerator()
        gen.generate_dataset()

    df = pd.read_csv(str(sensor_path))
    normal_df = df[~df["is_anomaly"]]

    sensor_cols = ["temperature", "pressure", "vibration_x", "vibration_y",
                   "vibration_z", "rpm", "current", "voltage"]
    data = normal_df[sensor_cols].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    seq_len = 32
    sequences = []
    for i in range(len(data_scaled) - seq_len):
        sequences.append(data_scaled[i:i + seq_len])

    sequences = np.array(sequences)
    tensor_seq = torch.from_numpy(sequences).float()

    dataset = TensorDataset(tensor_seq)
    split = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = LSTMAutoencoder(input_dim=len(sensor_cols), hidden_dim=64, latent_dim=32)
    trainer = Trainer(model)

    print("Training LSTM-Autoencoder for sensor data...")
    for epoch in range(30):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        trainer.scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/30 | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    trainer.save("models/lstm_ae.pt")
    import joblib
    joblib.dump(scaler, "models/sensor_scaler.pkl")
    print(f"Scaler saved to models/sensor_scaler.pkl")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("image", "all"):
        train_image_model()
    if mode in ("sensor", "all"):
        train_sensor_model()
