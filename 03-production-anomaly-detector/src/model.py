import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, img_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def get_latent(self, x):
        return self.encoder(x)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, img_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 1, 1)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64,
                 latent_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, input_dim, num_layers, batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        enc_out, (hidden, _) = self.encoder_lstm(x)
        z = self.encoder_fc(enc_out[:, -1, :])

        dec_input = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)

        return dec_out, z

    def get_latent(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        return self.encoder_fc(hidden[-1])


class AnomalyScorer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.normal_scores = []
        self.threshold = None

    def score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            if isinstance(self.model, VariationalAutoencoder):
                recon, _, _, _ = self.model(x)
            else:
                recon, _ = self.model(x)
            mse = F.mse_loss(recon, x, reduction='none')
            if mse.dim() == 4:
                mse = mse.view(mse.size(0), -1).mean(dim=1)
            elif mse.dim() == 3:
                mse = mse.view(mse.size(0), -1).mean(dim=1)
            return mse

    def fit_threshold(self, loader, percentile: float = 95.0):
        scores = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch_scores = self.score(batch)
            scores.extend(batch_scores.cpu().numpy())
        self.normal_scores = scores
        self.threshold = np.percentile(scores, percentile)
        print(f"Anomaly threshold set at {self.threshold:.6f} (P{percentile})")
        return self.threshold

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x)
        anomalies = scores > self.threshold if self.threshold else scores > 0.5
        return anomalies, scores


def get_anomaly_model(model_type: str = "conv_ae", **kwargs):
    models = {
        "conv_ae": ConvAutoencoder,
        "vae": VariationalAutoencoder,
        "lstm_ae": LSTMAutoencoder,
    }
    model_class = models.get(model_type, ConvAutoencoder)
    return model_class(**kwargs)
