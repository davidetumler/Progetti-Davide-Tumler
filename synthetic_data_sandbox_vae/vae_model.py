"""VAE denso implementato con PyTorch, adatto a piccole tabelle demo."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class TabularVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, input_dim), nn.Sigmoid())
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu, logvar = self.mu(hidden), self.logvar(hidden)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std  # reparameterization trick
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, rows: int) -> np.ndarray:
        self.eval()
        z = torch.randn(rows, self.latent_dim)
        return self.decoder(z).cpu().numpy()


def train_vae(x: np.ndarray, latent_dim: int = 4, epochs: int = 25,
              beta: float = 0.01) -> tuple[TabularVAE, list[float]]:
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = TabularVAE(x.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                        batch_size=min(64, len(x)), shuffle=True)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            reconstructed, mu, logvar = model(batch)
            mse = F.mse_loss(reconstructed, batch, reduction='sum') / len(batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / len(batch)
            loss = mse + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        history.append(round(total / len(x), 5))
    return model, history
