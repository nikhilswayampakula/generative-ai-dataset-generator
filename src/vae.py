from __future__ import annotations
import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, zdim)
        self.logvar = nn.Linear(400, zdim)
        self.dec = nn.Sequential(
            nn.Linear(zdim, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec(z)
        return x.view(-1,1,28,28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar):
    bce = nn.functional.binary_cross_entropy(xhat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)
