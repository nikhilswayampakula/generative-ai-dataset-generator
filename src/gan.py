from __future__ import annotations
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, zdim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 28*28), nn.Sigmoid()
        )

    def forward(self, z):
        x = self.net(z).view(-1,1,28,28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
