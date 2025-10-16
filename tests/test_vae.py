import torch
from src.vae import VAE

def test_vae_forward():
    m = VAE(zdim=8)
    x = torch.rand(4,1,28,28)
    xhat, mu, logvar = m(x)
    assert xhat.shape == x.shape
