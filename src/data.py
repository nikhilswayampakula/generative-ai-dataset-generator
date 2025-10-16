from __future__ import annotations
import torch
from torchvision import datasets, transforms

def mnist_loader(batch_size=128, image_set="MNIST"):
    tfm = transforms.Compose([transforms.ToTensor()])
    if image_set.lower() == "fashion":
        ds_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)
        ds_test  = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)
    else:
        ds_train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
        ds_test  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True),            torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
