from __future__ import annotations
import argparse, torch
from torch import optim
from src.data import mnist_loader
from src.vae import VAE, vae_loss
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST","fashion"])
    ap.add_argument("--models-dir", type=str, default="models")
    args = ap.parse_args()

    train_loader, _ = mnist_loader(batch_size=128, image_set=args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(zdim=args.zdim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, args.epochs+1):
        total = 0.0
        for x,_ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            xhat, mu, logvar = model(x)
            loss = vae_loss(x, xhat, mu, logvar)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"Epoch {epoch}: loss={total/len(train_loader):.4f}")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{args.models_dir}/vae_{args.dataset.lower()}.pt")
    print("Saved VAE model.")

if __name__ == "__main__":
    main()
