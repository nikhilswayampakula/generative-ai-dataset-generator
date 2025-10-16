from __future__ import annotations
import argparse, torch
from torch import optim, nn
from src.data import mnist_loader
from src.gan import Generator, Discriminator
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST","fashion"])
    ap.add_argument("--models-dir", type=str, default="models")
    args = ap.parse_args()

    train_loader, _ = mnist_loader(batch_size=128, image_set=args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(zdim=args.zdim).to(device)
    D = Discriminator().to(device)
    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    for epoch in range(1, args.epochs+1):
        for x,_ in train_loader:
            x = x.to(device)
            b = x.size(0)
            real = torch.ones(b,1, device=device)
            fake = torch.zeros(b,1, device=device)

            # Train D
            z = torch.randn(b, args.zdim, device=device)
            xf = G(z).detach()
            D_real = D(x)
            D_fake = D(xf)
            lossD = bce(D_real, real) + bce(D_fake, fake)
            optD.zero_grad(); lossD.backward(); optD.step()

            # Train G
            z = torch.randn(b, args.zdim, device=device)
            xg = G(z)
            lossG = bce(D(xg), real)
            optG.zero_grad(); lossG.backward(); optG.step()
        print(f"Epoch {epoch}: lossD={lossD.item():.4f} lossG={lossG.item():.4f}")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), f"{args.models_dir}/ganG_{args.dataset.lower()}.pt")
    torch.save(D.state_dict(), f"{args.models_dir}/ganD_{args.dataset.lower()}.pt")
    print("Saved GAN models.")

if __name__ == "__main__":
    main()
