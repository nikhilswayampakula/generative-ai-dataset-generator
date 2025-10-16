from __future__ import annotations
import argparse, torch, numpy as np
from pathlib import Path
from src.vae import VAE
from src.gan import Generator
from src.utils import save_image_grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="vae", choices=["vae","gan"])
    ap.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST","fashion"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", type=str, default="data/synthetic/image_grid.png")
    ap.add_argument("--zdim", type=int, default=16)
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "vae":
        m = VAE(zdim=args.zdim).to(device)
        m.load_state_dict(torch.load(f"models/vae_{args.dataset.lower()}.pt", map_location=device))
        with torch.no_grad():
            z = torch.randn(args.n, args.zdim, device=device)
            x = m.decode(z).cpu().numpy()
    else:
        zdim = max(args.zdim, 64)
        G = Generator(zdim=zdim).to(device)
        G.load_state_dict(torch.load(f"models/ganG_{args.dataset.lower()}.pt", map_location=device))
        with torch.no_grad():
            z = torch.randn(args.n, zdim, device=device)
            x = G(z).cpu().numpy()

    path = save_image_grid(x, grid=int(max(2, args.n**0.5)), out=args.out)
    print(f"Saved image grid at {path}")

if __name__ == "__main__":
    main()
