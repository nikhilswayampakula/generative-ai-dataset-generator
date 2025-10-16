from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import base64, io
import torch, pandas as pd

from src.vae import VAE
from src.gan import Generator
from src.generate_tabular import gaussian_copula_synth
from src.utils import save_image_grid

app = FastAPI(title="Generative Dataset API")

class TabularReq(BaseModel):
    n: int = 100
    csv_path: str = "data/tabular/sample_bank.csv"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/generate/images")
def gen_images(model: str = "vae", n: int = 16, grid: int = 4, dataset: str = "MNIST", zdim: int = 16):
    device = "cpu"
    if model == "vae":
        m = VAE(zdim=zdim)
        p = Path(f"models/vae_{dataset.lower()}.pt")
        if not p.exists():
            raise HTTPException(400, "VAE model not trained. Run train_vae.py first.")
        m.load_state_dict(torch.load(p, map_location=device))
        with torch.no_grad():
            z = torch.randn(n, zdim)
            x = m.decode(z).cpu().numpy()
    else:
        zdim = max(zdim, 64)
        G = Generator(zdim=zdim)
        p = Path(f"models/ganG_{dataset.lower()}.pt")
        if not p.exists():
            raise HTTPException(400, "GAN model not trained. Run train_gan.py first.")
        G.load_state_dict(torch.load(p, map_location=device))
        with torch.no_grad():
            z = torch.randn(n, zdim)
            x = G(z).cpu().numpy()

    path = save_image_grid(x, grid=grid, out="data/synthetic/api_grid.png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"image_base64": b64}

@app.post("/generate/tabular")
def gen_tabular(req: TabularReq):
    df = pd.read_csv(req.csv_path)
    out = gaussian_copula_synth(df, req.n)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return {"csv": buf.getvalue()}
