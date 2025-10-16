import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from src.vae import VAE
from src.gan import Generator
from src.generate_tabular import gaussian_copula_synth
from src.utils import save_image_grid

st.set_page_config(page_title="Generative Dataset Studio", layout="wide")
st.title("🧬 Generative AI Dataset Studio (VAE + GAN)")

tab1, tab2 = st.tabs(["🖼 Image Generation", "📊 Tabular Synthesis"])

with tab1:
    st.subheader("Generate images with VAE or GAN")
    model = st.selectbox("Model", ["vae","gan"])
    dataset = st.selectbox("Dataset", ["MNIST","fashion"])
    n = st.slider("Number of samples", 4, 64, 16, step=4)
    zdim = st.number_input("Latent dim (VAE default 16, GAN min 64)", min_value=8, value=16, step=8)
    if st.button("Generate Images"):
        device = "cpu"
        if model=="vae":
            m = VAE(zdim=zdim)
            p = Path(f"models/vae_{dataset.lower()}.pt")
            if not p.exists():
                st.error("Train VAE first: python src/train_vae.py"); st.stop()
            m.load_state_dict(torch.load(p, map_location=device))
            with torch.no_grad():
                z = torch.randn(n, zdim)
                x = m.decode(z).cpu().numpy()
        else:
            zdim = max(64, int(zdim))
            G = Generator(zdim=zdim)
            p = Path(f"models/ganG_{dataset.lower()}.pt")
            if not p.exists():
                st.error("Train GAN first: python src/train_gan.py"); st.stop()
            G.load_state_dict(torch.load(p, map_location=device))
            with torch.no_grad():
                z = torch.randn(n, zdim)
                x = G(z).cpu().numpy()
        out = save_image_grid(x, grid=int(max(2, n**0.5)), out="data/synthetic/ui_grid.png")
        st.image(out, caption="Generated samples", use_column_width=True)

with tab2:
    st.subheader("Synthesize tabular data from a CSV (Gaussian Copula baseline)")
    csv = st.file_uploader("Upload CSV (optional; else uses sample_bank.csv)", type=["csv"])
    n = st.slider("Number of rows", 50, 5000, 200, step=50)
    if st.button("Generate Table"):
        if csv is not None:
            df = pd.read_csv(csv)
        else:
            df = pd.read_csv("data/tabular/sample_bank.csv")
        synth = gaussian_copula_synth(df, n)
        st.write("Preview", synth.head())
        st.download_button("Download CSV", synth.to_csv(index=False).encode("utf-8"), file_name="synthetic.csv", mime="text/csv")
