# 🧬 Building a Generative AI Model from Scratch to Generate Datasets

A **from-scratch Generative AI toolkit** that trains **VAE** and **GAN** models to create
**synthetic datasets** for images and tabular data. Use it to generate realistic samples
for testing, privacy-preserving ML, or simulation.

- 🖼 **Images**: Train **VAE** or **DCGAN** on MNIST/Fashion‑MNIST and sample new images
- 📊 **Tabular**: Learn simple probabilistic models from CSV and synthesize rows
- ⚡ **Real-time**: **FastAPI** endpoints to generate images/CSV on demand
- 🧪 **Demo**: **Streamlit** UI to preview images and export CSV
- 🧰 **Production**: MIT license, tests, CI, Docker, Makefile, clean structure

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Train models (downloads MNIST automatically)
python src/train_vae.py --epochs 3
python src/train_gan.py --epochs 3

# 2) Generate samples
python src/generate_images.py --model vae --n 16
python src/generate_tabular.py --csv data/tabular/sample_bank.csv --n 200 --out data/synthetic/bank_synth.csv

# 3) Run API and UI
uvicorn app.api:app --reload --port 8000
streamlit run app/streamlit_app.py
```

## API Endpoints
- `GET /health`
- `GET /generate/images?model=vae&n=16&grid=4` → base64 PNG grid
- `POST /generate/tabular` with JSON `{"n":100,"csv_path":"data/tabular/sample_bank.csv"}` → CSV bytes

## Project Structure
```
generative-ai-dataset-generator/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── data/
│   ├── tabular/sample_bank.csv
│   └── synthetic/  # generated outputs
├── models/         # checkpoints will be saved here
├── src/
│   ├── vae.py
│   ├── gan.py
│   ├── data.py
│   ├── utils.py
│   ├── train_vae.py
│   ├── train_gan.py
│   ├── generate_images.py
│   └── generate_tabular.py
├── tests/
│   ├── test_vae.py
│   └── test_tabular.py
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
├── requirements.txt
├── setup.cfg
└── LICENSE
```

## Why Synthetic Data?
- **Privacy**: shareable datasets without exposing real records (e.g., healthcare)
- **Coverage**: create rare scenarios for autonomous systems
- **Balance**: augment minority classes to fight class imbalance

## Notes
- Default datasets are small and auto-downloaded; no manual data needed.
- For tabular generation we provide a lightweight **Gaussian copula** baseline.
- Swap the image dataset or plug in your CSVs to adapt to your domain.

**License:** MIT  
**Author:** Nikhil Swayampakula · LinkedIn: https://www.linkedin.com/in/nikhil-swa-47b479366/
