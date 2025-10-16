install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
train-vae:
	python src/train_vae.py --epochs 3
train-gan:
	python src/train_gan.py --epochs 3
gen-images:
	python src/generate_images.py --model vae --n 16
gen-tabular:
	python src/generate_tabular.py --csv data/tabular/sample_bank.csv --n 200 --out data/synthetic/bank_synth.csv
api:
	uvicorn app.api:app --reload --port 8000
ui:
	streamlit run app/streamlit_app.py
test:
	pytest -q
