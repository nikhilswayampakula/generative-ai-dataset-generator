from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def gaussian_copula_synth(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # Simple numeric-only baseline: z-score, sample multivariate normal, invert
    num = df.select_dtypes(include=[np.number]).copy()
    mu = num.mean().values
    cov = np.cov(num.values.T)
    z = np.random.multivariate_normal(mu, cov, size=n)
    synth = pd.DataFrame(z, columns=num.columns)

    # attach categorical columns via bootstrap if present
    cat = df.select_dtypes(exclude=[np.number])
    for c in cat.columns:
        synth[c] = np.random.choice(cat[c].values, size=n, replace=True)
    return synth[df.columns] if not cat.empty else synth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/tabular/sample_bank.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=str, default="data/synthetic/synth.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_df = gaussian_copula_synth(df, args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote synthetic CSV to {args.out}")

if __name__ == "__main__":
    main()
