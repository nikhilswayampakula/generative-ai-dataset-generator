import pandas as pd
from src.generate_tabular import gaussian_copula_synth

def test_tabular_synth():
    df = pd.DataFrame({
        "age":[22,30,40],
        "income":[30000,50000,70000],
        "grade":["A","B","A"]
    })
    out = gaussian_copula_synth(df, 5)
    assert len(out)==5 and set(out.columns)==set(df.columns)
