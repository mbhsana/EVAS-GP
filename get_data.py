"""Download the Credit Card Fraud data set from OpenML (no Kaggle account needed) and check it.

python get_data.py            -> data/creditcard.csv
The file is accepted only if it has the expected shape of the ULB/Kaggle data set
(284,807 rows, Time, V1..V28, Amount, Class with 492 positives).
"""
import os
import sys

import numpy as np
from sklearn.datasets import fetch_openml

OUT = os.path.join("data", "creditcard.csv")
COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def main():
    if os.path.exists(OUT):
        print(f"{OUT} already exists"); return
    os.makedirs("data", exist_ok=True)
    ds = fetch_openml(data_id=1597, as_frame=True, parser="auto")   # OpenML 'creditcard'
    df = ds.frame
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(float).astype(int)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"unexpected columns, missing {missing}; download creditcard.csv from Kaggle instead")
    df = df[COLS]
    n, npos = len(df), int(df["Class"].sum())
    print(f"rows={n} positives={npos} rate={npos / n:.5f}")
    if n != 284_807 or npos != 492 or not np.isfinite(df.to_numpy(dtype=float)).all():
        sys.exit("data set does not match the ULB credit-card data; download creditcard.csv from Kaggle instead")
    df.to_csv(OUT, index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
