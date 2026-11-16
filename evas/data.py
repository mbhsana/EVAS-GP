"""Data sets and leakage-free splits.

split(): stratified 60/20/20 train/validation/test split drawn with the run's seed; the
StandardScaler is fitted on the training part only. Exact duplicates are kept (as in the
original data) but their number is reported by describe().
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Highly imbalanced data sets of the imbalanced-learn benchmark (Ding, 2011) usable with --dataset imb:<name>
IMB_CANDIDATES = ["abalone_19", "protein_homo", "mammography", "ozone_level", "webpage", "yeast_me2"]


def load_creditcard(path):
    """Kaggle / ULB 'Credit Card Fraud Detection' (creditcard.csv): Time, V1..V28, Amount, Class."""
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("creditcard.csv must contain a 'Class' column")
    y = df["Class"].to_numpy().astype(int)
    X = df.drop(columns=["Class"])
    return X.to_numpy(dtype=float), y, list(X.columns)


def load_csv(path, target):
    df = pd.read_csv(path)
    y_raw = df[target].to_numpy()
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    classes, counts = np.unique(y_raw, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"binary target expected, found {len(classes)} classes")
    y = (y_raw == classes[np.argmin(counts)]).astype(int)   # minority class -> 1
    return X.to_numpy(dtype=float), y, list(X.columns)


def load_imblearn(name, cache_dir=None):
    """Benchmark of imbalanced-learn (needs `pip install imbalanced-learn` and internet once)."""
    from imblearn.datasets import fetch_datasets
    d = fetch_datasets(filter_data=(name,), data_home=cache_dir)[name]
    return np.asarray(d.data, dtype=float), (np.asarray(d.target) == 1).astype(int), None


def synthetic(n=60_000, d=20, pos_rate=0.005, seed=1234, class_sep=1.0):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=max(3, d // 3),
                               n_redundant=max(1, d // 6), n_clusters_per_class=3,
                               weights=[1 - pos_rate], flip_y=0.0, class_sep=class_sep, random_state=seed)
    return X, y, None


def synthetic_cc(seed=1234):
    """Synthetic surrogate with the shape of creditcard.csv (284,807 x 30, 492 positives)."""
    return synthetic(n=284_807, d=30, pos_rate=492 / 284_807, seed=seed, class_sep=1.5)


def load(name, csv=None, target="Class"):
    if name == "creditcard":
        if not csv or not os.path.exists(csv):
            raise FileNotFoundError("give --csv path/to/creditcard.csv")
        return load_creditcard(csv)
    if name == "synthetic":
        return synthetic()
    if name == "synthetic_cc":
        return synthetic_cc()
    if name.startswith("imb:"):
        return load_imblearn(name[4:])
    if name.startswith("csv:"):
        return load_csv(name[4:], target)
    raise ValueError(f"unknown data set {name!r}")


def split(X, y, seed=0, val=0.2, test=0.2, names=None, mode="stratified", time_col=None):
    """mode='stratified': random stratified split drawn with `seed`;
    mode='temporal'  : rows ordered by column `time_col` (e.g. creditcard 'Time'); the first 60%
                       form the training set, the next 20% validation, the last 20% test
                       (the split is then identical for all seeds; only the GP stream varies)."""
    if mode == "temporal":
        if time_col is None:
            raise ValueError("temporal split needs time_col")
        order = np.argsort(X[:, time_col], kind="stable")
        n = len(y); a = int(round((1 - val - test) * n)); b = int(round((1 - test) * n))
        itr, iva, ite = order[:a], order[a:b], order[b:]
        Xtr, ytr, Xva, yva, Xte, yte = X[itr], y[itr], X[iva], y[iva], X[ite], y[ite]
    else:
        Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=test, stratify=y, random_state=seed)
        Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=val / (1 - test), stratify=ytmp,
                                              random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return {"Xtr": sc.transform(Xtr), "ytr": ytr, "Xva": sc.transform(Xva), "yva": yva,
            "Xte": sc.transform(Xte), "yte": yte, "feature_names": names}


def describe(X, y):
    n1 = int(y.sum())
    dup = int(pd.DataFrame(X).duplicated().sum())
    return {"n": int(len(y)), "d": int(X.shape[1]), "n_pos": n1, "pos_rate": n1 / len(y),
            "IR": (len(y) - n1) / max(n1, 1), "duplicates": dup}
