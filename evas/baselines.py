"""Non-evolutionary baselines evaluated with exactly the same protocol as the GP methods:
same splits (same seed), rebalancing on the training part only, threshold chosen on validation
(F1-optimal), test set used once.

Classifiers : LR, DT, RF, GB (XGBoost if installed, else sklearn HistGradientBoosting).
Rebalancing : none, class weighting, RUS, ROS, SMOTE, Borderline-SMOTE, ADASYN, Safe-Level-SMOTE
              (full rebalancing, as in common practice; same generator code as EVAS).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .generators import GENERATORS, Neighbourhood
from .metrics import f1_threshold, full_report

REBALANCING = ["none", "weight", "rus", "ros", "smote", "borderline", "adasyn", "safelevel"]


def has_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def make_classifier(name, seed, weighted=False, pos_weight=1.0, n_estimators=200):
    cw = "balanced" if weighted else None
    if name == "LR":
        return LogisticRegression(max_iter=5000, class_weight=cw)
    if name == "DT":
        return DecisionTreeClassifier(class_weight=cw, min_samples_leaf=1, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=n_estimators, class_weight=cw, n_jobs=1, random_state=seed)
    if name == "GB":
        if has_xgboost():
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="aucpr", n_jobs=1, random_state=seed,
                                 scale_pos_weight=pos_weight if weighted else 1.0)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight=cw,
                                              random_state=seed)
    raise ValueError(name)


CLASSIFIERS = ["LR", "RF", "GB"]


def rebalance(X, y, how, rng):
    if how in ("none", "weight"):
        return X, y
    Xp, Xn = X[y == 1], X[y == 0]
    if how == "rus":
        sel = rng.choice(len(Xn), len(Xp), replace=False)
        return np.vstack([Xn[sel], Xp]), np.r_[np.zeros(len(Xp)), np.ones(len(Xp))].astype(int)
    n_new = len(Xn) - len(Xp)
    if how == "ros":
        Z = Xp[rng.choice(len(Xp), n_new, replace=True)]
    else:
        nb = Neighbourhood(Xp, Xn, rng=rng)
        Z, _ = GENERATORS[how](rng, Xp, nb, np.arange(len(Xp)), n_new)
    return np.vstack([X, Z]), np.r_[y, np.ones(len(Z))].astype(int)


def run_baseline(data, clf_name, how, seed=0, n_estimators=200):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    X, y = rebalance(data["Xtr"], data["ytr"], how, rng)
    pw = (len(y) - y.sum()) / max(y.sum(), 1)
    clf = make_classifier(clf_name, seed, weighted=(how == "weight"), pos_weight=pw,
                          n_estimators=n_estimators).fit(X, y)
    train_time = time.perf_counter() - t0
    s_va = clf.predict_proba(data["Xva"])[:, 1]
    thr, _ = f1_threshold(s_va, data["yva"])
    gb_impl = ("xgboost" if has_xgboost() else "sklearn-hgb") if clf_name == "GB" else None
    return {"test": full_report(clf.predict_proba(data["Xte"])[:, 1], data["yte"], thr),
            "val": full_report(s_va, data["yva"], thr),
            "train": full_report(clf.predict_proba(data["Xtr"])[:, 1], data["ytr"], thr),
            "threshold": thr, "train_time_s": train_time, "gb_impl": gb_impl,
            "n_train_rebalanced": int(len(y))}
