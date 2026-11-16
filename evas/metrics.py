"""Fitness functions (computed on the current training sample) and evaluation metrics.

All fitness functions take a real score vector and binary labels. Threshold-dependent
fitnesses use the threshold 0 (i.e. sigmoid(f(x)) > 0.5); the AUC fitness is threshold-free.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)


# ============================================================================ fitness
def _counts(pred, y):
    pos = y == 1
    tp = int(np.sum(pred & pos)); fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & ~pos)); tn = int(np.sum(~pred & ~pos))
    return tp, fp, tn, fn


def fit_auc(scores, y):
    """Wilcoxon-Mann-Whitney estimate of the ROC AUC (ties count 1/2)."""
    pos = y == 1
    n1 = int(pos.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_gmean(scores, y):
    tp, fp, tn, fn = _counts(scores > 0, y)
    return float(np.sqrt(tp / max(tp + fn, 1) * tn / max(tn + fp, 1)))


def fit_bacc(scores, y):
    tp, fp, tn, fn = _counts(scores > 0, y)
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))


def fit_mcc(scores, y):
    tp, fp, tn, fn = (float(v) for v in _counts(scores > 0, y))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def fit_f1(scores, y):
    tp, fp, tn, fn = _counts(scores > 0, y)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def fit_ap(scores, y, sample_weight=None):
    """Average precision (area under the precision-recall curve), ties handled as in scikit-learn.
    With sample_weight, estimates the AP at the true prevalence from an under-sampled majority
    (each kept negative weighted by N_total/N_kept, synthetic cases excluded)."""
    if y.min() == y.max():
        return 0.0
    return float(average_precision_score(y, scores, sample_weight=sample_weight))


FITNESS = {"ap": fit_ap, "auc": fit_auc, "gmean": fit_gmean, "bacc": fit_bacc, "mcc": fit_mcc, "f1": fit_f1}


# ============================================================================ thresholds
def youden_threshold(scores, y):
    """Threshold maximising TPR - FPR (used for committee members on the current sample)."""
    if y.min() == y.max():
        return 0.0
    fpr, tpr, thr = roc_curve(y, scores)
    i = int(np.argmax(tpr - fpr))
    t = thr[i]
    if not np.isfinite(t):           # sklearn puts +inf as first threshold
        t = scores.max()
    # predictions are "score >= t" in roc_curve; we use "score > t'" -> shift just below t
    return float(np.nextafter(t, -np.inf))


def f1_threshold(scores, y):
    """Exact F1-optimal threshold (decision rule: score > threshold)."""
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    if len(f1) == 0:
        return float(np.max(scores)), 0.0
    i = int(np.argmax(f1))
    return float(np.nextafter(thr[i], -np.inf)), float(f1[i])


# ============================================================================ evaluation
def recall_at_fpr(scores, y, max_fpr=1e-3):
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def full_report(scores, y, thr):
    scores = np.asarray(scores, dtype=float)
    pred = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = float("nan")
    return {
        "auc_pr": float(average_precision_score(y, scores)),
        "auc_roc": float(auc),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "gmean": float(np.sqrt(tpr * tnr)),
        "precision": float(prec), "recall": float(tpr), "specificity": float(tnr),
        "recall_at_fpr1e-3": recall_at_fpr(scores, y, 1e-3),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
