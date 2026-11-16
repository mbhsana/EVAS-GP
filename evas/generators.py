"""SMOTE-family synthetic minority generators that start from an arbitrary pivot subset.

Re-implemented (rather than calling imbalanced-learn) because active pivot selection needs
generators that seed new points from a chosen subset of the minority class while taking
neighbours in the whole minority class. Each generator follows its original definition:

- SMOTE            (Chawla et al., 2002): x + u (x_nn - x), u ~ U(0,1), x_nn among the k
                    minority nearest neighbours of x.
- Borderline-SMOTE (Han et al., 2005, variant 1): only "danger" pivots seed new points,
                    i.e. m/2 <= #majority among the m nearest neighbours (all classes) < m.
- ADASYN           (He et al., 2008): the number of points seeded by x is proportional to
                    r(x) = #majority among its k nearest neighbours (all classes) / k.
- Safe-Level-SMOTE (Bunkhumpornpat et al., 2009): the interpolation gap depends on the
                    ratio of the safe levels (#minority among the k nearest neighbours) of
                    x and of the chosen minority neighbour.

Neighbourhoods are computed once, on the training set only.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Neighbourhood:
    """Pre-computed neighbourhood information for the minority instances of the training set.

    min_nn[i]   : indices (into X_pos) of the k nearest *minority* neighbours of positive i
    maj_m[i]    : #majority instances among the m nearest neighbours (all classes) of positive i
    maj_k[i]    : #majority instances among the k nearest neighbours (all classes)
    safe[i]     : #minority instances among the k nearest neighbours (all classes) = safe level
    """

    def __init__(self, X_pos, X_neg, k=5, m=10, max_neg=300_000, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        if len(X_pos) < 2:
            raise ValueError("at least two minority instances are needed")
        self.k = min(k, len(X_pos) - 1)
        self.m = m
        nn_pos = NearestNeighbors(n_neighbors=self.k + 1).fit(X_pos)
        idx_p = nn_pos.kneighbors(X_pos, return_distance=False)
        # drop the query point itself (robust to duplicated rows)
        self.min_nn = np.array([row[row != i][:self.k] for i, row in enumerate(idx_p)])
        self.subsampled_neg = len(X_neg) > max_neg
        if self.subsampled_neg:
            X_neg = X_neg[rng.choice(len(X_neg), max_neg, replace=False)]
        X_all = np.vstack([X_pos, X_neg])
        lab = np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))]
        kk = max(self.m, self.k)
        nn_all = NearestNeighbors(n_neighbors=kk + 1).fit(X_all)
        idx = nn_all.kneighbors(X_pos, return_distance=False)
        # drop the query point itself (first neighbour of a training point is itself)
        idx = np.array([row[row != i][:kk] for i, row in enumerate(idx)])
        self.maj_m = (lab[idx[:, :self.m]] == 0).sum(1)
        self.maj_k = (lab[idx[:, :self.k]] == 0).sum(1)
        self.safe = self.k - self.maj_k


# ============================================================================ helpers
def _neighbours(rng, nb, base):
    return nb.min_nn[base, rng.integers(nb.min_nn.shape[1], size=len(base))]


def _interp(rng, X_pos, base, nbr, gap=None):
    if gap is None:
        gap = rng.random(len(base))
    return X_pos[base] + gap[:, None] * (X_pos[nbr] - X_pos[base])


def danger_mask(nb, idx):
    c = nb.maj_m[idx]
    return (c >= nb.m / 2) & (c < nb.m)


# ============================================================================ generators
# Every generator returns (Z, meta) where Z has n_new rows and meta records what happened.
def gen_smote(rng, X_pos, nb, pivots, n_new):
    base = rng.choice(pivots, size=n_new, replace=True)
    return _interp(rng, X_pos, base, _neighbours(rng, nb, base)), {"seeds": len(np.unique(base)), "fallback": False}


def gen_borderline(rng, X_pos, nb, pivots, n_new):
    danger = pivots[danger_mask(nb, pivots)]
    if len(danger) == 0:  # no borderline pivot: fall back to SMOTE (logged)
        Z, meta = gen_smote(rng, X_pos, nb, pivots, n_new)
        meta["fallback"] = True
        return Z, meta
    Z, meta = gen_smote(rng, X_pos, nb, danger, n_new)
    return Z, meta


def gen_adasyn(rng, X_pos, nb, pivots, n_new):
    r = nb.maj_k[pivots] / nb.k
    if r.sum() == 0:
        Z, meta = gen_smote(rng, X_pos, nb, pivots, n_new)
        meta["fallback"] = True
        return Z, meta
    base = rng.choice(pivots, size=n_new, replace=True, p=r / r.sum())
    return _interp(rng, X_pos, base, _neighbours(rng, nb, base)), {"seeds": len(np.unique(base)), "fallback": False}


def safe_level_gap(rng, sl_p, sl_n):
    """Gap rule of Safe-Level-SMOTE; returns (gap, valid) arrays."""
    sl_p = np.asarray(sl_p, float); sl_n = np.asarray(sl_n, float)
    gap = np.empty(len(sl_p)); valid = np.ones(len(sl_p), bool)
    both0 = (sl_p == 0) & (sl_n == 0)
    n0 = (sl_n == 0) & ~both0                       # neighbour is noise-like: duplicate x
    eq = (sl_p == sl_n) & ~both0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sl_n > 0, sl_p / np.where(sl_n > 0, sl_n, 1), np.inf)
    gt = (ratio > 1) & ~n0 & ~both0
    lt = (ratio < 1) & ~both0
    u = rng.random(len(sl_p))
    gap[both0] = 0.0; valid[both0] = False
    gap[n0] = 0.0
    gap[eq] = u[eq]
    gap[gt] = u[gt] / ratio[gt]                     # U(0, 1/ratio)
    gap[lt] = 1 - ratio[lt] + u[lt] * ratio[lt]     # U(1-ratio, 1)
    return gap, valid


def gen_safelevel(rng, X_pos, nb, pivots, n_new, max_rounds=20):
    out_b, out_n, out_g = [], [], []
    need = n_new
    for _ in range(max_rounds):
        if need == 0:
            break
        base = rng.choice(pivots, size=need, replace=True)
        nbr = _neighbours(rng, nb, base)
        gap, ok = safe_level_gap(rng, nb.safe[base], nb.safe[nbr])
        out_b.append(base[ok]); out_n.append(nbr[ok]); out_g.append(gap[ok])
        need -= int(ok.sum())
    fallback = need > 0
    if fallback:  # pivots are all noise for Safe-Level: complete with SMOTE (logged)
        base = rng.choice(pivots, size=need, replace=True)
        out_b.append(base); out_n.append(_neighbours(rng, nb, base)); out_g.append(rng.random(need))
    base = np.concatenate(out_b); nbr = np.concatenate(out_n); gap = np.concatenate(out_g)
    return _interp(rng, X_pos, base, nbr, gap), {"seeds": len(np.unique(base)), "fallback": bool(fallback)}


GENERATORS = {"smote": gen_smote, "borderline": gen_borderline,
              "adasyn": gen_adasyn, "safelevel": gen_safelevel}
