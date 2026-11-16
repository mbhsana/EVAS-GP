"""EVAS active sampler: builds S_e = U_e (majority) + D_tr^+ (all genuine positives) + Z_e (synthetic).

Committee signals
-----------------
The committee is a list of (tree, threshold) pairs: the q best *distinct* individuals of the
current population, each with its own decision threshold (Youden-optimal on the current
sample, so that the votes do not depend on the scale of the evolved scores). For an
instance x, v(x) is the fraction of members voting positive and u(x) = 1 - |2 v(x) - 1|
is the committee disagreement.
  - majority instance: v(x) = false-positive vote rate ("hardness")
  - minority instance: u(x) = uncertainty; 1 - v(x) = false-negative vote rate
An external, static scorer can replace the committee (ablation EVAS-*-RF): it is given as
two arrays (P(y=1) for every training negative and positive), computed out-of-bag by a random
forest trained once on D_tr, so that training instances are not scored by a model that has
memorised them.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .generators import GENERATORS, Neighbourhood
from .gp import evaluate


def committee_votes(committee, X):
    if not committee:
        return np.full(len(X), 0.5)
    v = np.zeros(len(X))
    for member, thr in committee:      # member: callable X -> scores, or a raw tree
        sc = member(X) if callable(member) else evaluate(member, X)
        v += sc > thr
    return v / len(committee)


class ActiveSampler:
    """See evas/config.py for the meaning and default value of every key of `cfg`."""

    def __init__(self, X_pos, X_neg, cfg, rng, external_scores=None):
        """external_scores: None or (p_neg, p_pos), P(y=1) for every row of X_neg / X_pos."""
        self.Xp, self.Xn, self.cfg, self.rng = X_pos, X_neg, cfg, rng
        self.nb = Neighbourhood(X_pos, X_neg, k=cfg["k_neighbors"], m=cfg["m_neighbors"], rng=rng)
        self.age = np.zeros(len(X_neg))
        self.n_events = 0
        self.external = external_scores
        cap = int(cfg["maj_cap_frac"] * len(X_neg))
        self.n_maj = int(max(1, min(cfg["n_maj"], cap)))
        self.signal_evals = 0          # number of (tree, instance) evaluations spent on signals

    # ------------------------------------------------------------------ signals
    def _votes(self, committee, idx, side):
        if self.external is not None:
            return self.external[0 if side == "neg" else 1][idx]
        X = (self.Xn if side == "neg" else self.Xp)[idx]
        self.signal_evals += len(committee) * len(X)
        return committee_votes(committee, X)

    def _active(self, committee):
        return self.external is not None or bool(committee)

    # ------------------------------------------------------------------ majority side
    def select_majority(self, committee):
        c, rng = self.cfg, self.rng
        n = self.n_maj
        info = {}
        if c["maj_mode"] == "random" or not self._active(committee):
            sel = rng.choice(len(self.Xn), n, replace=False)
        else:
            pool = np.arange(len(self.Xn))
            if c["pool_size"] and len(pool) > c["pool_size"]:
                pool = rng.choice(pool, c["pool_size"], replace=False)
            v = self._votes(committee, pool, "neg")
            u = 1.0 - np.abs(2.0 * v - 1.0)
            age = self.age[pool] / max(self.age.max(), 1.0)
            w = c["maj_lambda"] * v + (1 - c["maj_lambda"]) * u + c["age_weight"] * age + 1e-3
            n_rand = int(round(c["maj_random_frac"] * n))
            n_act = min(n - n_rand, len(pool))
            act = rng.choice(pool, n_act, replace=False, p=w / w.sum())
            mask = np.ones(len(self.Xn), bool); mask[act] = False
            rest = np.flatnonzero(mask)
            rnd = rng.choice(rest, min(n - n_act, len(rest)), replace=False)
            sel = np.r_[act, rnd]
            info = {"pool_fp_rate": float(v.mean()), "sel_fp_rate": float(v[np.isin(pool, act)].mean())}
        self.age += 1
        self.age[sel] = 0
        return sel, info

    # ------------------------------------------------------------------ minority pivots
    def select_pivots(self, committee):
        c, rng = self.cfg, self.rng
        allp = np.arange(len(self.Xp))
        mode = c["pivot_mode"]
        if mode == "all" or not self._active(committee):
            return allp
        v = self._votes(committee, allp, "pos")
        score = (1.0 - v) if mode == "hardness" else 1.0 - np.abs(2.0 * v - 1.0)
        n_keep = max(2, int(np.ceil(c["rho_unc"] * len(allp))))
        order = np.lexsort((rng.random(len(allp)), -score))   # random tie-breaking
        cand = allp[order[:n_keep]]
        if mode != "uncertainty_diversity":
            return cand
        k = min(c["n_clusters"], len(cand))
        if k < 2:
            return cand
        lab = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(2**31 - 1))).fit_predict(self.Xp[cand])
        n_div = max(k, int(np.ceil(c["rho_div"] * len(cand))))
        # proportional allocation across clusters, at least one pivot per cluster
        sizes = np.bincount(lab, minlength=k)
        alloc = np.maximum(1, np.floor(n_div * sizes / sizes.sum())).astype(int)
        piv = []
        for cl in range(k):
            members = cand[lab == cl]
            piv.extend(rng.choice(members, min(alloc[cl], len(members)), replace=False))
        return np.array(piv, dtype=int)

    # ------------------------------------------------------------------ generator schedule
    def generator_name(self):
        pool, sched = self.cfg["generators"], self.cfg["gen_schedule"]
        if sched == "cycle":
            return pool[self.n_events % len(pool)]
        if sched == "random":
            return pool[int(self.rng.integers(len(pool)))]
        return pool[0]

    # ------------------------------------------------------------------ build
    def build(self, committee):
        sel, maj_info = self.select_majority(committee)
        Xu = self.Xn[sel]
        n_target = int(round(self.cfg["minority_ratio"] * len(sel)))
        n_new = max(0, n_target - len(self.Xp))
        gname = self.generator_name() if n_new > 0 else None
        if gname is None:
            Z, piv, gmeta = np.empty((0, self.Xp.shape[1])), np.arange(0), {"seeds": 0, "fallback": False}
        else:
            piv = self.select_pivots(committee)
            Z, gmeta = GENERATORS[gname](self.rng, self.Xp, self.nb, piv, n_new)
        X = np.vstack([Xu, self.Xp, Z])
        y = np.r_[np.zeros(len(Xu)), np.ones(len(self.Xp) + len(Z))].astype(int)
        info = {"event": self.n_events, "generator": gname, "n_maj": int(len(Xu)),
                "n_pos": int(len(self.Xp)), "n_syn": int(len(Z)), "n_pivots": int(len(piv)),
                "n_seeds": int(gmeta["seeds"]), "gen_fallback": bool(gmeta["fallback"]),
                "active": self._active(committee), **maj_info}
        self.n_events += 1
        return X, y, info
