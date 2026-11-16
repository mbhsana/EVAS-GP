"""EVAS main loop: GP with dynamic, active and variable construction of the training sample.

Strict data roles
-----------------
  train      : the only data the evolutionary process can see (through the sample S_e);
  validation : used once, after evolution, to choose the returned model among the archived
               candidates, to calibrate the ensemble and to set the decision thresholds;
  test       : used once, to report the results.
tests/test_all.py checks that the whole evolutionary trajectory is unchanged when the
validation and test sets are replaced by noise.

Two returned models
-------------------
  single  : the archived individual with the best validation AUC-PR;
  ensemble: the ens_k archived individuals with the best validation AUC-PR (from different
            checkpoints, i.e. samples built with different generators), each score
            mapped to [0,1] by its empirical CDF on the validation set, then averaged.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score

from . import gp
from .engines import MultiGeneEngine, TreeEngine
from .metrics import FITNESS, f1_threshold, fit_ap, full_report, youden_threshold
from .sampler import ActiveSampler


def make_engine(cfg, n_features):
    gcfg = gp.GPConfig(n_features=n_features, pop_size=cfg["pop_size"], max_depth=cfg["max_depth"],
                       init_min_depth=cfg["init_min_depth"], init_max_depth=cfg["init_max_depth"],
                       p_cx=cfg["p_cx"], p_mut=cfg["p_mut"], tournament=cfg["tournament"],
                       elites=cfg["elites"])
    if cfg["engine"] == "tree":
        return TreeEngine(gcfg)
    if cfg["engine"] == "mggp":
        return MultiGeneEngine(gcfg, max_genes=cfg["max_genes"], ridge=cfg["ridge"],
                               base_features=cfg["base_features"], gene_max_depth=cfg["gene_max_depth"])
    raise ValueError(cfg["engine"])


def _distinct_best(E, pop, fits, k):
    out, seen = [], set()
    for i in np.argsort(-fits, kind="stable"):
        kk = E.key(pop[i])
        if kk not in seen:
            seen.add(kk); out.append(i)
        if len(out) == k:
            break
    return out


def _rf_oob_scores(Xtr, ytr, seed):
    """Out-of-bag P(y=1) of a random forest trained once on D_tr (external committee)."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, oob_score=True, class_weight="balanced_subsample",
                                min_samples_leaf=2, n_jobs=1, random_state=seed).fit(Xtr, ytr)
    p = np.nan_to_num(rf.oob_decision_function_[:, 1], nan=0.5)
    return p[ytr == 0], p[ytr == 1]


class _Scored:
    """A committee member: score() + its own threshold (duck-typed for sampler.committee_votes)."""

    def __init__(self, E, ind, params):
        self.E, self.ind, self.params = E, ind, params

    def __call__(self, X):
        return self.E.score(self.ind, self.params, X)


def _ecdf(ref):
    ref = np.sort(ref)
    return lambda s: np.searchsorted(ref, s, side="right") / len(ref)


def run_gp(data, cfg, seed=0, verbose=False):
    """Run one GP / EVAS configuration. `cfg` is a resolved dict (see config.resolve)."""
    rng = np.random.default_rng(seed)
    Xtr, ytr = data["Xtr"], data["ytr"]
    E = make_engine(cfg, Xtr.shape[1])
    fit_fn = FITNESS[cfg["fitness"]]
    t0 = time.perf_counter()

    sampler = None
    if cfg["sampling"] != "none":
        ext = _rf_oob_scores(Xtr, ytr, seed) if cfg["committee"] == "rf" else None
        sampler = ActiveSampler(Xtr[ytr == 1], Xtr[ytr == 0], cfg, rng, external_scores=ext)

    pop = E.init_pop(rng, cfg["pop_size"])
    S_X, S_y = Xtr, ytr
    # prevalence-corrected fitness: judge on the REAL part of the sample only, with each kept
    # negative re-weighted to stand for the whole majority class (synthetic cases still train
    # the model through E.fit, but no longer distort the measured AP)
    n_neg_total = int((ytr == 0).sum())
    use_pw = bool(cfg.get("prevalence_fitness", True)) and cfg["fitness"] == "ap" and cfg["sampling"] != "none"
    n_fit, S_w = len(S_y), None
    fits = params = None
    evaluated = None      # the last population whose fitness/params are known (for the committee)
    cache = {}
    archive = {}          # key -> (ind, params, gen, fit)

    log, events = [], []
    last_event, epoch_best = 0, []
    case_evals = 0

    for g in range(cfg["generations"]):
        # ---------------------------------------------------------------- sampling event?
        fire = False
        if sampler is not None:
            if g == 0:
                fire = True
            elif cfg["sampling"] == "dynamic":
                since = g - last_event
                if cfg["schedule"] == "fixed":
                    fire = since >= cfg["tau"]
                else:  # adaptive: stagnation on the current sample, bounded epoch length
                    p = cfg["patience"]
                    stagn = len(epoch_best) > p and epoch_best[-1] - epoch_best[-1 - p] < cfg["delta"]
                    fire = since >= cfg["tau_max"] or (since >= cfg["tau_min"] and stagn)
        if fire:
            committee = []
            if g > 0 and cfg["committee"] == "population":
                for i in _distinct_best(E, evaluated, fits, cfg["committee_size"]):
                    f = _Scored(E, evaluated[i], params[i])
                    committee.append((f, youden_threshold(f(S_X), S_y)))
                case_evals += len(committee) * len(S_y)
            elif g > 0:          # RF committee: signals are pre-computed; any non-empty marker
                committee = [None]
            S_X, S_y, info = sampler.build(committee)
            if use_pw:
                n_fit = info["n_maj"] + info["n_pos"]
                S_w = np.r_[np.full(info["n_maj"], n_neg_total / max(info["n_maj"], 1)),
                            np.ones(info["n_pos"])]
            else:
                n_fit = len(S_y)
            info["gen"] = g
            events.append(info)
            cache, last_event, epoch_best = {}, g, []

        # ---------------------------------------------------------------- evaluation on S_e
        fits = np.empty(len(pop)); params = [None] * len(pop)
        for i, ind in enumerate(pop):
            kk = E.key(ind)
            hit = cache.get(kk)
            if hit is None:
                pr = E.fit(ind, S_X, S_y)
                sc = E.score(ind, pr, S_X)
                hit = (fit_ap(sc[:n_fit], S_y[:n_fit], S_w) if S_w is not None else fit_fn(sc, S_y), pr)
                cache[kk] = hit
                case_evals += len(S_y)
            fits[i], params[i] = hit
        ib = int(np.argmax(fits))
        epoch_best.append(fits[ib])
        sizes = [E.size(t) for t in pop]
        log.append({"gen": g, "best": float(fits[ib]), "mean": float(fits.mean()),
                    "median": float(np.median(fits)), "size_mean": float(np.mean(sizes)),
                    "size_best": int(sizes[ib]), "n_unique": len({E.key(t) for t in pop}),
                    "sample": int(len(S_y))})
        if verbose and (g % 10 == 0 or g == cfg["generations"] - 1):
            print(f"  gen {g:3d}  best={fits[ib]:.4f}  mean={fits.mean():.4f}  "
                  f"size={np.mean(sizes):.1f}  |S|={len(S_y)}")

        # ---------------------------------------------------------------- checkpoint archive
        if (g + 1) % cfg["checkpoint_every"] == 0 or g == cfg["generations"] - 1:
            top = _distinct_best(E, pop, fits, cfg["archive_k"])
            for i in top:
                archive.setdefault(E.key(pop[i]), (pop[i], params[i], g, float(fits[i])))

        evaluated = pop
        if g < cfg["generations"] - 1:
            pop = E.next_generation(rng, pop, fits)

    train_time = time.perf_counter() - t0
    if sampler is not None:
        case_evals += sampler.signal_evals

    # -------------------------------------------------------------------- single model (validation)
    t1 = time.perf_counter()
    Xva, yva, Xte, yte = data["Xva"], data["yva"], data["Xte"], data["yte"]
    best = None
    for ind, pr, gen, fit in archive.values():
        ap = average_precision_score(yva, E.score(ind, pr, Xva))
        cand = (ap, -E.size(ind))
        if best is None or cand > best[0]:
            best = (cand, ind, pr, gen, fit)
    _, ind, pr0, sel_gen, sel_fit = best
    arch = list(archive.values())
    scored = sorted(((average_precision_score(yva, E.score(a_ind, a_pr, Xva)), -E.size(a_ind), i)
                     for i, (a_ind, a_pr, _, _) in enumerate(arch)), reverse=True)
    top = [i for _, _, i in scored[:cfg["ens_k"]]]
    refit_cache = {}

    def _refit(i_ind, i_pr, key):
        if key not in refit_cache:
            refit_cache[key] = E.fit(i_ind, Xtr, ytr)   # logistic layer refitted on the full training set
        return refit_cache[key]

    def _finish(refit):
        """Single model + ensemble of views, with or without the final refit (evolution unchanged)."""
        pr = _refit(ind, pr0, E.key(ind)) if refit else pr0
        s_va = E.score(ind, pr, Xva)
        thr, _ = f1_threshold(s_va, yva)
        res = {"test": full_report(E.score(ind, pr, Xte), yte, thr), "val": full_report(s_va, yva, thr),
               "train": full_report(E.score(ind, pr, Xtr), ytr, thr), "threshold": thr,
               "model": E.to_str(ind, pr, data.get("feature_names"))}
        # members: the ens_k archived models with the best validation AUC-PR (different checkpoints,
        # hence samples built with different generators = different 'views')
        va_p, te_p, tr_p = [], [], []
        for i in top:
            m_ind = arch[i][0]
            m_pr = _refit(m_ind, arch[i][1], E.key(m_ind)) if refit else arch[i][1]
            sv = E.score(m_ind, m_pr, Xva)
            F = _ecdf(sv)
            va_p.append(F(sv)); te_p.append(F(E.score(m_ind, m_pr, Xte))); tr_p.append(F(E.score(m_ind, m_pr, Xtr)))
        ens_va, ens_te, ens_tr = (np.mean(p, axis=0) for p in (va_p, te_p, tr_p))
        ethr, _ = f1_threshold(ens_va, yva)
        res.update({"test_ens": full_report(ens_te, yte, ethr), "val_ens": full_report(ens_va, yva, ethr),
                    "train_ens": full_report(ens_tr, ytr, ethr)})
        return res

    main_refit = bool(cfg.get("final_refit", True))
    out = _finish(main_refit)
    alt = _finish(not main_refit)
    out["variant"] = "refit" if main_refit else "norefit"
    out["alt_" + ("norefit" if main_refit else "refit")] = {k: alt[k] for k in ("test", "val", "test_ens", "val_ens", "threshold")}
    out.update({
        "selected_gen": int(sel_gen), "selected_sample_fitness": sel_fit,
        "model_size": E.size(ind), "model_depth": E.depth(ind), "model_features": E.features(ind),
        "archive_size": len(archive), "n_events": len(events), "engine": E.name,
        "ens_members": len(top), "ens_size": int(sum(E.size(arch[i][0]) for i in top)),
        "train_time_s": train_time, "select_time_s": time.perf_counter() - t1,
        "case_evals": int(case_evals), "log": log, "events": events,
    })
    if E.name == "mggp":
        out["model_genes"] = len(ind)
    return out
