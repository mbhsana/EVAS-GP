"""Default parameters and the method presets compared in the paper.

Every run stores the full resolved configuration in its JSON output.
"""
from __future__ import annotations

DEFAULTS = dict(
    # ---------------------------------------------------------------- GP
    pop_size=200, generations=80, max_depth=10, init_min_depth=2, init_max_depth=6,
    p_cx=0.8, p_mut=0.2, tournament=5, elites=2,
    fitness="auc",                 # 'ap' | 'auc' (threshold-free) | 'gmean' | 'mcc' | 'f1' | 'bacc'
    prevalence_fitness=True,       # 'ap' fitness on the REAL part of the sample, negatives re-weighted
                                   # to the true prevalence; synthetic cases train the model, real
                                   # cases (at true prevalence) judge it. No-op without sampling.
    final_refit=True,              # refit the logistic layer of the returned models on the full
                                   # training set after evolution (no-op for GP-Full and trees)
    engine="tree",                 # 'tree' (one tree) | 'mggp' (multi-gene GP + logistic layer)
    max_genes=6,                   # mggp: maximum number of trees per individual
    gene_max_depth=6,              # mggp: maximum depth of each gene
    ridge=1.0,                     # mggp: L2 penalty of the logistic layer
    base_features=False,           # mggp: also give the raw features to the logistic layer
    # ---------------------------------------------------------------- archive / model selection
    checkpoint_every=5,            # archive the archive_k best distinct trees every 5 generations
    archive_k=5,                   # (same checkpoints for ALL methods -> same number of candidates)
    ens_k=5,                       # ensemble: the ens_k archived models with the best validation AUC-PR
    # ---------------------------------------------------------------- sampling
    sampling="dynamic",            # 'none' (whole D_tr) | 'static' (one sample) | 'dynamic'
    n_maj=10_000,                  # |U_e| ...
    maj_cap_frac=0.5,              # ... capped at this fraction of |D_tr^-| (small data sets)
    minority_ratio=0.5,            # r: (|D_tr^+| + |Z_e|) / |U_e|
    maj_mode="random",             # 'random' | 'active'
    maj_lambda=0.5,                # weight of FP votes vs disagreement
    maj_random_frac=0.3,           # beta: share of U_e drawn uniformly
    age_weight=0.2,                # gamma: DSS-like age term
    pool_size=None,                # candidates scored per event (None = all of D_tr^-)
    pivot_mode="all",              # 'all' | 'uncertainty' | 'uncertainty_diversity' | 'hardness'
    rho_unc=0.5, rho_div=0.6, n_clusters=5,
    generators=("smote",), gen_schedule="fixed",   # 'fixed' | 'cycle' | 'random'
    k_neighbors=5, m_neighbors=10,
    committee="population",        # 'population' | 'rf'
    committee_size=15,
    schedule="fixed",              # 'fixed' (every tau generations) | 'adaptive'
    tau=5, tau_min=3, tau_max=15, patience=3, delta=1e-3,
)

VIEWS = ("smote", "borderline", "adasyn", "safelevel")

# Simplified method set (14/11/2026): synthetic cases are seeded from ALL real minority cases
# (Borderline-SMOTE, ADASYN and Safe-Level-SMOTE apply their own selection strategies); the
# committee only drives the active under-sampling of the majority class (DSS lineage).
METHODS = {
    # ------------------------------------------------ references (same learner and budget)
    "GP-Full":       dict(sampling="none"),
    "GP-Static":     dict(sampling="static"),
    "GP-RSS":        dict(),                                      # dynamic random U + SMOTE
    # ------------------------------------------------ EVAS
    "EVAS":          dict(maj_mode="active"),                     # active majority, SMOTE
    "EVAS-Views":    dict(maj_mode="active", generators=VIEWS, gen_schedule="cycle"),  # full method
    # ------------------------------------------------ controls
    "RSS-Views":     dict(generators=VIEWS, gen_schedule="cycle"),  # views without active selection
    "EVAS-Views-RF": dict(maj_mode="active", generators=VIEWS, gen_schedule="cycle", committee="rf"),
}

CORE_METHODS = ["GP-Full", "GP-Static", "GP-RSS", "EVAS", "EVAS-Views", "RSS-Views", "EVAS-Views-RF"]


def resolve(method, **overrides):
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose among {sorted(METHODS)}")
    cfg = dict(DEFAULTS)
    cfg.update(METHODS[method])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["generators"] = tuple(cfg["generators"])
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown parameters {unknown}")
    return cfg
