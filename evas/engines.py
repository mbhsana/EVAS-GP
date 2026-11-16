"""GP engines used by the EVAS loop.

TreeEngine       one tree per individual; the score is the tree output.
MultiGeneEngine  multi-gene GP (MG-GP): an individual is a list of 1..max_genes trees ("genes");
                 their outputs are standardised and combined by an L2-regularised logistic
                 regression fitted (IRLS) on the CURRENT training sample. GP builds non-linear
                 features, the logistic layer finds their weights (as in multi-gene symbolic
                 regression, but with a logistic output). Optionally the raw input features are
                 added to the logistic layer (base_features=True): the model is then a logistic
                 regression augmented with evolved non-linear features.

Both engines expose the same interface:
    init_pop(rng, n) -> list of individuals
    fit(ind, X, y)   -> params   (combiner fitted on the sample; None for trees)
    score(ind, params, X) -> real scores
    next_generation(rng, pop, fits) -> new population
    key(ind), size(ind), to_str(ind, params, names), features(ind), depth(ind)
"""
from __future__ import annotations

import numpy as np

from . import gp

ZCLIP = 10.0


# ============================================================================ tree engine
class TreeEngine:
    name = "tree"

    def __init__(self, gcfg):
        self.g = gcfg

    def init_pop(self, rng, n):
        return gp.ramped_half_and_half(rng, self.g, n)

    def fit(self, ind, X, y):
        return None

    def score(self, ind, params, X):
        return gp.evaluate(ind, X)

    def next_generation(self, rng, pop, fits):
        return gp.next_generation(rng, pop, fits, self.g)

    key = staticmethod(gp.key)

    @staticmethod
    def size(ind):
        return len(ind)

    @staticmethod
    def depth(ind):
        return gp.depth(ind)

    @staticmethod
    def features(ind):
        return gp.used_features(ind)

    @staticmethod
    def to_str(ind, params, names=None):
        return gp.to_str(ind, names)


# ============================================================================ logistic layer
def logistic_irls(Z, y, ridge=1.0, iters=10):
    """L2-regularised logistic regression (intercept not penalised) by Newton / IRLS."""
    n, k = Z.shape
    A = np.hstack([Z, np.ones((n, 1))])
    w = np.zeros(k + 1)
    R = ridge * np.eye(k + 1); R[-1, -1] = 0.0
    yf = y.astype(float)
    for _ in range(iters):
        eta = np.clip(A @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-9)
        grad = A.T @ (p - yf) + R @ w
        H = (A * W[:, None]).T @ A + R + 1e-9 * np.eye(k + 1)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w[:-1], w[-1]


# ============================================================================ multi-gene engine
class MultiGeneEngine:
    name = "mggp"

    def __init__(self, gcfg, max_genes=6, p_high=0.3, p_add=0.1, p_del=0.1, ridge=1.0,
                 base_features=False, gene_max_depth=6):
        self.g = gcfg
        self.max_genes = max_genes
        self.p_high, self.p_add, self.p_del = p_high, p_add, p_del
        self.ridge = ridge
        self.base = base_features
        # genes are kept shallower than single trees: several small readable features
        self.gg = gp.GPConfig(**{**gcfg.__dict__, "max_depth": gene_max_depth,
                                 "init_max_depth": min(gcfg.init_max_depth, gene_max_depth)})

    # ------------------------------------------------------------------ init
    def init_pop(self, rng, n):
        trees = gp.ramped_half_and_half(rng, self.gg, n * self.max_genes)
        pop = []
        for i in range(n):
            k = int(rng.integers(1, self.max_genes + 1))
            pop.append([trees[i * self.max_genes + j] for j in range(k)])
        return pop

    # ------------------------------------------------------------------ combiner
    def _features(self, ind, X):
        F = np.column_stack([gp.evaluate(t, X) for t in ind])
        if self.base:
            F = np.hstack([X, F])
        return F

    def fit(self, ind, X, y):
        F = self._features(ind, X)
        mu = F.mean(0); sd = F.std(0)
        keep = sd > 1e-9
        Z = np.clip((F[:, keep] - mu[keep]) / sd[keep], -ZCLIP, ZCLIP)
        if Z.shape[1] == 0:
            return (keep, mu, sd, np.zeros(0), 0.0)
        w, b = logistic_irls(Z, y, ridge=self.ridge)
        return (keep, mu, sd, w, b)

    def score(self, ind, params, X):
        keep, mu, sd, w, b = params
        if len(w) == 0:
            return np.zeros(len(X))
        F = self._features(ind, X)
        Z = np.clip((F[:, keep] - mu[keep]) / sd[keep], -ZCLIP, ZCLIP)
        return Z @ w + b

    # ------------------------------------------------------------------ variation
    def _mutate(self, rng, ind):
        ind = [list(t) for t in ind]
        r = rng.random()
        if r < self.p_add and len(ind) < self.max_genes:
            ind.append(gp.gen_tree(rng, self.gg, int(rng.integers(1, self.gg.init_max_depth + 1)), "grow"))
        elif r < self.p_add + self.p_del and len(ind) > 1:
            ind.pop(int(rng.integers(len(ind))))
        else:
            i = int(rng.integers(len(ind)))
            ind[i] = gp.mutate(rng, ind[i], self.gg)
        return ind

    def _crossover(self, rng, a, b):
        a = [list(t) for t in a]; b = [list(t) for t in b]
        if rng.random() < self.p_high:
            # high-level crossover: exchange whole genes (two-point on the gene lists)
            i1, i2 = sorted(rng.integers(0, len(a) + 1, size=2))
            j1, j2 = sorted(rng.integers(0, len(b) + 1, size=2))
            c1 = a[:i1] + b[j1:j2] + a[i2:]
            c2 = b[:j1] + a[i1:i2] + b[j2:]
            c1 = (c1 or [a[0]])[:self.max_genes]
            c2 = (c2 or [b[0]])[:self.max_genes]
            return c1, c2
        # low-level crossover: subtree crossover between one gene of each parent
        i = int(rng.integers(len(a))); j = int(rng.integers(len(b)))
        a[i], b[j] = gp.crossover(rng, a[i], b[j], self.gg)
        return a, b

    def next_generation(self, rng, pop, fits):
        cfg = self.g
        sizes = np.array([self.size(t) for t in pop])
        order = np.lexsort((sizes, -np.asarray(fits)))
        new = [[list(t) for t in pop[i]] for i in order[:cfg.elites]]
        while len(new) < cfg.pop_size:
            a = pop[gp.tournament_select(rng, fits, sizes, cfg.tournament)]
            b = pop[gp.tournament_select(rng, fits, sizes, cfg.tournament)]
            if rng.random() < cfg.p_cx:
                a, b = self._crossover(rng, a, b)
            if rng.random() < cfg.p_mut:
                a = self._mutate(rng, a)
            if rng.random() < cfg.p_mut:
                b = self._mutate(rng, b)
            new.append([list(t) for t in a])
            if len(new) < cfg.pop_size:
                new.append([list(t) for t in b])
        return new

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def key(ind):
        return tuple(tuple(t) for t in ind)

    @staticmethod
    def size(ind):
        return sum(len(t) for t in ind)

    @staticmethod
    def depth(ind):
        return max(gp.depth(t) for t in ind)

    def features(self, ind):
        f = sorted({v for t in ind for v in gp.used_features(t)})
        return f

    def to_str(self, ind, params, names=None):
        keep, mu, sd, w, b = params
        terms = []
        cols = ([(names[i] if names else f"x{i}") for i in range(len(mu) - len(ind))] if self.base else []) + \
               [gp.to_str(t, names) for t in ind]
        kept = [c for c, k in zip(cols, keep) if k]
        for c, wi, m, s in zip(kept, w, mu[keep], sd[keep]):
            terms.append(f"{wi:+.3g}*(({c})-{m:.3g})/{s:.3g}")
        return f"logit = {b:+.3g} " + " ".join(terms)
