"""Vectorised tree-based Genetic Programming engine (NumPy only, no DEAP).

Representation
--------------
A tree is a list, in prefix order, of nodes (kind, value):
    kind 0 = function   (value = index in FUNCS)
    kind 1 = variable   (value = feature index)
    kind 2 = constant   (value = float)
A tree f maps a data matrix X (n x d) to a real score vector f(X). Classification
thresholds are handled outside the engine (fitness functions / model selection).

Design choices (reported in the paper)
--------------------------------------
- protected operators (division, sqrt, log) and clipping of all intermediate values;
- ramped half-and-half initialisation;
- subtree crossover with 90% internal-node bias (Koza), subtree + point mutation;
- static depth limit: an offspring that exceeds it is replaced by its parent;
- tournament selection with lexicographic parsimony (ties on fitness -> smaller tree);
- elitism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIP = 1e6


def _c(a):
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)


def _add(a, b): return _c(a + b)
def _sub(a, b): return _c(a - b)
def _mul(a, b): return _c(a * b)


def _pdiv(a, b):
    a = np.broadcast_to(a, np.broadcast(a, b).shape).astype(float)
    b = np.broadcast_to(b, a.shape)
    out = np.ones_like(a)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-6)
    return _c(out)


def _neg(a): return -a
def _psqrt(a): return np.sqrt(np.abs(a))
def _plog(a): return np.log1p(np.abs(a))
def _sin(a): return np.sin(a)
def _cos(a): return np.cos(a)
def _ifgt(a, b, c): return np.where(a > 0, b, c)


FUNCS = [
    ("add", 2, _add), ("sub", 2, _sub), ("mul", 2, _mul), ("div", 2, _pdiv),
    ("neg", 1, _neg), ("sqrt", 1, _psqrt), ("log", 1, _plog),
    ("sin", 1, _sin), ("cos", 1, _cos), ("ifgt", 3, _ifgt),
]


# ============================================================================ evaluation
def evaluate(tree, X):
    """Score vector f(X), always finite, shape (n,)."""
    n = X.shape[0]

    def rec(i):
        kind, v = tree[i]
        if kind == 0:
            _, ar, fn = FUNCS[v]
            args, j = [], i + 1
            for _ in range(ar):
                a, j = rec(j)
                args.append(a)
            return fn(*args), j
        if kind == 1:
            return X[:, v], i + 1
        return np.float64(v), i + 1

    with np.errstate(all="ignore"):
        out, _ = rec(0)
    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.full(n, float(out))
    return _c(out)


# ============================================================================ structure helpers
def subtree_end(tree, i):
    """Index one past the end of the subtree rooted at i."""
    need, j = 1, i
    while need > 0:
        kind, v = tree[j]
        need += (FUNCS[v][1] if kind == 0 else 0) - 1
        j += 1
    return j


def depth(tree):
    """Depth of the tree (a single terminal has depth 0)."""
    stack = []
    for kind, v in reversed(tree):
        if kind == 0:
            stack.append(1 + max(stack.pop() for _ in range(FUNCS[v][1])))
        else:
            stack.append(0)
    return stack[-1]


def is_valid(tree):
    try:
        return len(tree) > 0 and subtree_end(tree, 0) == len(tree)
    except IndexError:
        return False


def key(tree):
    """Hashable identity of a tree (used for fitness caching and de-duplication)."""
    return tuple(tree)


def to_str(tree, names=None):
    def rec(i):
        kind, v = tree[i]
        if kind == 0:
            name, ar, _ = FUNCS[v]
            j, args = i + 1, []
            for _ in range(ar):
                s, j = rec(j)
                args.append(s)
            return f"{name}({', '.join(args)})", j
        if kind == 1:
            return (names[v] if names is not None else f"x{v}"), i + 1
        return f"{v:.4g}", i + 1
    return rec(0)[0]


def used_features(tree):
    return sorted({v for k, v in tree if k == 1})


# ============================================================================ configuration
@dataclass
class GPConfig:
    n_features: int
    pop_size: int = 200
    max_depth: int = 10
    init_min_depth: int = 2
    init_max_depth: int = 6
    p_cx: float = 0.8
    p_mut: float = 0.2
    tournament: int = 5
    elites: int = 2
    p_const: float = 0.2      # probability that a terminal is a constant
    p_leaf: float = 0.3       # 'grow': probability of stopping above the maximum depth
    mut_max_depth: int = 3
    p_internal: float = 0.9   # crossover point bias towards internal nodes
    p_point_mut: float = 0.3  # share of mutations that are point mutations


# ============================================================================ generation
def _terminal(rng, cfg):
    if rng.random() < cfg.p_const:
        return (2, float(rng.uniform(-1.0, 1.0)))
    return (1, int(rng.integers(cfg.n_features)))


def gen_tree(rng, cfg, max_d, method="grow"):
    def rec(d):
        if d == 0 or (method == "grow" and d < max_d and rng.random() < cfg.p_leaf):
            return [_terminal(rng, cfg)]
        f = int(rng.integers(len(FUNCS)))
        out = [(0, f)]
        for _ in range(FUNCS[f][1]):
            out += rec(d - 1)
        return out
    return rec(max_d)


def ramped_half_and_half(rng, cfg, n):
    lo, hi = cfg.init_min_depth, cfg.init_max_depth
    return [gen_tree(rng, cfg, lo + i % (hi - lo + 1), "full" if i % 2 else "grow") for i in range(n)]


# ============================================================================ variation
def _pick_point(rng, tree, cfg):
    internals = [i for i, (k, _) in enumerate(tree) if k == 0]
    if internals and rng.random() < cfg.p_internal:
        return internals[int(rng.integers(len(internals)))]
    leaves = [i for i, (k, _) in enumerate(tree) if k != 0]
    return leaves[int(rng.integers(len(leaves)))]


def crossover(rng, t1, t2, cfg):
    i = _pick_point(rng, t1, cfg); ie = subtree_end(t1, i)
    j = _pick_point(rng, t2, cfg); je = subtree_end(t2, j)
    c1 = t1[:i] + t2[j:je] + t1[ie:]
    c2 = t2[:j] + t1[i:ie] + t2[je:]
    return (c1 if depth(c1) <= cfg.max_depth else list(t1),
            c2 if depth(c2) <= cfg.max_depth else list(t2))


def mutate(rng, t, cfg):
    if rng.random() < cfg.p_point_mut:
        i = int(rng.integers(len(t)))
        kind, v = t[i]
        t = list(t)
        if kind == 2:
            t[i] = (2, float(v + rng.normal(0.0, 0.3)))
        elif kind == 1:
            t[i] = (1, int(rng.integers(cfg.n_features)))
        else:  # function -> another function of the same arity
            same = [j for j, f in enumerate(FUNCS) if f[1] == FUNCS[v][1]]
            t[i] = (0, int(same[int(rng.integers(len(same)))]))
        return t
    i = int(rng.integers(len(t)))
    ie = subtree_end(t, i)
    new = gen_tree(rng, cfg, int(rng.integers(0, cfg.mut_max_depth + 1)), "grow")
    c = t[:i] + new + t[ie:]
    return c if depth(c) <= cfg.max_depth else list(t)


def tournament_select(rng, fits, sizes, k):
    idx = rng.integers(len(fits), size=k)
    return int(max(idx, key=lambda i: (fits[i], -sizes[i])))


def next_generation(rng, pop, fits, cfg):
    sizes = np.array([len(t) for t in pop])
    order = np.lexsort((sizes, -np.asarray(fits)))
    new = [list(pop[i]) for i in order[:cfg.elites]]
    while len(new) < cfg.pop_size:
        a = pop[tournament_select(rng, fits, sizes, cfg.tournament)]
        b = pop[tournament_select(rng, fits, sizes, cfg.tournament)]
        if rng.random() < cfg.p_cx:
            a, b = crossover(rng, a, b, cfg)
        else:
            a, b = list(a), list(b)
        if rng.random() < cfg.p_mut:
            a = mutate(rng, a, cfg)
        if rng.random() < cfg.p_mut:
            b = mutate(rng, b, cfg)
        new.append(a)
        if len(new) < cfg.pop_size:
            new.append(b)
    return new
