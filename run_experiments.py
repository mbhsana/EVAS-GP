"""Run GP/EVAS methods and non-evolutionary baselines over several seeds.

One JSON file per (data set, method, seed) in results/<dataset>/ ; runs already present are
skipped, so an interrupted campaign can simply be relaunched.  Seed s fixes BOTH the
train/val/test split and the GP random stream, so that run s of every method uses the
same split (paired statistical tests).

Examples
--------
python run_experiments.py --dataset synthetic --methods GP-RSS EVAS-UD-Views --seeds 0-1 --gens 20 --pop 60
python run_experiments.py --dataset creditcard --csv data/creditcard.csv --methods core --seeds 0-29 --jobs 8
python run_experiments.py --dataset creditcard --csv data/creditcard.csv --baselines --seeds 0-9 --jobs 8
python run_experiments.py --dataset imb:abalone_19 --methods core --seeds 0-29 --jobs 8
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker: no oversubscription, reproducible

import argparse            # noqa: E402
import json                # noqa: E402
import platform            # noqa: E402
import sys                 # noqa: E402
import time                # noqa: E402
import traceback           # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np         # noqa: E402
import scipy               # noqa: E402
import sklearn             # noqa: E402

from evas import data as D                                           # noqa: E402
from evas.baselines import CLASSIFIERS, REBALANCING, run_baseline    # noqa: E402
from evas.config import CORE_METHODS, METHODS, resolve               # noqa: E402
from evas.core import run_gp                                         # noqa: E402

_CACHE = {}


def _data(args, seed):
    k = (args.dataset, args.csv)
    if k not in _CACHE:
        _CACHE[k] = D.load(args.dataset, args.csv, args.target)
    X, y, names = _CACHE[k]
    time_col = None
    if args.split == "temporal":
        if not names or args.time_col not in names:
            raise ValueError(f"temporal split needs a '{args.time_col}' column")
        time_col = names.index(args.time_col)
    return D.split(X, y, seed=seed if args.split_per_seed else 0, names=names,
                   mode=args.split, time_col=time_col)


def env_info():
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform()}
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return info


def job_gp(args, method, seed):
    overrides = dict(generations=args.gens, pop_size=args.pop, fitness=args.fitness, n_maj=args.n_maj,
                     minority_ratio=args.minority_ratio, tau=args.tau, engine=args.engine,
                     max_genes=args.max_genes, base_features=args.base_features, ens_k=args.ens_k,
                     prevalence_fitness=None if args.prevalence_fitness is None else bool(args.prevalence_fitness),
                     final_refit=None if args.final_refit is None else bool(args.final_refit))
    cfg = resolve(method, **overrides)
    data = _data(args, seed)
    res = run_gp(data, cfg, seed=seed, verbose=args.verbose)
    res.update(kind="gp", dataset=args.dataset, method=method, seed=seed, config=cfg, env=env_info())
    return res


def job_base(args, clf, how, seed):
    data = _data(args, seed)
    res = run_baseline(data, clf, how, seed=seed, n_estimators=args.n_estimators)
    res.update(kind="baseline", dataset=args.dataset, method=f"{clf}+{how}", classifier=clf,
               rebalancing=how, seed=seed, env=env_info())
    return res


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic",
                    help="creditcard | synthetic | synthetic_cc | imb:<name> | csv:<path>")
    ap.add_argument("--csv", help="path to creditcard.csv")
    ap.add_argument("--target", default="Class")
    ap.add_argument("--methods", nargs="+", default=["core"], help="'core', 'all' or method names")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--classifiers", nargs="+", default=CLASSIFIERS)
    ap.add_argument("--rebalancing", nargs="+", default=REBALANCING)
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--fitness", default=None)
    ap.add_argument("--n_maj", type=int, default=None)
    ap.add_argument("--minority_ratio", type=float, default=None)
    ap.add_argument("--tau", type=int, default=None)
    ap.add_argument("--engine", default=None, choices=["tree", "mggp"])
    ap.add_argument("--max_genes", type=int, default=None)
    ap.add_argument("--ens_k", type=int, default=None, help="size of the ensemble of views")
    ap.add_argument("--prevalence_fitness", type=int, default=None, help="1: AP fitness on real points at true prevalence")
    ap.add_argument("--final_refit", type=int, default=None, help="1: refit the logistic layer on the full training set")
    ap.add_argument("--base_features", type=int, default=None, help="mggp: 1 = raw features in the logistic layer")
    ap.add_argument("--split_per_seed", type=int, default=1, help="1: split drawn with the seed (default)")
    ap.add_argument("--split", default="stratified", choices=["stratified", "temporal"])
    ap.add_argument("--time_col", default="Time", help="column used by --split temporal")
    ap.add_argument("--tag", default="", help="suffix of the result folder (e.g. a sensitivity setting)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tag = "__".join(t for t in (args.tag, "temporal" if args.split == "temporal" else "") if t)
    folder = args.dataset.replace(":", "_").replace("/", "_").replace("csv_", "csv-") + (f"__{tag}" if tag else "")
    outdir = os.path.join(args.out, folder)
    os.makedirs(outdir, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    jobs = []
    if args.baselines:
        for clf in args.classifiers:
            for how in args.rebalancing:
                for s in seeds:
                    jobs.append(("base", (clf, how, s), f"{clf}+{how}_s{s}.json"))
    else:
        methods = CORE_METHODS if args.methods == ["core"] else list(METHODS) if args.methods == ["all"] else args.methods
        for m in methods:
            resolve(m)  # fail fast on typos
            for s in seeds:
                jobs.append(("gp", (m, s), f"{m}_s{s}.json"))
    todo = [j for j in jobs if not os.path.exists(os.path.join(outdir, j[2]))]
    print(f"{len(jobs)} runs planned, {len(todo)} to do -> {outdir}", flush=True)

    def submit(ex, kind, a):
        return ex.submit(job_gp if kind == "gp" else job_base, args, *a)

    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {submit(ex, k, a): fn for k, a, fn in todo}
        for f in as_completed(futs):
            fn = futs[f]
            try:
                res = f.result()
            except Exception:  # keep the campaign going, report at the end
                failed += 1
                print(f"FAILED {fn}\n{traceback.format_exc()}", flush=True)
                continue
            tmp = os.path.join(outdir, fn + ".tmp")
            with open(tmp, "w") as fh:
                json.dump(res, fh)
            os.replace(tmp, os.path.join(outdir, fn))   # atomic: no half-written result files
            done += 1
            t = res["test"]
            print(f"[{done}/{len(todo)} {time.time() - t0:7.0f}s] {fn:36s} AUC-PR={t['auc_pr']:.3f} "
                  f"F1={t['f1']:.3f} MCC={t['mcc']:.3f} train={res['train_time_s']:.0f}s", flush=True)
    print(f"finished: {done} ok, {failed} failed")


if __name__ == "__main__":
    main()
