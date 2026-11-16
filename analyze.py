"""Aggregate the runs and produce the paper's tables, figures and statistics.

python analyze.py --results results --ref EVAS-UD-Views --metric auc_pr

Outputs (in <results>/analysis/):
  runs.csv, summary.csv, stats.txt
  tables/main_<metric>.tex      methods x data sets, mean ± sd, Wilcoxon-Holm marks vs ref, avg. rank
  tables/detail_<ds>.tex         all metrics for one data set
  tables/baselines_<ds>.tex      classifier x rebalancing (F1 and AUC-PR)
  tables/datasets.tex            data set characteristics (if --describe is given)
  figures/box_<ds>.pdf, figures/gap_<ds>.pdf, figures/dynamics_<ds>.pdf, figures/cd_<metric>.pdf

Statistics: paired two-sided Wilcoxon signed-rank tests on the runs that share a seed
(= the same split), Holm correction within each data set; Vargha-Delaney A12 effect size;
Friedman test and Nemenyi critical difference over data sets (Demsar, 2006).
Baselines shown in the main table: for each classifier, the rebalancing method with the best
mean VALIDATION AUC-PR (never chosen on test).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from evas.config import CORE_METHODS, METHODS

METRICS = ["auc_pr", "auc_roc", "f1", "mcc", "gmean", "precision", "recall", "recall_at_fpr1e-3"]
LABEL = {"auc_pr": "AUC-PR", "auc_roc": "ROC-AUC", "f1": "F1", "mcc": "MCC", "gmean": "G-mean",
         "precision": "Prec.", "recall": "Rec.", "recall_at_fpr1e-3": "R@FPR$_{10^{-3}}$"}
# Nemenyi critical values q_0.05 (two-tailed), Demsar (2006), Table 5(a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489,
       19: 3.517, 20: 3.544}


# ============================================================================ loading
def load_runs(results):
    rows, logs = [], []
    for f in sorted(glob.glob(os.path.join(results, "*", "*.json"))):
        r = json.load(open(f))
        ds = os.path.basename(os.path.dirname(f))
        row = {"dataset": ds, "method": r["method"], "kind": r.get("kind", "gp"), "seed": r["seed"],
               "train_time_s": r["train_time_s"], "model_size": r.get("model_size", np.nan),
               "case_evals": r.get("case_evals", np.nan), "classifier": r.get("classifier"),
               "rebalancing": r.get("rebalancing")}
        for part in ("test", "val", "train"):
            for m in METRICS:
                row[f"{part}_{m}" if part != "test" else m] = r[part][m]
        row["gap_auc_pr"] = row["train_auc_pr"] - row["auc_pr"]
        rows.append(row)
        if r.get("kind", "gp") == "gp":
            for e in r.get("events", []):
                logs.append({"dataset": ds, "method": r["method"], "seed": r["seed"], **{
                    k: e.get(k) for k in ("event", "gen", "generator", "n_syn", "n_pivots", "n_seeds",
                                          "sel_fp_rate", "pool_fp_rate", "gen_fallback")}})
    return pd.DataFrame(rows), pd.DataFrame(logs)


def order_methods(methods):
    gp_order = CORE_METHODS + [m for m in METHODS if m not in CORE_METHODS]
    known = [m for m in gp_order if m in methods]
    return known + sorted(m for m in methods if m not in known)


# ============================================================================ statistics
def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def a12(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def paired_tests(d, ref, metric, min_pairs=5):
    """Return {method: (mean_diff ref-method, p_holm, A12(ref, method), n)}."""
    refv = d[d.method == ref].set_index("seed")[metric]
    out, keys, ps = {}, [], []
    if refv.empty:
        return out
    for m, dm in d.groupby("method"):
        if m == ref:
            continue
        v = dm.set_index("seed")[metric]
        common = refv.index.intersection(v.index)
        if len(common) < min_pairs:
            continue
        a, b = refv[common].to_numpy(), v[common].to_numpy()
        p = 1.0 if np.allclose(a, b) else wilcoxon(a, b, zero_method="pratt").pvalue
        out[m] = [float((a - b).mean()), p, float(a12(a, b)), len(common)]
        keys.append(m); ps.append(p)
    for m, pa in zip(keys, holm(ps)):
        out[m][1] = float(pa)
    return out


def best_baselines(df):
    """Per data set and classifier: rebalancing with the best mean validation AUC-PR."""
    b = df[df.kind == "baseline"]
    keep = []
    for (ds, clf), g in b.groupby(["dataset", "classifier"]):
        best = g.groupby("method")["val_auc_pr"].mean().idxmax()
        keep.append((ds, best))
    return keep


# ============================================================================ LaTeX helpers
def fmt(mu, sd, nd=3):
    return f"{mu:.{nd}f}\\,$\\pm$\\,{sd:.{nd}f}"


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def main_table(df, ref, metric, path):
    keep = set(best_baselines(df))
    sel = df[(df.kind == "gp") | df.apply(lambda r: (r.dataset, r.method) in keep, axis=1)]
    datasets = sorted(sel.dataset.unique())
    gp_methods = order_methods(sel[sel.kind == "gp"].method.unique())
    base_methods = sorted(sel[sel.kind == "baseline"].method.unique())
    methods = gp_methods + base_methods
    means = sel.groupby(["dataset", "method"])[metric].mean().unstack()
    sds = sel.groupby(["dataset", "method"])[metric].std().unstack()
    # average ranks over data sets (methods present on all data sets only)
    ranks = None
    full = means.dropna(axis=1)
    if len(datasets) >= 2 and full.shape[1] >= 2:
        ranks = full.rank(axis=1, ascending=False).mean()
    lines = ["\\begin{tabular}{l" + "c" * len(datasets) + ("c" if ranks is not None else "") + "}",
             "\\toprule",
             "Method & " + " & ".join(tex_escape(d) for d in datasets) + (" & Rank" if ranks is not None else "") + " \\\\",
             "\\midrule"]
    tests = {ds: paired_tests(sel[sel.dataset == ds], ref, metric) for ds in datasets}
    for i, m in enumerate(methods):
        if i == len(gp_methods) and base_methods:
            lines.append("\\midrule")
        cells = []
        for ds in datasets:
            if m not in means.columns or pd.isna(means.loc[ds, m]):
                cells.append("--"); continue
            c = fmt(means.loc[ds, m], sds.loc[ds, m])
            if means.loc[ds, m] == means.loc[ds].max():
                c = "\\textbf{" + c + "}"
            if m != ref and m in tests[ds]:
                diff, p, _, _ = tests[ds][m]
                mark = "$\\approx$" if p >= 0.05 else ("$-$" if diff > 0 else "$+$")
                c += "\\,{\\scriptsize " + mark + "}"
            cells.append(c)
        rk = (f" & {ranks[m]:.2f}" if (ranks is not None and m in ranks.index) else (" & --" if ranks is not None else ""))
        name = tex_escape(m) + (" (ref.)" if m == ref else "")
        lines.append(name + " & " + " & ".join(cells) + rk + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def detail_table(d, path, ref):
    cols = ["auc_pr", "f1", "mcc", "precision", "recall", "recall_at_fpr1e-3", "gap_auc_pr", "model_size", "train_time_s"]
    head = ["AUC-PR", "F1", "MCC", "Prec.", "Rec.", "R@FPR$_{10^{-3}}$", "Gap", "Size", "Time (s)"]
    g = d[d.kind == "gp"]
    methods = order_methods(g.method.unique())
    mu = g.groupby("method")[cols].mean(); sd = g.groupby("method")[cols].std()
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(head) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for c in cols:
            if c in ("model_size", "train_time_s"):
                cells.append(f"{mu.loc[m, c]:.0f}")
            else:
                s = f"{mu.loc[m, c]:.3f}"
                best = mu[c].min() if c == "gap_auc_pr" else mu[c].max()
                cells.append("\\textbf{" + s + "}" if np.isclose(mu.loc[m, c], best) else s)
        lines.append(tex_escape(m) + (" (ref.)" if m == ref else "") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def baseline_table(d, path):
    b = d[d.kind == "baseline"]
    if b.empty:
        return
    clfs = [c for c in ["LR", "DT", "RF", "GB"] if c in b.classifier.unique()]
    rebs = [r for r in ["none", "weight", "rus", "ros", "smote", "borderline", "adasyn", "safelevel"]
            if r in b.rebalancing.unique()]
    names = {"none": "None", "weight": "Class weighting", "rus": "Random under-sampling",
             "ros": "Random over-sampling", "smote": "SMOTE", "borderline": "Borderline-SMOTE",
             "adasyn": "ADASYN", "safelevel": "Safe-Level-SMOTE"}
    f1 = b.groupby(["rebalancing", "classifier"])["f1"].mean().unstack()
    ap = b.groupby(["rebalancing", "classifier"])["auc_pr"].mean().unstack()
    lines = ["\\begin{tabular}{l" + "c" * (2 * len(clfs)) + "}", "\\toprule",
             " & \\multicolumn{%d}{c}{F1} & \\multicolumn{%d}{c}{AUC-PR} \\\\" % (len(clfs), len(clfs)),
             "Rebalancing & " + " & ".join(clfs + clfs) + " \\\\", "\\midrule"]
    for r in rebs:
        cells = [f"{f1.loc[r, c]:.3f}" if c in f1.columns else "--" for c in clfs] + \
                [f"{ap.loc[r, c]:.3f}" if c in ap.columns else "--" for c in clfs]
        lines.append(names[r] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


# ============================================================================ figures
def figures(df, ev, outdir, ref, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    for ds, d in df[df.kind == "gp"].groupby("dataset"):
        methods = order_methods(d.method.unique())
        # box plots of the primary metric
        fig, ax = plt.subplots(figsize=(6.2, 2.4))
        data = [d[d.method == m][metric].to_numpy() for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor("#c44e52" if m == ref else "#8da0cb"); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(methods) + 1)); ax.set_xticklabels(methods, rotation=40, ha="right")
        ax.set_ylabel(f"test {LABEL[metric]}"); fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"box_{ds}.pdf")); plt.close(fig)
        # generalisation gap vs test performance
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        g = d.groupby("method")[["gap_auc_pr", "auc_pr"]].mean()
        ax.scatter(g.gap_auc_pr, g.auc_pr, c=["#c44e52" if m == ref else "#4c72b0" for m in g.index], s=18)
        for m, row in g.iterrows():
            ax.annotate(m, (row.gap_auc_pr, row.auc_pr), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("AUC-PR gap (train $-$ test)"); ax.set_ylabel("test AUC-PR"); fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"gap_{ds}.pdf")); plt.close(fig)
    # sample dynamics: false-positive rate of the selected negatives over events
    if not ev.empty and ev.sel_fp_rate.notna().any():
        for ds, e in ev[ev.sel_fp_rate.notna()].groupby("dataset"):
            fig, ax = plt.subplots(figsize=(3.2, 2.4))
            for m, em in e.groupby("method"):
                s = em.groupby("event")[["sel_fp_rate", "pool_fp_rate"]].mean()
                ax.plot(s.index, s.sel_fp_rate, label=f"{m} (selected)")
                ax.plot(s.index, s.pool_fp_rate, ls=":", color=ax.lines[-1].get_color())
            ax.set_xlabel("sampling event"); ax.set_ylabel("committee FP vote rate")
            ax.legend(fontsize=5, frameon=False); fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"dynamics_{ds}.pdf")); plt.close(fig)


def cd_diagram(avg_ranks, cd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = avg_ranks.sort_values(); k = len(r)
    fig, ax = plt.subplots(figsize=(6, 0.35 * k + 0.8)); ax.set_xlim(1, k); ax.set_ylim(-k - 1, 1)
    ax.invert_xaxis(); ax.get_yaxis().set_visible(False)
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.tick_top(); ax.set_xticks(range(1, k + 1))
    for i, (m, v) in enumerate(r.items()):
        ax.plot([v, v], [0, -i - 1], color="k", lw=0.6); ax.text(v, -i - 1, f" {m} ({v:.2f})", va="center", fontsize=6)
    ax.plot([1, 1 + cd], [0.6, 0.6], color="r", lw=2); ax.text(1 + cd / 2, 0.8, "CD", ha="center", fontsize=6, color="r")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--ref", default="EVAS-UD-Views")
    ap.add_argument("--metric", default="auc_pr")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_figures", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "analysis")
    tdir, fdir = os.path.join(out, "tables"), os.path.join(out, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)

    df, ev = load_runs(a.results)
    if df.empty:
        print("no results found"); return
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    agg_cols = METRICS + ["gap_auc_pr", "model_size", "train_time_s", "case_evals"]
    summ = df.groupby(["dataset", "method"])[agg_cols].agg(["mean", "std"])
    summ["runs"] = df.groupby(["dataset", "method"]).size()
    summ.to_csv(os.path.join(out, "summary.csv"))

    report = []
    pd.set_option("display.width", 200)
    for ds, d in df.groupby("dataset"):
        report.append(f"\n=== {ds} ===")
        view = d.groupby("method").agg(runs=("seed", "count"), auc_pr=("auc_pr", "mean"), sd=("auc_pr", "std"),
                                       f1=("f1", "mean"), mcc=("mcc", "mean"), gap=("gap_auc_pr", "mean"),
                                       size=("model_size", "mean"), time=("train_time_s", "mean"))
        report.append(view.sort_values("auc_pr", ascending=False).round(4).to_string())
        t = paired_tests(d[d.kind == "gp"], a.ref, a.metric)
        if t:
            report.append(f"\n{a.ref} vs others on {a.metric}: mean diff (ref - other), Holm p, A12(ref>other), n")
            for m in order_methods(list(t)):
                diff, p, eff, n = t[m]
                report.append(f"  {m:24s} {diff:+.4f}  p={p:.4f}  A12={eff:.2f}  n={n}")
        detail_table(d, os.path.join(tdir, f"detail_{ds}.tex"), a.ref)
        baseline_table(d, os.path.join(tdir, f"baselines_{ds}.tex"))

    main_table(df, a.ref, a.metric, os.path.join(tdir, f"main_{a.metric}.tex"))

    # Friedman / Nemenyi over data sets (GP methods present everywhere)
    gpdf = df[df.kind == "gp"]
    piv = gpdf.groupby(["dataset", "method"])[a.metric].mean().unstack().dropna(axis=1)
    if piv.shape[0] >= 2 and piv.shape[1] >= 3:
        st = friedmanchisquare(*[piv[c] for c in piv.columns])
        rk = piv.rank(axis=1, ascending=False).mean()
        k, n = piv.shape[1], piv.shape[0]
        cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
        report.append(f"\nFriedman over {n} data sets, {k} methods: chi2={st.statistic:.3f}, p={st.pvalue:.4g}; "
                      f"Nemenyi CD(0.05)={cd:.3f}")
        report.append("Average ranks: " + ", ".join(f"{m}={v:.2f}" for m, v in rk.sort_values().items()))
        if not a.no_figures:
            cd_diagram(rk, cd, os.path.join(fdir, f"cd_{a.metric}.pdf"))
    if not a.no_figures:
        figures(df, ev, fdir, a.ref, a.metric)
    txt = "\n".join(report)
    open(os.path.join(out, "stats.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\ntables -> {tdir}\nfigures -> {fdir}")


if __name__ == "__main__":
    main()
