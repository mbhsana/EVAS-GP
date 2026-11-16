# EVAS — EVolutionary Active Sampling for GP on highly imbalanced data

Code and raw results of the paper *Population-Guided Active Sampling for Genetic Programming on Highly
Imbalanced Data* (anonymous submission).

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # unit tests, < 1 min
```
Tested with Python 3.14, NumPy 2.5, SciPy 1.18, scikit-learn 1.9. The gradient-boosting baseline is
scikit-learn's HistGradientBoosting. All SMOTE variants are implemented in `evas/generators.py`;
imbalanced-learn is only used to download the benchmark data sets.

## Data
* Credit Card Fraud: `python get_data.py` (OpenML id 1597) or put Kaggle's `creditcard.csv` in `data/`.
* Benchmark (mammography, ozone_level, protein_homo): downloaded automatically with `--dataset imb:<name>`.

## Main files
| file | content |
|---|---|
| `evas/core.py` | GP loop with active sampling, prevalence-weighted AUC-PR fitness, archive and ensemble |
| `evas/sampler.py` | construction of the sample S = U + D+ + Z (committee votes, weights, generators) |
| `evas/engines.py` | multi-gene GP with logistic layer (MG-GP) and single-tree GP |
| `evas/generators.py` | SMOTE, Borderline-SMOTE, ADASYN, Safe-Level-SMOTE |
| `evas/baselines.py` | LR / RF / GB with 8 rebalancing options, same protocol |
| `evas/config.py` | default parameters and the compared methods (GP-Full, GP-Static, GP-RSS, EVAS, ...) |
| `run_experiments.py` | runs (method x seed), one JSON file per run, resumable |
| `run_paper.sh` | all the experiments of the paper |
| `paper/make_tables.py`, `paper/fig_results.py` | tables, statistics (Wilcoxon-Holm, A12, Friedman) and figure |
| `results.zip` | raw results of the paper (one JSON per run; per-generation logs removed): `unzip results.zip` |

## Tables of the paper from the raw results
```bash
unzip results.zip
python paper/make_tables.py --res results --out paper
python paper/fig_results.py results
```

## Quick check (2 min)
```bash
python run_experiments.py --dataset synthetic --engine mggp --base_features 1 --fitness ap \
  --methods GP-RSS EVAS --seeds 0-1 --gens 20 --pop 60 --out results_test
```

## Protocol
Seed s fixes both the stratified 60/20/20 split and the GP random stream, so run s of every method uses the
same split (paired tests). Training data only is seen during evolution; the validation set is used once to
select the archived models and the threshold; the test set is used once to report the results
(`tests/test_all.py` checks that the evolution is unchanged when validation and test are replaced by noise).

## License
MIT (see LICENSE).
