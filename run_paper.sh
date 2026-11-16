#!/usr/bin/env bash
# Reproduces all the experiments of the paper (configuration of the final campaign).
# Runs are resumable: an interrupted campaign can simply be relaunched.
set -euo pipefail
JOBS=${JOBS:-4}
CFG="--engine mggp --base_features 1 --fitness ap --pop 300 --gens 100 --ens_k 10 --prevalence_fitness 1 --final_refit 0"
O="--jobs $JOBS --out results"
GP="GP-Static GP-RSS EVAS EVAS-Views RSS-Views EVAS-Views-RF"
python -m pytest -q tests
# Credit Card Fraud (data/creditcard.csv): 30 runs; GP-Full 10 runs (about 5 h per run)
python run_experiments.py --dataset creditcard --csv data/creditcard.csv $CFG --methods $GP --seeds 0-29 $O
python run_experiments.py --dataset creditcard --csv data/creditcard.csv $CFG --methods GP-Full --seeds 0-9 $O
python run_experiments.py --dataset creditcard --csv data/creditcard.csv --baselines --seeds 0-29 $O
python run_experiments.py --dataset creditcard --csv data/creditcard.csv $CFG --split temporal \
  --methods EVAS-Views GP-Static GP-RSS --seeds 0-29 $O
# Benchmark data sets (imbalanced-learn)
for d in ozone_level mammography; do
  python run_experiments.py --dataset imb:$d --baselines --seeds 0-29 $O
  python run_experiments.py --dataset imb:$d $CFG --methods $GP GP-Full --seeds 0-29 $O
done
python run_experiments.py --dataset imb:protein_homo --baselines --seeds 0-9 $O
python run_experiments.py --dataset imb:protein_homo $CFG --methods $GP GP-Full --seeds 0-9 $O
# Tables and figures of the paper
python paper/make_tables.py --res results --out paper
python paper/fig_results.py results
