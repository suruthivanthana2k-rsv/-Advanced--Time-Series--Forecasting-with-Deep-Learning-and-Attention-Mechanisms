#!/usr/bin/env bash
set -e
python data/generate_data.py --out data/synthetic.csv --n 5000
# quick Optuna runs (reduce trials for quick run)
python -m src.train_optuna --data data/synthetic.csv --model lstm --trials 12 --device cpu
python -m src.train_optuna --data data/synthetic.csv --model transformer --trials 12 --device cpu
# Baseline evaluation
python - <<'PY'
from src.backtest import evaluate_baselines
print(evaluate_baselines('data/synthetic.csv'))
PY
echo "Experiments done. Check optuna_*.pkl files for results."
