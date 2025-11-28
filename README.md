# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

This repo implements:
- Synthetic multivariate time series generator with seasonality, trend, regime shifts (5 features)
- Attention-LSTM and Transformer multi-step forecasting (predict next 10 steps)
- Baselines: SARIMAX and vanilla LSTM
- Bayesian hyperparameter tuning via Optuna
- Walk-forward backtesting and evaluation (RMSE, MAE, MAPE)
- Plots & final report scaffolding

## Quick start
1. Create virtual env and install requirements:
   bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
