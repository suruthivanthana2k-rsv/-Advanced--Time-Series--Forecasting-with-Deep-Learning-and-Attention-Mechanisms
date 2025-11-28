import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_module import create_supervised, train_val_test_split_time
from src.models import AttnLSTM, TransformerForecast
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils import rmse, mae, mape, inverse_transform_predictions
import joblib
import warnings
warnings.filterwarnings("ignore")

def simple_sarimax_backtest(series, exog=None, order=(1,0,1), seasonal_order=(0,0,0,0), forecast_steps=10):
    # series: 1D array for target (we'll use feat0 as target)
    n = len(series)
    window = int(n*0.7)
    preds = []
    truth = []
    for start in range(window, n - forecast_steps, forecast_steps):
        train = series[:start]
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        f = res.forecast(steps=forecast_steps)
        preds.append(f)
        truth.append(series[start:start+forecast_steps])
    preds = np.concatenate(preds)
    truth = np.concatenate(truth)
    return preds, truth

def evaluate_baselines(df_path='data/synthetic.csv'):
    df = pd.read_csv(df_path)
    feats = [c for c in df.columns if c.startswith('feat')]
    series = df[feats[0]].values
    preds, truth = simple_sarimax_backtest(series, forecast_steps=10)
    r = rmse(truth, preds)
    a = mae(truth, preds)
    mp = mape(truth, preds)
    return {"sarimax": {"rmse": r, "mae": a, "mape": mp}}

def evaluate_dl_model(model_file, data_csv='data/synthetic.csv', input_len=48, out_len=10, scalers=None, device='cpu'):
    # model_file could be a saved state dict or a pickled torch model (simplify: we expect a saved state_dict and model class info)
    raise NotImplementedError("Use train/export code to save final models and call evaluation script in repo's evaluate.py")
