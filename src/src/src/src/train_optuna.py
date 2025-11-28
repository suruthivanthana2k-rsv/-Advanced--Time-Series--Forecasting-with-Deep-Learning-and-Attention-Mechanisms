import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import optuna
from tqdm import tqdm
import joblib

from src.data_module import create_supervised, train_val_test_split_time
from src.models import AttnLSTM, TransformerForecast
from src.utils import rmse

import pandas as pd

def load_data_csv(path, input_len=48, out_len=10, stride=1):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    feats = [c for c in df.columns if c.startswith('feat')]
    data = df[feats].values
    X, Y = create_supervised(data, input_len=input_len, out_len=out_len, stride=stride)
    X_tr, y_tr, X_val, y_val, X_test, y_test = train_val_test_split_time(X, Y, val_ratio=0.1, test_ratio=0.1)
    return X_tr, y_tr, X_val, y_val, X_test, y_test

def train_epoch(model, opt, loss_fn, loader, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

def eval_loss(model, loss_fn, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            preds = model(xb)
            total += loss_fn(preds, yb).item() * xb.size(0)
    return total / len(loader.dataset)

def objective_lstm(trial, X_tr, y_tr, X_val, y_val, device):
    # hyperparams
    hidden = trial.suggest_int("hidden", 32, 256)
    layers = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch = trial.suggest_categorical("batch", [64, 128, 256])

    model = AttnLSTM(input_dim=X_tr.shape[-1], hidden_dim=hidden, num_layers=layers, dropout=dropout, out_steps=y_tr.shape[1])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    best_val = float('inf')
    for epoch in range(1, 31):
        train_epoch(model, opt, loss_fn, tr_loader, device)
        val_loss = eval_loss(model, loss_fn, val_loader, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val = min(best_val, val_loss)
    return best_val

def objective_transformer(trial, X_tr, y_tr, X_val, y_val, device):
    d_model = trial.suggest_int("d_model", 32, 128)
    nhead = trial.suggest_categorical("nhead", [2,4,8])
    nlayers = trial.suggest_int("nlayers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    batch = trial.suggest_categorical("batch", [64, 128, 256])

    model = TransformerForecast(input_dim=X_tr.shape[-1], d_model=d_model, nhead=nhead, num_layers=nlayers, dropout=dropout, out_steps=y_tr.shape[1])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    best_val = float('inf')
    for epoch in range(1, 31):
        train_epoch(model, opt, loss_fn, tr_loader, device)
        val_loss = eval_loss(model, loss_fn, val_loader, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val = min(best_val, val_loss)
    return best_val

def run_optuna(path, model_type='lstm', n_trials=30, seed=42, device='cpu'):
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_data_csv(path)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner())
    if model_type == 'lstm':
        func = lambda trial: objective_lstm(trial, X_tr, y_tr, X_val, y_val, device)
    else:
        func = lambda trial: objective_transformer(trial, X_tr, y_tr, X_val, y_val, device)
    study.optimize(func, n_trials=n_trials, timeout=None)
    print("Best params:", study.best_params)
    joblib.dump(study, f"optuna_{model_type}_study.pkl")
    return study

if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic.csv')
    parser.add_argument('--model', choices=['lstm','transformer'], default='lstm')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run_optuna(args.data, model_type=args.model, n_trials=args.trials, device=args.device)
