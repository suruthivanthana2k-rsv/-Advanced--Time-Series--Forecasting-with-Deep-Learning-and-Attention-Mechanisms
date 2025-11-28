import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_module import create_supervised, train_val_test_split_time
from src.utils import rmse, mae, mape, inverse_transform_predictions
from sklearn.preprocessing import StandardScaler

def evaluate_saved_model(model, X_test, y_test, scalers, device='cpu', plot_n=3):
    model.to(device)
    model.eval()
    X_t = torch.tensor(X_test).float().to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()  # (N, out_len, features)
    # choose first feature for metrics
    preds_f0 = preds[..., 0]
    y_f0 = y_test[..., 0]
    # inverse scale
    inv_preds = inverse_transform_predictions(preds_f0, scalers, feature_index=0)
    inv_truth = inverse_transform_predictions(y_f0, scalers, feature_index=0)
    r = rmse(inv_truth.flatten(), inv_preds.flatten())
    a = mae(inv_truth.flatten(), inv_preds.flatten())
    mp = mape(inv_truth.flatten(), inv_preds.flatten())
    print("RMSE:", r, "MAE:", a, "MAPE:", mp)
    # plots
    for i in range(min(plot_n, len(inv_preds))):
        plt.figure(figsize=(8,3))
        plt.plot(inv_truth[i], label='truth')
        plt.plot(inv_preds[i], label='pred')
        plt.legend()
        plt.title(f"Example {i}")
        plt.show()
    return {"rmse": r, "mae": a, "mape": mp}

if _name_ == "_main_":
    print("Run evaluation from the notebook or call evaluate_saved_model with a trained model and scalers.")
