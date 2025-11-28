import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def scale_train_val_test(X_train, X_val, X_test):
    n_features = X_train.shape[-1]
    scalers = []
    X_train_s = np.zeros_like(X_train)
    X_val_s = np.zeros_like(X_val)
    X_test_s = np.zeros_like(X_test)
    for feat in range(n_features):
        s = StandardScaler()
        X_feat_train = X_train[..., feat].reshape(-1,1)
        s.fit(X_feat_train)
        scalers.append(s)
        # reshape back
        def _transform(X):
            orig_shape = X.shape
            flat = X[..., feat].reshape(-1,1)
            flat_t = s.transform(flat).reshape(orig_shape[:-1] + (1,))
            return flat_t
        X_train_s[..., feat] = _transform(X_train).squeeze(-1)
        X_val_s[..., feat] = _transform(X_val).squeeze(-1)
        X_test_s[..., feat] = _transform(X_test).squeeze(-1)
    return X_train_s, X_val_s, X_test_s, scalers

def inverse_transform_predictions(preds, scalers, feature_index=0):
    # preds: (batch, steps) or (..., steps)
    s = scalers[feature_index]
    shape = preds.shape
    flat = preds.reshape(-1,1)
    inv = s.inverse_transform(flat).reshape(shape)
    return inv
