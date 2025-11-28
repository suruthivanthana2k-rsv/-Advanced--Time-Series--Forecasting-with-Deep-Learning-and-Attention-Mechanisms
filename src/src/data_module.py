import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_supervised(data: np.ndarray, input_len=48, out_len=10, stride=1):
    """
    data: (T, features)
    returns X (N, input_len, features), y (N, out_len, features)
    """
    T, F = data.shape
    X, Y = [], []
    i = 0
    while i + input_len + out_len <= T:
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+out_len])
        i += stride
    return np.array(X), np.array(Y)

def train_val_test_split_time(X, y, val_ratio=0.1, test_ratio=0.1):
    N = len(X)
    test_n = int(N * test_ratio)
    val_n = int(N * val_ratio)
    train_n = N - val_n - test_n
    X_train = X[:train_n]
    y_train = y[:train_n]
    X_val = X[train_n:train_n+val_n]
    y_val = y[train_n:train_n+val_n]
    X_test = X[train_n+val_n:]
    y_test = y[train_n+val_n:]
    return X_train, y_train, X_val, y_val, X_test, y_test
