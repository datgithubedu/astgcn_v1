import numpy as np


def mae(y_pred, y_true):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_pred, y_true):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def mape(y_pred, y_true):
    """Mean Absolute Percentage Error"""
    eps = 1e-5
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps)))


def r2_score(y_pred, y_true):
    """Coefficient of Determination (R²)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0  # tránh chia 0

    return 1 - (ss_res / ss_tot)
