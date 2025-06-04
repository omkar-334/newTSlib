import json
import os
from datetime import datetime

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def save_results(task, setting: str, metrics: dict):
    print(setting, end="---")
    for key in metrics:
        print(f"{key} - {metrics[key]}", end="---")

    json_path = f"./{task}_results.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    results_dict[setting] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }

    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)


def save_preds(setting, preds, trues):
    path = "results/{}_{}"
    print("test shape:", preds.shape, trues.shape)
    np.save(path.format(setting, "pred.npy"), preds)
    np.save(path.format(setting, "true.npy"), trues)
