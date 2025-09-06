import csv
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


def save_results(task, setting: str, metrics: dict, sweep=False):
    print(setting, end="---")
    for key in metrics:
        print(f"{key} - {metrics[key]}", end="---")

    task = f"{task}_sweep" if sweep else task
    json_path = f"./results/{task}_results.json"
    csv_path = f"./results/{task}_results.csv"

    if os.path.exists(json_path):
        with open(json_path) as f:
            try:
                results_dict = json.load(f)
            except json.JSONDecodeError:
                results_dict = {}
    else:
        results_dict = {}

    if setting in results_dict:
        setting += "_new"
    results_dict[setting] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }

    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    fieldnames = ["setting", "time"] + list(metrics.keys())
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"setting": setting, "time": results_dict[setting]["time"], **metrics}
        writer.writerow(row)


def save_preds(setting, preds, trues):
    path = "results/{}_{}"
    print("test shape:", preds.shape, trues.shape)
    np.save(path.format(setting, "pred.npy"), preds)
    np.save(path.format(setting, "true.npy"), trues)
