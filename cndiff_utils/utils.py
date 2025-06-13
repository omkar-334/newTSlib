import os

import numpy as np
import torch
import torch.nn.functional as F


def extract(tensor, t, x):
    shape = x.shape
    out = torch.gather(tensor, 0, t.to(tensor.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def get_gammas(alphas, one_minus_alphas_bar_sqrt, t, y_t, squeeze=False):
    alpha_t = extract(alphas, t, y_t)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y_t)

    if squeeze:
        alpha_t = alpha_t.squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t_m_1 = (
            (sqrt_one_minus_alpha_bar_t_m_1).squeeze(1).squeeze(1)
        )

    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

    gamma_0 = (
        (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_1 = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        * (alpha_t.sqrt())
        / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square()
    )

    beta_t_hat = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        / (sqrt_one_minus_alpha_bar_t.square())
        * (1 - alpha_t)
    )
    return sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat


def modify_gammas(sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat):
    return (
        sqrt_alpha_bar_t.unsqueeze(1).unsqueeze(2),
        gamma_0.unsqueeze(1).unsqueeze(2),
        gamma_1.unsqueeze(1).unsqueeze(2),
        gamma_2.unsqueeze(1).unsqueeze(2),
        beta_t_hat.unsqueeze(1).unsqueeze(2),
    )


# preprocessing in dlinear
def instance_normalization(x, y0):
    x_mean = x[:, -1:, :]
    x_std = torch.ones_like(x_mean)
    x_norm = (x - x_mean) / x_std
    y0_norm = (y0 - x_mean) / x_std
    return x_norm, y0_norm, x_mean, x_std


def instance_denormalization(y0, mean, std, pred_len):
    B = mean.shape[0]
    n_samples = y0.shape[0] // B
    std = torch.repeat_interleave(std, n_samples, dim=0).repeat(1, pred_len, 1)
    mean = torch.repeat_interleave(mean, n_samples, dim=0).repeat(1, pred_len, 1)
    y0 = y0 * std + mean
    return y0


def calculate_mse_mae(preds_save, trues_save):
    preds_save = np.mean(preds_save, axis=2)

    mse = F.mse_loss(torch.tensor(preds_save), torch.tensor(trues_save))
    mae = F.l1_loss(torch.tensor(preds_save), torch.tensor(trues_save))
    print(f"mse: {mse}, mae: {mae}")
    return mse, mae


class EarlyStopping:
    def __init__(self, config, patience=7, verbose=False, delta=0) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.device = config.device
        self.run_name = config.run_name

    def __call__(self, val_loss, model):
        is_best = False
        score = -val_loss
        if val_loss is None:
            self.save_checkpoint(val_loss, model)
            print("Early Stopping due to NAN values")
            self.early_stop = True
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
                is_best = True
            return is_best
        return None

    def save_checkpoint(self, val_loss, model, path: str = "checkpoints") -> None:
        os.makedirs(path, exist_ok=True)
        self.ckpt_model = model
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + f"{self.run_name}.pth")
        self.val_loss_min = val_loss
