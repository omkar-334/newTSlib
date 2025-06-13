import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from cndiff_utils.layers import StepEmbedding, make_beta_schedule
from cndiff_utils.modules import Condition, Denoiser
from cndiff_utils.utils import extract

defaults = {
    # === Default Settings ===
    "seed": 2024,
    "run_name": "exchange",
    "test": False,
    # === Data Configuration ===
    "dataset": "exchange",
    "split_ratio": [0.7, 0.1, 0.2],  # train, val, test
    # "pred_len": 14,
    "pred_len": 96,
    "d_model": 128,
    "data_path": "data/exchange_rate.csv",
    "scale": True,  # normalize data
    "target": "OT",  # target column
    "features": "M",
    "timeenc": 1,
    "freq": "d",
    "batch_size": 64,
    "test_batch_size": 32,
    "shuffle": True,
    "drop_last": True,
    # === Training Configuration ===
    "device": "cuda:0",
    "gpu_type": "cuda",
    "optimizer": "Adam",
    "epochs": 100,
    "lr": 0.001,
    "patience": 10,
    "mask_rate": 0.375,
    # === Model Configuration ===
    "hidden_dim": 64,
    "n_emb": 2,
    "n_heads": 8,
    "attn_dropout": 0.1,
    "mlp_ratio": 1,
    "n_depth": 1,
    # === Conditional Usage Flag ===
    "use_cond": False,
    # === Diffusion Configuration ===
    "noise_type": "t_phi",
    "beta_schedule": "quad",
    "beta_start": 0.0001,
    "beta_end": 0.1,
    "timesteps": 100,
    "model_type": "CN_Diff",
    "n_copies_to_test": 10,
}


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        merged = {**vars(config), **defaults}
        self.config = config = SimpleNamespace(**merged)

        self.device = self.config.device

        # betas and alphas for diffusion
        betas = make_beta_schedule(
            schedule=self.config.beta_schedule,
            num_timesteps=self.config.timesteps,
            start=self.config.beta_start,
            end=self.config.beta_end,
        )

        betas = self.betas = betas.float().to(self.device)
        alphas = 1.0 - betas
        self.alphas = alphas
        alphas_cumprod = alphas.to("cpu").cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if self.config.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= (
                0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            )

        # self.pred_len = config.pred_len
        self.noise_type = self.config.noise_type

        # model initialisation for condition network
        self.diffusion_model = Denoiser(config)
        if self.config.use_cond:
            self.condition_model = Condition(config)

        self.t_phi = Tphi(config)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Aggregate over features
            nn.Flatten(),
            nn.Linear(64, self.config.num_class),
        )

    def q_sample(self, batch_y, t):
        """
        Forward process for conditional and learnable mean
        """
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        if self.noise_type == "t_phi":
            batch_y_trans = self.t_phi(t=t, batch_y=batch_y)
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y_trans + sqrt_one_minus_alpha_bar_t * noise

        else:
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y + sqrt_one_minus_alpha_bar_t * noise

        if self.config.use_cond:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * self.condition_info

        return y_t, noise

    def classification(self, x, y, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(x, y_t_batch, t, self.condition_info)
        logits = self.classifier(dec_out)
        return logits

    def forward(self, x, mask, *args):
        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]
        # print(x.shape, t.shape, mask.shape)
        # print(t)
        return self.classification(x, x, t)


class Tphi(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()

        self.w1 = nn.Parameter(torch.empty(config.feature_dim, config.feature_dim))
        self.b1 = nn.Parameter(torch.empty(config.feature_dim))

        param = config.seq_len
        self.w2 = nn.Parameter(torch.empty(param, param))
        self.b2 = nn.Parameter(torch.empty(param))
        self.act = nn.Tanh()
        self.time_emb = StepEmbedding(config.feature_dim, freq_dim=256)

        self.init_weights(self.w1, self.b1)

    @staticmethod
    def init_weights(weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, batch_y, t):
        t_emb = self.time_emb(t).unsqueeze(1)
        out = batch_y + t_emb

        out = (out.permute(0, 2, 1) @ self.w2.T) + self.b2
        out = out.permute(0, 2, 1)

        out = (out @ self.w1.T) + self.b1
        out = self.act(out)

        return out
