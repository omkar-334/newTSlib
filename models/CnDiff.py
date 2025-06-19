import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from cndiff_utils.layers import StepEmbedding, make_beta_schedule
from cndiff_utils.modules import Condition, Denoiser
from cndiff_utils.utils import extract


class Model(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        self.config = config
        self.device = self.config.device
        config.d_model = config.hidden_dim * (2 if config.use_cond else 1)

        # betas and alphas for diffusion
        betas = make_beta_schedule(
            schedule=self.config.beta_schedule,
            num_timesteps=self.config.timesteps,
            start=self.config.beta_start,
            end=self.config.beta_end,
        )

        betas = betas.float().to(self.device)
        alphas = 1.0 - betas
        self.alphas = alphas
        alphas_cumprod = alphas.to("cpu").cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if self.config.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= (
                0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            )
        if self.config.use_tphi:
            self.t_phi = Tphi(config)

        # model initialisation for condition network
        self.diffusion_model = Denoiser(config)
        if self.config.use_cond:
            self.condition_model = Condition(config)

        if self.config.task_name == "classification":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(config.d_model, self.config.num_class),
            )
        if self.config.task_name in {"anomaly_detection", "imputation"}:
            self.projection = nn.Linear(config.d_model, config.c_out, bias=True)

    def q_sample(self, batch_y, t):
        """
        Forward process for conditional and learnable mean
        """
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        if self.config.use_tphi:
            batch_y_trans = self.t_phi(t=t, batch_y=batch_y)  # type: ignore
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y_trans + sqrt_one_minus_alpha_bar_t * noise

        else:
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y + sqrt_one_minus_alpha_bar_t * noise

        if self.config.use_cond:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * self.condition_info

        return y_t, noise

    def classification(self, x, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(x, y_t_batch, t, self.condition_info)
        dec_out = self.classifier(dec_out.permute(0, 2, 1))

        # return logits
        return dec_out

    def anomaly_detection(self, x, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(x, y_t_batch, t, self.condition_info)
        dec_out = self.projection(dec_out)
        return dec_out

    def forward(self, x, x_mark, arg1, arg2, mask):
        if self.config.task_name == "imputation":
            return self.imputation(x, mask)

        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]
        if self.config.task_name == "classification":
            return self.classification(x, t)
        if self.config.task_name == "anomaly_detection":
            return self.anomaly_detection(x, t)

        return None

    def imputation(self, x, mask):
        observed_mask = mask
        missing_mask = 1 - observed_mask

        self.condition_info = self.condition_model(x) if self.config.use_cond else None

        # Sample random timestep for each sample
        B = x.size(0)
        t = torch.randint(0, self.config.timesteps, (B,), device=x.device).long()

        x_t, _ = self.q_sample(x, t)

        # Mix observed values with noised input
        x_t_masked = observed_mask * x + missing_mask * x_t

        # Predict noise using diffusion model
        pred_noise = self.diffusion_model(x, x_t_masked, t, self.condition_info)

        # Project to correct output dim
        pred_noise = self.projection(pred_noise)

        return pred_noise


class Tphi(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()

        self.w1 = nn.Parameter(torch.empty(config.feature_dim, config.feature_dim))
        self.b1 = nn.Parameter(torch.empty(config.feature_dim))

        param = config.pred_len
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
