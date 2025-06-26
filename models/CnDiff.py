import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from cndiff_utils.layers import StepEmbedding, make_beta_schedule
from cndiff_utils.modules import (
    ClassificationCondition,
    Condition,
    Denoiser,
)
from cndiff_utils.utils import extract, get_gammas


class Model(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        self.config = config
        self.device = self.config.device
        config.d_model = config.hidden_dim * (2 if config.use_cond else 1)
        self.num_timesteps = config.timesteps
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
            if self.config.task_name == "classification":
                self.condition_model = ClassificationCondition(config)
            else:
                self.condition_model = Condition(config)

        if self.config.task_name == "classification":
            # self.classifier = nn.Sequential(
            #     nn.AdaptiveAvgPool1d(1),
            #     nn.Flatten(),
            #     nn.Linear(config.feature_dim, self.config.num_class),
            # )
            # self.classifier = nn.Sequential(
            #     nn.Conv1d(config.d_model, 64, kernel_size=3, padding=1),
            #     nn.ReLU(),
            #     nn.AdaptiveAvgPool1d(1),
            #     nn.Flatten(),
            #     nn.Linear(64, config.num_class),
            # )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(config.d_model, self.config.num_class),
            )

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

    def classification(self, x, x_mark_enc, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(x, y_t_batch, t, self.condition_info)
        output = self.classifier(dec_out.permute(0, 2, 1))
        return output

    def anomaly_detection(self, x, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(x, y_t_batch, t, self.condition_info)
        return dec_out

    def forward(self, x, x_mark=None, original_x=None, arg2=None, mask=None):
        if self.config.task_name == "imputation":
            return self.imputation(x, original_x)

        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]
        if self.config.task_name == "classification":
            return self.classification(x, x_mark, t)
        if self.config.task_name == "anomaly_detection":
            return self.anomaly_detection(x, t)

        return None

    def imputation(self, x, original_x):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None

        # Sample random timestep for each sample
        B = x.size(0)
        t = torch.randint(0, self.config.timesteps, (B,), device=x.device).long()

        x_t, _ = self.q_sample(original_x, t)
        # x_t_masked = observed_mask * x + missing_mask * x_t
        pred_noise = self.diffusion_model(x, x_t, t, self.condition_info)
        return pred_noise

    def p_sample_loop(self, batch_y, x):
        """
        Inference for diffusion model
        """
        t = (
            torch.tensor([self.num_timesteps - 1])
            .repeat(batch_y.shape[0])
            .to(self.device)
        )
        y_t = torch.randn_like(batch_y)

        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
            y_t = self.condition_info + y_t
        else:
            self.condition_info = None

        for t in reversed(range(1, self.num_timesteps)):
            y_t = self.p_sample(x, y_t, t)

        z = self.p_sample_t_1to0(x, y_t)
        return z

    def p_sample(self, x, y_t, t):
        t = torch.tensor([t]).to(self.device)

        sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat = get_gammas(
            self.alphas,
            self.one_minus_alphas_bar_sqrt,
            t,
            y_t,
        )

        if self.config.use_cond:
            y_0_reparam = self.forecast(x, y_t, t).to(self.device).detach()
        else:
            y_0_reparam = self.forecast(x, y_t, t).to(self.device).detach()

        if self.config.use_tphi:
            z = torch.randn_like(y_0_reparam)
            t1 = ((gamma_1 * sqrt_alpha_bar_t) + gamma_0) * (
                self.t_phi(batch_y=y_0_reparam, t=t - 1)
            )
            t2 = (gamma_1 * sqrt_alpha_bar_t) * (self.t_phi(batch_y=y_0_reparam, t=t))

            y_t_m_1_hat = (gamma_1 * y_t) - (t2 - t1)

        else:
            z = torch.randn_like(y_t)
            y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t

        if self.config.use_cond:
            y_t_m_1_hat = y_t_m_1_hat + gamma_2
            # * self.condition_info

        y_t_m_1 = y_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(
            self.device
        ) * z.to(self.device)

        return y_t_m_1

    def p_sample_t_1to0(self, x, y_t):
        t = torch.tensor([0]).to(self.device)

        y_0_reparam = self.forecast(x, y_t, t).to(self.device).detach()

        y_t_m_1 = y_0_reparam.to(self.device)

        return y_t_m_1

    def forecast(self, x, y, t):
        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
        else:
            self.condition_info = None

        y_t_batch, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(x, y_t_batch, t, self.condition_info)
        return dec_out


class Tphi(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()

        param1 = (
            config.c_out if config.task_name != "classification" else config.feature_dim
        )
        param2 = config.pred_len

        self.w1 = nn.Parameter(torch.empty(param1, param1))
        self.b1 = nn.Parameter(torch.empty(param1))

        self.w2 = nn.Parameter(torch.empty(param2, param2))
        self.b2 = nn.Parameter(torch.empty(param2))
        self.act = nn.Tanh()
        self.time_emb = StepEmbedding(param1, freq_dim=256)

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

        return out + batch_y
