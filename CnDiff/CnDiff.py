from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

from CnDiff.utils import (
    Condition,
    Denoiser,
    KanTphi,
    Tphi,
    extract,
    get_gammas,
    make_beta_schedule,
)


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
        if self.config.use_tphi == 1:
            self.t_phi = Tphi(config)
        elif self.config.use_tphi == 2:
            self.t_phi = KanTphi(config)

        # model initialisation for condition network
        self.diffusion_model = Denoiser(config)
        if self.config.use_cond:
            self.condition_model = Condition(config)

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

    def anomaly_detection(self, x, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        return dec_out

    def forward(self, x, original_x=None, padding_mask=None):
        if self.config.task_name == "imputation":
            return self.imputation(x, original_x)

        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]
        self.t = t
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
        pred_noise = self.diffusion_model(x_t, t, self.condition_info)
        return pred_noise

    def p_sample_loop(self, x):
        """
        Inference for diffusion model
        """
        t = torch.tensor([self.num_timesteps - 1]).repeat(x.shape[0]).to(self.device)
        y_t = torch.randn_like(x)

        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
            y_t = self.condition_info + y_t
        else:
            self.condition_info = None

        for t in tqdm(
            range(self.num_timesteps - 1, 0, -1), desc="Reverse diffusion sampling"
        ):
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
            y_t_m_1_hat = y_t_m_1_hat + gamma_2 * self.condition_info

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

        # y_t_batch, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(y, t, self.condition_info)
        return dec_out

    def get_prior(self, batch_y, cond_info=None):
        """
        Prior loss term in transformed forward process
        """
        T = (
            torch.tensor([self.num_timesteps - 1])
            .repeat(batch_y.shape[0])
            .to(self.device)
        )
        batch_y_mean = self.t_phi(t=T, batch_y=batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, T, batch_y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

        u = sqrt_alpha_bar_t * batch_y_mean

        if self.config.use_cond:
            u = u - (sqrt_alpha_bar_t) * cond_info

        return (1 / 2) * (torch.mean((u) ** 2, dim=(1, 2)))

    def get_mu_t_phi_loss(self, pred_noise, batch_y, t, condition_info=None):
        gamma_0, gamma_1, gamma_2, sqrt_alpha_bar_t, beta_t_hat = get_gammas(
            self.alphas,
            self.one_minus_alphas_bar_sqrt,
            t,
            pred_noise,
        )

        term_1 = (gamma_1 * sqrt_alpha_bar_t) * (
            self.t_phi(batch_y=pred_noise, t=t) - self.t_phi(batch_y=batch_y, t=t)
        )
        term_2 = ((gamma_1 * sqrt_alpha_bar_t) + gamma_0) * (
            self.t_phi(batch_y=batch_y, t=t - 1)
            - self.t_phi(batch_y=pred_noise, t=t - 1)
        )

        diff_term = (torch.mean((term_1 + term_2) ** 2, dim=(1, 2), keepdim=True)) * (
            1 / (2 * beta_t_hat)
        )

        prior_term = self.get_prior(batch_y=batch_y, cond_info=condition_info)
        recon_term = torch.mean((pred_noise - batch_y) ** 2, dim=(1, 2), keepdim=True)

        return torch.mean(diff_term + prior_term + recon_term)
