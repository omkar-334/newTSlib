from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.num_class = getattr(config, "num_class", 1)

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
            self.one_minus_alphas_bar_sqrt *= 0.9999

        if self.config.use_tphi == 1:
            self.t_phi = Tphi(config)
        elif self.config.use_tphi == 2:
            self.t_phi = KanTphi(config)

        self.diffusion_model = Denoiser(config)
        if self.config.use_cond:
            self.condition_model = Condition(config)

    def q_sample(self, batch_y, t):
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        if self.config.use_tphi:
            batch_y_trans = self.t_phi(t=t, batch_y=batch_y)
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y_trans + sqrt_one_minus_alpha_bar_t * noise
        else:
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y + sqrt_one_minus_alpha_bar_t * noise

        # Only add condition_info if it exists (set in p_sample_loop_classification)
        if self.config.use_cond and getattr(self, "condition_info", None) is not None:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * self.condition_info

        return y_t, noise

    def anomaly_detection(self, x, t):
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        return dec_out

    def forward(self, x, cond_info=None, padding_mask=None):
        n = x.size(0)
        t = torch.randint(
            low=1, high=self.config.timesteps, size=(n // 2 + 1,), device=self.device
        )
        self.t = t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]

        if self.config.task_name == "imputation":
            self.condition_info = (
                self.condition_model(x) if self.config.use_cond else None
            )
            return self.imputation(cond_info, t)
        if self.config.task_name == "anomaly_detection":
            self.condition_info = (
                self.condition_model(x) if self.config.use_cond else None
            )
            return self.anomaly_detection(x, t)
        if self.config.task_name == "classification":
            # For classification, use cond_info (labels_inp) as the conditioning tensor
            return self.classification(x, t, cond_info)
        return None

    def imputation(self, original_x, t):
        x_t, _ = self.q_sample(original_x, t)
        pred_noise = self.diffusion_model(x_t, t, self.condition_info)
        return pred_noise

    def classification(self, original_x, t, cond_info=None):
        x_t, _ = self.q_sample(original_x, t)
        # For classification, use cond_info (labels_inp) as the conditioning tensor
        pred_noise = self.diffusion_model(x_t, t, cond_info)
        return pred_noise

    def p_sample_loop(self, x):
        t = torch.tensor([self.num_timesteps - 1]).repeat(x.shape[0]).to(self.device)
        y_t = torch.randn_like(x)

        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
            y_t = self.condition_info + y_t
        else:
            self.condition_info = None

        for t in reversed(range(1, self.num_timesteps)):
            y_t = self.p_sample(x, y_t, t)

        z = self.p_sample_t_1to0(x, y_t)
        return z

    def p_sample_loop_classification(self, x):
        batch_size = x.shape[0]
        recons = []

        for cls in range(self.num_class):
            class_labels = torch.full(
                (batch_size,), cls, dtype=torch.long, device=self.device
            )
            label_cond = F.one_hot(class_labels, num_classes=self.num_class).float()
            # Expand to [batch, seq_len, num_class] for Condition
            label_cond = label_cond.unsqueeze(1).expand(-1, x.shape[1], -1)

            self.condition_info = (
                self.condition_model(label_cond) if self.config.use_cond else None
            )

            y_t = torch.randn_like(x)
            if self.config.use_cond:
                y_t = self.condition_info + y_t

            for t in reversed(range(1, self.num_timesteps)):
                y_t = self.p_sample(x, y_t, t)

            z = self.p_sample_t_1to0(x, y_t)
            recons.append(z.unsqueeze(1))

        return torch.cat(recons, dim=1)  # [B, num_class, L, D]

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
            t1 = ((gamma_1 * sqrt_alpha_bar_t) + gamma_0) * self.t_phi(
                batch_y=y_0_reparam, t=t - 1
            )
            t2 = (gamma_1 * sqrt_alpha_bar_t) * self.t_phi(batch_y=y_0_reparam, t=t)
            y_t_m_1_hat = (gamma_1 * y_t) - (t2 - t1)
        else:
            z = torch.randn_like(y_t)
            y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t

        if self.config.use_cond and self.condition_info is not None:
            y_t_m_1_hat = y_t_m_1_hat + gamma_2 * self.condition_info

        y_t_m_1 = y_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(
            self.device
        ) * z.to(self.device)

        return y_t_m_1

    def p_sample_t_1to0(self, x, y_t):
        t = torch.tensor([0]).to(self.device)
        y_0_reparam = self.forecast(x, y_t, t).to(self.device).detach()
        return y_0_reparam

    def forecast(self, x, y, t):
        if self.config.use_cond and self.config.task_name != "classification":
            self.condition_info = self.condition_model(x)
        else:
            self.condition_info = None

        dec_out = self.diffusion_model(y, t, self.condition_info)
        return dec_out

    def get_prior(self, batch_y, cond_info=None):
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
            u = u - sqrt_alpha_bar_t * cond_info

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
