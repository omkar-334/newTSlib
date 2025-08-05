from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F

from CnDiff.utils import (
    Condition,
    KanTphi,
    Tphi,
    TransformerCondition,
    extract,
    get_gammas,
    invalid,
    make_beta_schedule,
)
from CnDiff.utils.modules import (
    CDenoiser,
)


class Model(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        self.num_class = config.num_class
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
            self.one_minus_alphas_bar_sqrt *= 0.9999

        if self.config.use_tphi == 1:
            self.t_phi = Tphi(config)
        elif self.config.use_tphi == 2:
            self.t_phi = KanTphi(config)

        self.diffusion_model = CDenoiser(config)
        if self.config.use_cond == 1:
            self.condition_model = Condition(config)
        elif self.config.use_cond == 2:
            self.condition_model = TransformerCondition(config)

        self.parameter_dict = {
            "diffusion_model": sum(
                p.numel() for p in self.diffusion_model.parameters()
            ),
            "condition_model": sum(
                p.numel() for p in self.condition_model.parameters()
            ),
            "t_phi": sum(p.numel() for p in self.t_phi.parameters()),
        }

    def forward(self, x, original_x=None, padding_mask=None, label=None):
        # This is the training forward pass
        self.condition_info = self.condition_model(x) if self.config.use_cond else None

        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]

        y_0 = original_x if original_x is not None else x
        y_t, noise = self.q_sample(y_0, t)

        pred_noise = self.diffusion_model(y_t, t, self.condition_info, class_idx=label)

        return pred_noise, noise

    @torch.no_grad()
    def classify_by_reconstruction(self, x):
        batch_size = x.shape[0]
        errors = torch.zeros(batch_size, self.num_class, device=self.device)

        for class_idx in range(self.num_class):
            reconstructed = self.p_sample_loop(x, class_idx)

            error = F.mse_loss(reconstructed, x, reduction="none").mean(dim=[1, 2])
            errors[:, class_idx] = error

        return errors

    def q_sample(self, batch_y, t):
        """
        Forward process for conditional and learnable mean
        """
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        if self.config.use_tphi:
            batch_y_trans = self.t_phi(t=t, batch_y=batch_y)
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y_trans + sqrt_one_minus_alpha_bar_t * noise

        else:
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y + sqrt_one_minus_alpha_bar_t * noise

        if self.config.use_cond:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * self.condition_info

        return y_t, noise

    def p_sample_loop(self, x, class_idx):
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

        for t in reversed(range(1, self.num_timesteps)):
            y_t = self.p_sample(x, y_t, t, class_idx)

        z = self.p_sample_t_1to0(x, y_t, class_idx)
        return z

    def p_sample(self, x, y_t, t, class_idx):
        t = torch.tensor([t]).to(self.device)

        sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat = get_gammas(
            self.alphas,
            self.one_minus_alphas_bar_sqrt,
            t,
            y_t,
        )
        y_0_reparam = self.forecast(x, y_t, t, class_idx).detach()

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

    def p_sample_t_1to0(self, x, y_t, class_idx):
        t = torch.tensor([0]).to(self.device)

        y_0_reparam = self.forecast(x, y_t, t, class_idx).detach()

        y_t_m_1 = y_0_reparam.to(self.device)

        return y_t_m_1

    def forecast(self, x, y, t, class_idx):
        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
        else:
            self.condition_info = None

        # dec_out = self.diffusion_model(y, t, self.condition_info)
        dec_out = self.diffusion_model(y, t, self.condition_info, class_idx)
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
            u = u - sqrt_alpha_bar_t * cond_info

        return (1 / 2) * (torch.mean((u) ** 2, dim=(1, 2)))

    def get_mu_t_phi_loss(self, pred_noise, batch_y, t, condition_info=None, mask=None):
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

        diff_term = (torch.mean((term_1 + term_2) ** 2, dim=(1, 2), keepdim=True)) / (
            2 * beta_t_hat + 1e-6
        )

        prior_term = self.get_prior(batch_y=batch_y, cond_info=condition_info)

        if mask is not None:
            outputs = torch.where(mask.bool(), batch_y, pred_noise)
        else:
            outputs = pred_noise
        recon_term = torch.mean((outputs - batch_y) ** 2, dim=(1, 2), keepdim=True)

        for term_name, term in [
            ("diff_term", diff_term),
            ("prior_term", prior_term),
            ("recon_term", recon_term),
        ]:
            if invalid(term_name, term):
                print("pred_noise:", pred_noise)
                exit()

        return torch.mean(diff_term + prior_term + recon_term)
