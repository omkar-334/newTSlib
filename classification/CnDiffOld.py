from types import SimpleNamespace

import torch
import torch.nn as nn

from cndiff_utils.class_utils import Tphi, classifier, condition
from cndiff_utils.layers import (
    DataEmbedding,
    StepEmbedding,
    make_beta_schedule,
)
from cndiff_utils.modules import DiTBlock
from cndiff_utils.utils import extract, get_gammas, modulate


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
        self.diffusion_model = OldDenoiser(config)
        if self.config.use_cond:
            self.condition_model = condition(config)
        self.classifier = classifier(config)

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

    def classification(self, x, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        dec_out = self.classifier(dec_out)
        return dec_out

    def forward(self, x, padding_mask=None):
        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]
        if self.config.task_name == "classification":
            return self.classification(x, t)

        return None

    def p_sample_loop(self,  x):
        """
        Inference for diffusion model
        """
        t = (
            torch.tensor([self.num_timesteps - 1])
            .repeat(x.shape[0])
            .to(self.device)
        )
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

        y_t_batch, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        return dec_out


# Decoder module with conditioning support
class OldDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(config.d_model, config.hidden_dim, config.n_emb - 1),
            nn.Linear(config.hidden_dim, config.d_model),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_dim, 2 * config.d_model, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x


class OldDenoiser(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_embedder = DataEmbedding(
            config.feature_dim, config.hidden_dim, config.n_emb
        )
        self.k_embedder = StepEmbedding(config.hidden_dim, freq_dim=256)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_depth)])
        self.decoder = OldDecoder(config)
        self.act = nn.Identity()
        self.config = config

        if config.use_cond:
            self.cond_embedder = DataEmbedding(
                config.feature_dim, config.hidden_dim, config.n_emb
            )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)

    def forward(self, y, k, cond_info):
        """
        y: (B, prediction_length, num_feat)
        k: (B,)
        cond_info: (B, context_length, num_feat)
        """
        h = self.input_embedder(y)

        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info)
            h = torch.cat([h, cond_info], dim=-1)

        c = self.k_embedder(k)

        for block in self.blocks:
            h = block(h, c)

        out = self.decoder(h, c)

        return out.permute(0, 2, 1)
