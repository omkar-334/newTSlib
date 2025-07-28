import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from CnDiff.utils import (
    DataEmbedding,
    DiTBlock,
    StepEmbedding,
    extract,
    get_gammas,
    make_beta_schedule,
    modulate,
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
        if self.config.use_tphi:
            self.t_phi = Tphi_cls(config)

        # model initialisation for condition network
        self.diffusion_model = OldDenoiser(config)
        if self.config.use_cond:
            self.condition_model = condition_cls(config)
        # self.classifier = classifier(config)

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

        # print(y_t.shape, self.condition_info.shape)
        # exit()
        if self.config.use_cond:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * self.condition_info

        return y_t, noise

    def classification(self, x, y, t):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None
        y_t_batch, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        # dec_out = self.classifier(dec_out)
        dec_out = dec_out.squeeze(1)
        # print("dec_out shape:", dec_out.shape)
        # exit()
        return dec_out

    def forward(self, x, y, padding_mask=None):
        y = y.unsqueeze(2)
        n = x.size(0)
        t = torch.randint(low=1, high=self.config.timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]
        if self.config.task_name == "classification":
            return self.classification(x, y, t)

        return None

    def p_sample_loop(self, x, shape):
        """
        Inference for diffusion model
        """
        t = torch.tensor([self.num_timesteps - 1]).repeat(x.shape[0]).to(self.device)
        y_t = torch.randn(shape).unsqueeze(2).to(self.device)

        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
            # print(y_t.shape, self.condition_info.shape)
            # exit()
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

        # y_t_batch, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(y, t, self.condition_info)
        # print(dec_out.shape)
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out


# Decoder module with conditioning support
class OldDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(config.d_model, config.hidden_dim, config.n_emb - 1),
            nn.Linear(config.hidden_dim, 1),
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
        self.input_embedder = DataEmbedding(1, config.hidden_dim, config.n_emb)
        self.k_embedder = StepEmbedding(config.hidden_dim, freq_dim=256)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_depth)])
        self.decoder = OldDecoder(config)
        self.act = nn.Identity()
        self.config = config

        if config.use_cond:
            self.cond_embedder = DataEmbedding(1, config.hidden_dim, config.n_emb)

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
        # print(y.shape)
        h = self.input_embedder(y)
        # print(h.shape)
        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info)
            h = torch.cat([h, cond_info], dim=-1)
        # print(h.shape)
        c = self.k_embedder(k)
        # print(k.shape, c.shape)
        for block in self.blocks:
            h = block(h, c)
        # print(h.shape)
        out = self.decoder(h, c)

        return out.permute(0, 2, 1)


class Tphi_cls(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()

        param1 = 1
        param2 = config.num_class

        self.time_emb = StepEmbedding(param1, freq_dim=256)
        # self.backward_time_emb = StepEmbedding(config.num_class, freq_dim=256)

        self.w1 = nn.Parameter(torch.empty(param1, param1))
        self.b1 = nn.Parameter(torch.empty(param1))

        self.w2 = nn.Parameter(torch.empty(param2, param2))
        self.b2 = nn.Parameter(torch.empty(param2))
        self.act = nn.Tanh()

        self.init_weights(self.w1, self.b1)
        # self.init_weights(self.w2, self.b2)

        # self.backward_mapper = nn.Linear(128, param1, bias=False)

    @staticmethod
    def init_weights(weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, batch_y, t):
        # print("Tphi forward", batch_y.shape, t.shape)
        t_emb = self.time_emb(t).unsqueeze(1)
        # print("t_emb shape:", t_emb.shape)
        out = batch_y + t_emb
        # print(out.shape)

        out = (out.permute(0, 2, 1) @ self.w2.T) + self.b2
        out = out.permute(0, 2, 1)

        out = (out @ self.w1.T) + self.b1
        out = self.act(out)
        # print("out", out.shape)
        return out


class condition_cls(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # self.dec = nn.Sequential(*[nn.Linear(config.seq_len, config.seq_len), nn.Linear(config.seq_len, 1)]) #
        self.dec = nn.Linear(config.seq_len, 1)
        self.dec2 = nn.Linear(config.feature_dim, config.num_class)

    def forward(self, x):
        # print(x.shape)
        out = self.dec(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.dec2(out).permute(0, 2, 1)

        return out
