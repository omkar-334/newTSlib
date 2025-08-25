import math

import torch
import torch.nn as nn

from .katransformer import KanBlock
from .utils import modulate


class StepEmbedding(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        super().__init__()

        """
        Time embedding used in T_phi, and for time embedding
        """

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(k, freq_dim, max_period=1000):
        half_dim = freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=k.device)
        k_freqs = k[:, None].float() * freqs[None]
        k_emb = torch.cat([torch.cos(k_freqs), torch.sin(k_freqs)], dim=-1)
        return k_emb

    def forward(self, k):
        k_emb = self.sinusoidal_embedding(k, self.freq_dim)
        k_emb = self.mlp(k_emb)
        return k_emb


class MLPResidual(nn.Module):
    """
    Simple MLP residual network with one hidden state.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin_emb = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.lin_res = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x_emb = self.lin_emb(x)
        x_res = self.lin_res(x)
        x_out = self.norm(x_emb + x_res)
        return x_out


class DataEmbedding(nn.Module):
    """
    embed for x and y
    """

    def __init__(self, in_dim, out_dim, n_emb):
        super().__init__()
        layers = [MLPResidual(in_dim, out_dim)]
        if n_emb > 1:
            layers.extend(MLPResidual(out_dim, out_dim) for _ in range(n_emb - 1))
        self.feat_embedding = nn.Sequential(*layers)

    def forward(self, x):
        return self.feat_embedding(x)


# Decoder module with conditioning support
class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(config.d_model, config.d_model, config.n_emb - 1),
            nn.Linear(config.d_model, config.pred_len),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2 * config.d_model, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x


# Full model
class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_embedder = DataEmbedding(
            config.pred_len, config.hidden_dim, config.n_emb
        )
        self.k_embedder = StepEmbedding(config.hidden_dim, freq_dim=256)
        # self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_depth)])
        self.blocks = nn.ModuleList([
            KanBlock(
                dim=config.d_model,
                num_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                config=config,
            )
            for _ in range(config.n_depth)
        ])
        self.decoder = Decoder(config)
        self.act = nn.Identity()

        self.config = config
        if config.use_cond:
            self.cond_embedder = DataEmbedding(
                config.pred_len, config.hidden_dim, config.n_emb
            )

    def forward(self, y, k, cond_info):
        """
        y: (B, prediction_length, num_feat)
        k: (B,)
        cond_info: (B, context_length, num_feat)
        """
        y = self.input_embedder(y.permute(0, 2, 1))

        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info.permute(0, 2, 1))
            # cond_info = F.layer_norm(cond_info, cond_info.shape[-1:])
            y = torch.cat([y, cond_info], dim=-1)

        c = self.k_embedder(k)

        for block in self.blocks:
            y = block(y, c)

        out = self.decoder(y, c).permute(0, 2, 1)

        if self.config.task_name != "classification":
            out = self.act(out)
        return out
