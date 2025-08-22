import torch
import torch.nn as nn

from .katransformer import KanBlock
from .layers import (
    AttnMLP,
    DataEmbedding,
    FullAttention,
    StepEmbedding,
)
from .utils import modulate


# Encoder
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, config) -> None:
        super().__init__()
        d_model = config.d_model
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = FullAttention(
            d_model=d_model, n_heads=config.n_heads, attn_dropout=config.attn_dropout
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model * config.mlp_ratio)
        self.mlp = AttnMLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_dim, 6 * d_model, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, num_feat, d_model), d_model=hidden_dim*2
        c: (B, hidden_dim)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod, x_mod, x_mod)
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x


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
