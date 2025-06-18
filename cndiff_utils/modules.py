import torch
import torch.nn as nn

from cndiff_utils.layers import (
    AttnMLP,
    DataEmbedding,
    FullAttention,
    StepEmbedding,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Condition network
class Condition(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dec = nn.Linear(config.seq_len, config.pred_len)

    def forward(self, x):
        out = self.dec(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out


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
    def __init__(self, hidden_dim, d_model, pred_len, n_emb, config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(d_model, hidden_dim, n_emb - 1),
            nn.Linear(hidden_dim, d_model),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * d_model, bias=True)
        )
        self.config = config

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
            config.feature_dim, config.hidden_dim, config.n_emb
        )
        self.k_embedder = StepEmbedding(config.hidden_dim, freq_dim=256)

        d_model = config.d_model

        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_depth)])
        self.decoder = Decoder(
            config.hidden_dim,
            d_model,
            config.pred_len,
            config.n_emb,
            config,
        )
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

    def forward(self, x, y, k, cond_info):
        """
        x: (B, context_length, num_feat)
        y: (B, prediction_length, num_feat)
        k: (B,)
        cond_info: (B, context_length, num_feat)
        """
        if self.config.task_name == "classification":
            h = self.input_embedder(x)
        else:
            h = self.input_embedder(y)
            if self.config.task_name == "anomaly_detection":
                h = h[:, -self.config.pred_len :, :]

        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info)
            h = torch.cat([h, cond_info], dim=-1)

        c = self.k_embedder(k)

        for block in self.blocks:
            h = block(h, c)

        out = self.decoder(h, c)

        if self.config.task_name != "classification":
            out = self.act(out)
            if self.config.task_name != "anomaly_detection":
                out = out.permute(0, 2, 1)

        return out
