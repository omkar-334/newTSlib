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

    def __init__(
        self, hidden_dim, d_model, n_heads, attn_dropout, mlp_ratio=4.0
    ) -> None:
        d_model = 64
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = FullAttention(
            d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = AttnMLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * d_model, bias=True)
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


# Decoder
class Decoder(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_dim, d_model, pred_len, n_emb, config) -> None:
        super().__init__()
        d_model = 64
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(d_model, d_model, n_emb - 1), nn.Linear(d_model, pred_len)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * d_model, bias=True)
        )
        self.config = config

    def forward(self, x, k):
        """
        x: (B, num_feat, d_model)
        k: (B, hidden_dim)
        """
        shift, scale = self.adaLN_modulation(k).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        if self.config.task_name != "classification":
            x = self.mlp(x)
        return x


class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_embedder = DataEmbedding(
            config.feature_dim, config.hidden_dim, config.n_emb
        )
        self.k_embedder = StepEmbedding(config.hidden_dim, freq_dim=256)

        d_model = config.hidden_dim * 2
        self.blocks = nn.ModuleList([
            DiTBlock(
                config.hidden_dim,
                d_model,
                config.n_heads,
                config.attn_dropout,
                config.mlp_ratio,
            )
            for _ in range(config.n_depth)
        ])
        self.decoder = Decoder(
            config.hidden_dim,
            d_model,
            config.pred_len,
            config.n_emb,
            config,
        )
        self.act = nn.Identity()
        self.initialize_weights()
        self.config = config
        if config.use_cond:
            self.cond_embedder = DataEmbedding(
                config.pred_len, config.hidden_dim, config.n_emb
            )

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
        k: (B, )
        """
        # if self.config.task_name == "classification":
        #     h = self.input_embedder(x)
        # else:
        #     h = self.input_embedder(y.permute(0, 2, 1))

        # if self.config.use_cond:
        #     cond_info = self.cond_embedder(cond_info.permute(0, 2, 1))
        #     h = torch.cat([h, cond_info], dim=-1)

        h = self.input_embedder(x)
        c = self.k_embedder(k)

        for block in self.blocks:
            h = block(h, c)

        out = self.decoder(h, c).permute(0, 2, 1)

        if self.config.task_name != "classification":
            out = self.act(out)
        return out
