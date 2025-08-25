from math import sqrt

import torch
import torch.nn as nn
from fasterkan import FasterKANLayer

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


class AttnMLP(nn.Module):
    """
    mlp in model
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=None,
        out_dim=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = nn.Sigmoid()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FullAttention(nn.Module):
    """
    attention in model
    """

    def __init__(self, d_model, n_heads, attn_dropout, d_key=None, d_value=None):
        super().__init__()
        d_key = d_key or (d_model // n_heads)
        d_value = d_value or (d_model // n_heads)
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_key * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_key * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_value * n_heads, bias=False)
        self.WO = nn.Linear(d_value * n_heads, d_model)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value):
        B, l_query, d_query = query.shape
        _, l_key, _ = key.shape

        # Matrices projection
        Q = self.WQ(query).view(B, l_query, self.n_heads, -1)
        K = self.WK(key).view(B, l_key, self.n_heads, -1)
        V = self.WV(value).view(B, l_key, self.n_heads, -1)

        # Scaled dot-product attention
        scale = 1.0 / sqrt(Q.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        agg = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        out = self.WO(agg.view(B, l_query, -1))
        return out


class FullAttention(nn.Module):
    """
    attention in model
    """

    def __init__(self, d_model, n_heads, attn_dropout, d_key=None, d_value=None):
        super().__init__()
        d_key = d_key or (d_model // n_heads)
        d_value = d_value or (d_model // n_heads)
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_key * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_key * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_value * n_heads, bias=False)
        self.WO = nn.Linear(d_value * n_heads, d_model)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value):
        B, l_query, d_query = query.shape
        _, l_key, _ = key.shape

        # Matrices projection
        Q = self.WQ(query).view(B, l_query, self.n_heads, -1)
        K = self.WK(key).view(B, l_key, self.n_heads, -1)
        V = self.WV(value).view(B, l_key, self.n_heads, -1)

        # Scaled dot-product attention
        scale = 1.0 / sqrt(Q.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        agg = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        out = self.WO(agg.view(B, l_query, -1))
        return out


class AttnMLP(nn.Module):
    """
    mlp in model
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=None,
        out_dim=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = nn.Sigmoid()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttnKAN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=None,
        out_dim=None,
        drop=0.0,
        num_grids=8,
        exponent=2,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.kan1 = FasterKANLayer(
            in_dim, hidden_dim, num_grids=num_grids, exponent=exponent
        )
        self.kan2 = FasterKANLayer(
            hidden_dim, out_dim, num_grids=num_grids, exponent=exponent
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, s, d = x.shape
        h = self.norm(x).view(b * s, d)
        h = self.kan2(self.kan1(h)).view(b, s, -1)
        return self.drop(h)
