import torch
import torch.nn as nn

from .layers import (
    AttnMLP,
    DataEmbedding,
    FullAttention,
    StepEmbedding,
)
from .utils import modulate


class Condition(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dec = nn.Linear(config.seq_len, config.pred_len)

    def forward(self, x):
        out = self.dec(x.permute(0, 2, 1)).permute(0, 2, 1)

        return out


class TransformerCondition(nn.Module):
    """
    A Transformer-based conditioning module.

    It processes an input time series to extract a rich contextual representation
    using self-attention, which is then projected to the desired output shape.

    Assumes the following attributes in the config object:
    - config.num_feat: Number of features in the time series.
    - config.d_model: The hidden dimension of the Transformer.
    - config.n_heads: Number of attention heads.
    - config.seq_len: Length of the input sequence.
    - config.pred_len: Length of the output sequence.
    - config.cond_n_depth (optional): Number of layers in the Transformer. Defaults to 2.
    """

    def __init__(self, config) -> None:
        super().__init__()

        d_model = config.d_model

        # --- 1. Input Embedding ---
        # Project the input features into the Transformer's hidden dimension
        self.input_projection = nn.Linear(config.feature_dim, d_model)

        # --- 2. Positional Encoding ---
        # A learnable parameter to provide positional information
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.seq_len, d_model))

        # --- 3. Transformer Encoder ---
        # Standard Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * 4,  # Standard practice
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # Ensures input/output shape is (B, L, D)
        )

        # Stack multiple encoder layers
        num_layers = getattr(
            config, "cond_n_depth", 2
        )  # Default to 2 layers if not specified
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- 4. Output Projection ---
        # Two linear layers to project the output to the required shape
        # First, project the time dimension from seq_len to pred_len
        self.time_projection = nn.Linear(config.seq_len, config.pred_len)
        # Second, project the feature dimension from d_model back to num_feat
        self.feature_projection = nn.Linear(d_model, config.feature_dim)

    def forward(self, x):
        """
        Input x: (B, seq_len, num_feat)
        """
        # 1. Project input to d_model and add positional encoding
        x_emb = self.input_projection(x) + self.positional_encoding

        # 2. Pass through the Transformer encoder
        transformer_out = self.transformer_encoder(x_emb)  # -> (B, seq_len, d_model)

        # 3. Project to the final output shape
        # Permute to apply linear layer on the time dimension
        out_time_proj = self.time_projection(
            transformer_out.permute(0, 2, 1)
        )  # -> (B, d_model, pred_len)

        # Permute back and apply final feature projection
        out = self.feature_projection(
            out_time_proj.permute(0, 2, 1)
        )  # -> (B, pred_len, num_feat)

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
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_depth)])
        self.decoder = Decoder(config)
        self.act = nn.Identity()
        self.config = config

        if config.use_cond:
            self.cond_embedder = DataEmbedding(
                config.pred_len, config.hidden_dim, config.n_emb
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
        h = self.input_embedder(y.permute(0, 2, 1))

        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info.permute(0, 2, 1))
            h = torch.cat([h, cond_info], dim=-1)

        c = self.k_embedder(k)

        for block in self.blocks:
            h = block(h, c)

        out = self.decoder(h, c).permute(0, 2, 1)

        if self.config.task_name != "classification":
            out = self.act(out)
            # if self.config.task_name != "anomaly_detection":
            #     out = out.permute(0, 2, 1)
        # elif hasattr(self.config, "classifier") and self.config.classifier == 1:
        #     out = out.permute(0, 2, 1)

        return out
