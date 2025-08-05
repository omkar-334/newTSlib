from copy import deepcopy

import torch
import torch.nn as nn

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


# This is the original Decoder class, which will now serve as our class-specific head.
class CDecoder(nn.Module):
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

        # Initialize weights
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x


# NEW CLASS: This is the main architectural change for hard parameter sharing.
class CDenoiser(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_class = config.num_class

        # --- Shared Backbone ---
        self.input_embedder = DataEmbedding(
            config.pred_len, config.hidden_dim, config.n_emb
        )
        self.k_embedder = StepEmbedding(config.hidden_dim, freq_dim=256)
        self.blocks = nn.ModuleList()

        if config.use_cond:
            self.cond_embedder = DataEmbedding(
                config.pred_len, config.hidden_dim, config.n_emb
            )
        # --- End Shared Backbone ---

        # --- Class-Specific Heads ---
        self.class_decoders = nn.ModuleList([
            deepcopy(CDecoder(config)) for _ in range(self.num_class)
        ])

        # --- End Class-Specific Heads ---

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize weights for DiTBlock modulations
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, y, k, cond_info, class_idx):
        """
        y: (B, prediction_length, num_feat) - Noisy input
        k: (B,) - Timestep
        cond_info: (B, context_length, num_feat) - Condition info
        class_idx: (B, 1) or (B,) - Class label for each sample in the batch
        """
        # --- 1. Shared Backbone Forward Pass ---
        h = self.input_embedder(y.permute(0, 2, 1))

        if self.config.use_cond:
            # Embed condition and concatenate
            cond_emb = self.cond_embedder(cond_info.permute(0, 2, 1))
            h = torch.cat([h, cond_emb], dim=-1)

        c = self.k_embedder(k)

        for block in self.blocks:
            h = block(h, c)
        # `h` is now the shared latent representation

        # --- 2. Class-Specific Decoder Forward Pass ---
        # During training, we need to route each sample to its correct decoder
        if self.training:
            output = torch.zeros_like(y)  # Initialize output tensor

            # Iterate through each class present in the batch
            for i in range(self.num_class):
                # Create a mask for samples belonging to the current class
                mask = class_idx.squeeze() == i
                if mask.any():
                    # Select the specific decoder for this class
                    decoder = self.class_decoders[i]

                    # Apply the decoder to the relevant samples' latent representations
                    decoded_h = decoder(h[mask], c[mask])

                    # Place the results back into the correct positions in the output tensor
                    output[mask] = decoded_h.permute(0, 2, 1)
            return output

        # During inference, class_idx is a single integer, so we use one decoder
        decoder = self.class_decoders[class_idx]
        out = decoder(h, c).permute(0, 2, 1)
        return out
