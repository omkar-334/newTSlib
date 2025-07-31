import torch
import torch.nn as nn


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
