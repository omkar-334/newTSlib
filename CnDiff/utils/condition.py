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
    MODIFIED to include a convolutional pre-processing layer.
    """

    def __init__(self, config) -> None:
        super().__init__()

        # This assumes d_model is set correctly in your config
        # e.g., config.d_model = config.hidden_dim * 2
        d_model = config.d_model

        # --- 1. Input Embedding ---
        self.input_projection = nn.Linear(config.feature_dim, d_model)

        # --- 2. Positional Encoding ---
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.seq_len, d_model))

        # --- 3. NEW: Convolutional Pre-processing ---
        # This layer will learn local temporal patterns efficiently.
        self.conv_preprocess = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=7,
            padding="same",  # Keeps sequence length the same
            padding_mode="circular",  # Good for time-series
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_activation = nn.GELU()

        # --- 4. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * config.mlp_ratio,  # Use mlp_ratio from config
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        # Stack multiple encoder layers
        num_layers = getattr(
            config, "cond_n_depth", 2
        )  # Default to 2 layers if not specified
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- 5. Output Projection ---
        self.time_projection = nn.Linear(config.seq_len, config.pred_len)
        self.feature_projection = nn.Linear(d_model, config.feature_dim)

    def forward(self, x):
        """
        Input x: (B, seq_len, num_feat)
        """
        # 1. Project input and add positional encoding
        x_emb = self.input_projection(x) + self.positional_encoding

        # 2. NEW: Apply the convolutional pre-processing
        # Input for Conv1d needs to be (B, Channels, Length)
        x_conv_in = x_emb.permute(0, 2, 1)
        x_conv_out = self.conv_preprocess(x_conv_in)
        # Permute back to (B, Length, Channels)
        x_conv_out = x_conv_out.permute(0, 2, 1)

        # Add as a residual connection after activation and normalize
        x_processed = self.conv_norm(x_emb + self.conv_activation(x_conv_out))

        # 3. Pass the enhanced representation through the Transformer encoder
        transformer_out = self.transformer_encoder(x_processed)

        # 4. Project to the final output shape
        out_time_proj = self.time_projection(transformer_out.permute(0, 2, 1))
        out = self.feature_projection(out_time_proj.permute(0, 2, 1))

        return out

