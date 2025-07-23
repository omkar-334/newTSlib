import torch
import torch.nn as nn

from Other_Models.SSSD.imputers.SSSDS4Imputer import SSSDS4Imputer
from Other_Models.SSSD.imputers.util import (
    calc_diffusion_hyperparams,
    sampling,
)


class Model(nn.Module):
    """
    SSSD-S4 Model for time series imputation and anomaly detection
    Paper: https://arxiv.org/abs/2203.10825
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "imputation")
        self.seq_len = getattr(configs, "seq_len", 100)
        self.pred_len = getattr(configs, "pred_len", 100)
        self.enc_in = getattr(configs, "enc_in", 14)
        self.device = getattr(configs, "device", "cuda")

        # SSSD specific parameters (defaults)
        self.diffusion_config = {
            "T": getattr(configs, "diffusion_T", 100),
            "beta_0": getattr(configs, "diffusion_beta_0", 0.0001),
            "beta_T": getattr(configs, "diffusion_beta_T", 0.02),
        }
        if not hasattr(configs, "diffusion_T"):
            self.diffusion_config["T"] = 200
        if not hasattr(configs, "diffusion_beta_0"):
            self.diffusion_config["beta_0"] = 0.0001
        if not hasattr(configs, "diffusion_beta_T"):
            self.diffusion_config["beta_T"] = 0.02

        # SSSD-S4 model configuration (defaults)
        self.model_config = {
            "in_channels": self.enc_in,
            "out_channels": self.enc_in,
            "num_res_layers": getattr(configs, "sssd_num_res_layers", 36),
            "res_channels": getattr(configs, "sssd_res_channels", 256),
            "skip_channels": getattr(configs, "sssd_skip_channels", 256),
            "diffusion_step_embed_dim_in": getattr(
                configs, "sssd_diffusion_step_embed_dim_in", 128
            ),
            "diffusion_step_embed_dim_mid": getattr(
                configs, "sssd_diffusion_step_embed_dim_mid", 512
            ),
            "diffusion_step_embed_dim_out": getattr(
                configs, "sssd_diffusion_step_embed_dim_out", 512
            ),
            "s4_lmax": getattr(configs, "sssd_s4_lmax", 100),
            "s4_d_state": getattr(configs, "sssd_s4_d_state", 64),
            "s4_dropout": getattr(configs, "sssd_s4_dropout", 0.0),
            "s4_bidirectional": getattr(configs, "sssd_s4_bidirectional", 1),
            "s4_layernorm": getattr(configs, "sssd_s4_layernorm", 1),
        }
        # Set defaults if not present
        defaults = {
            "num_res_layers": 36,
            "res_channels": 256,
            "skip_channels": 256,
            "diffusion_step_embed_dim_in": 128,
            "diffusion_step_embed_dim_mid": 512,
            "diffusion_step_embed_dim_out": 512,
            "s4_lmax": 100,
            "s4_d_state": 64,
            "s4_dropout": 0.0,
            "s4_bidirectional": 1,
            "s4_layernorm": 1,
        }
        for k, v in defaults.items():
            if not hasattr(configs, f"sssd_{k}"):
                self.model_config[k] = v

        # Calculate diffusion hyperparameters
        self.diffusion_hyperparams = calc_diffusion_hyperparams(**self.diffusion_config)

        # Initialize the SSSD-S4 imputer
        self.sssd_model = SSSDS4Imputer(**self.model_config)

        # For anomaly detection
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # Training state
        self.t = None
        self.condition_info = None

    def _get_mask(self, x, mask_rate=0.25):
        """Generate random mask for imputation"""
        B, T, N = x.shape
        mask = torch.rand((B, T, N)).to(x.device)
        mask[mask <= mask_rate] = 0  # masked
        mask[mask > mask_rate] = 1  # remained
        return mask

    def _prepare_input(self, x, mask=None):
        """Prepare input for SSSD model"""
        if mask is None:
            mask = self._get_mask(x)

        # Apply mask
        inp = x.masked_fill(mask == 0, 0)

        # SSSD expects (batch, channels, seq_len) format
        inp = inp.permute(0, 2, 1)  # (B, N, T)
        x_permuted = x.permute(0, 2, 1)  # (B, N, T)
        mask_permuted = mask.permute(0, 2, 1)  # (B, N, T)

        return inp, x_permuted, mask_permuted

    def _diffusion_step(self, t):
        """Get diffusion step for training"""
        return torch.tensor([t]).to(self.device)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for SSSD model

        Args:
            x_enc: Input sequence (B, T, N)
            x_mark_enc: Encoder time features (optional)
            x_dec: Decoder input (optional)
            x_mark_dec: Decoder time features (optional)
            mask: Mask for missing values (optional)
        """
        if self.task_name == "imputation":
            return self.imputation(x_enc, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        raise ValueError(f"Task {self.task_name} not supported for SSSD model")

    def imputation(self, x_enc, mask=None):
        """Imputation task"""
        inp, x_permuted, mask_permuted = self._prepare_input(x_enc, mask)

        # For training - use forward function
        B, N, T = inp.shape
        t = torch.randint(0, self.diffusion_config["T"], (B, 1)).to(inp.device)

        # Generate noise
        noise = torch.randn_like(inp).to(inp.device)

        # Forward pass through SSSD model
        output = self.sssd_model((noise, inp, mask_permuted, t))

        # Convert back to (B, T, N) format
        output = output.permute(0, 2, 1)
        return output

    def anomaly_detection(self, x_enc):
        """Anomaly detection task"""
        # SSSD expects (batch, channels, seq_len) format
        x_permuted = x_enc.permute(0, 2, 1)  # (B, N, T)

        # For training - use forward function
        B, N, T = x_permuted.shape
        t = torch.randint(0, self.diffusion_config["T"], (B, 1)).to(x_permuted.device)

        # Generate noise
        noise = torch.randn_like(x_permuted).to(x_permuted.device)

        # For anomaly detection, create a dummy mask of all ones (no masking)
        dummy_mask = torch.ones_like(x_permuted).to(x_permuted.device)

        # Forward pass through SSSD model
        output = self.sssd_model((noise, x_permuted, dummy_mask, t))

        # Convert back to (B, T, N) format
        output = output.permute(0, 2, 1)
        return output

    def get_mu_t_phi_loss(self, pred, target, t, condition_info):
        """Custom loss function for SSSD training"""
        # This is a placeholder - the actual loss is handled in the training loop
        # The SSSD model uses its own training loss function
        return nn.MSELoss()(pred, target)

    def p_sample_loop(self, x_enc):
        """Sampling loop for inference"""
        # SSSD expects (batch, channels, seq_len) format
        x_permuted = x_enc.permute(0, 2, 1)  # (B, N, T)

        B, N, T = x_permuted.shape
        noise = torch.randn_like(x_permuted).to(x_permuted.device)

        # Sample from the model
        output = sampling(
            self.sssd_model,
            (B, N, T),
            self.diffusion_hyperparams,
            cond=x_permuted,
            mask=None,
            only_generate_missing=False,
        )

        # Convert back to (B, T, N) format
        output = output.permute(0, 2, 1)
        return output
