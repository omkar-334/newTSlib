import torch
import torch.nn as nn

from Other_Models.SSSD.imputers.CSDI import CSDI_base


class Model(nn.Module):
    """
    CSDI Model wrapper for time series imputation and anomaly detection
    Compatible with exp_imputation.py and exp_anomaly_detection.py

    Usage:
    - During training/validation: call forward(x, mask, is_train=1 or 0) to get loss.
    - During inference: call impute(x, mask, n_samples=1) to get imputed values.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "imputation")
        self.seq_len = getattr(configs, "seq_len", 100)
        self.pred_len = getattr(configs, "pred_len", 100)
        self.enc_in = getattr(configs, "enc_in", 14)
        self.device = getattr(configs, "device", "cuda")
        self.mask_rate = getattr(configs, "mask_rate", 0.25)
        self.batch_size = getattr(configs, "batch_size", 16)
        self.epochs = getattr(configs, "train_epochs", 200)
        self.lr = getattr(configs, "learning_rate", 1e-3)
        self.n_layers = getattr(configs, "csdi_layers", 4)
        self.channels = getattr(configs, "csdi_channels", 64)
        self.nheads = getattr(configs, "csdi_nheads", 8)
        self.diff_emb_dim = getattr(configs, "csdi_diff_emb_dim", 128)
        self.beta_start = getattr(configs, "csdi_beta_start", 0.0001)
        self.beta_end = getattr(configs, "csdi_beta_end", 0.5)
        self.num_steps = getattr(configs, "csdi_num_steps", 50)
        self.schedule = getattr(configs, "csdi_schedule", "quad")
        self.is_unconditional = getattr(configs, "csdi_is_unconditional", 0)
        self.timeemb = getattr(configs, "csdi_timeemb", 128)
        self.featureemb = getattr(configs, "csdi_featureemb", 16)
        self.target_strategy = getattr(configs, "csdi_target_strategy", "random")

        # Build config dict for CSDI_base
        self.config_dict = {
            "model": {
                "is_unconditional": self.is_unconditional,
                "timeemb": self.timeemb,
                "featureemb": self.featureemb,
                "target_strategy": self.target_strategy,
            },
            "diffusion": {
                "layers": self.n_layers,
                "channels": self.channels,
                "nheads": self.nheads,
                "diffusion_embedding_dim": self.diff_emb_dim,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_steps": self.num_steps,
                "schedule": self.schedule,
            },
        }
        self.target_dim = int(self.enc_in)
        self.csdi_model = CSDI_base(self.config_dict, self.device, self.target_dim)
        self.csdi_model.to(self.device)

    def _get_mask(self, x, mask_rate=None):
        if mask_rate is None:
            mask_rate = self.mask_rate
        B, T, N = x.shape
        mask = torch.rand((B, T, N), device=x.device)
        mask = (mask > mask_rate).float()
        return mask

    def forward(self, x, mask=None, is_train=1.0, *args, **kwargs):
        """
        Training/validation forward pass. Returns output and loss.
        x: (B, L, K)
        mask: (B, L, K) or None
        is_train: 1 for training, 0 for validation
        """
        if mask is None:
            mask = self._get_mask(x)
        batch = {
            "observed_data": x,  # (B, L, K)
            "observed_mask": mask,
            "timepoints": torch.arange(x.shape[1], device=x.device)
            .unsqueeze(0)
            .repeat(x.shape[0], 1),
            "gt_mask": mask,
            "cut_length": torch.zeros(x.shape[0], device=x.device).long(),
            "hist_mask": mask,
        }
        # Get model output (reconstruction) and loss
        output, loss = self.csdi_model.reconstruct_and_loss(batch, is_train=is_train)
        return output

    def impute(self, x, mask=None, n_samples=1):
        """
        Inference/imputation. Returns imputed values.
        x: (B, T, N)
        mask: (B, T, N) or None
        n_samples: number of imputed samples to generate
        Returns: (B, T, N) (mean of samples)
        """
        if mask is None:
            mask = self._get_mask(x)
        batch = {
            "observed_data": x.permute(0, 2, 1),
            "observed_mask": mask.permute(0, 2, 1),
            "timepoints": torch.arange(x.shape[1], device=x.device)
            .unsqueeze(0)
            .repeat(x.shape[0], 1),
            "gt_mask": mask.permute(0, 2, 1),
            "cut_length": torch.zeros(x.shape[0], device=x.device).long(),
            "hist_mask": mask.permute(0, 2, 1),
        }
        samples, _, _, _, _ = self.csdi_model.evaluate(batch, n_samples)
        # samples: (B, n_samples, N, T)
        # Return mean over samples, permute to (B, T, N)
        output = samples.mean(dim=1).permute(0, 2, 1)
        return output

    def anomaly_detection(self, x):
        # For anomaly detection, reconstruct the input (no masking)
        B, T, N = x.shape
        mask = torch.ones((B, T, N), device=x.device)
        return self.impute(x, mask, n_samples=1)

    def get_mu_t_phi_loss(self, pred, target, t, condition_info):
        return nn.MSELoss()(pred, target)
