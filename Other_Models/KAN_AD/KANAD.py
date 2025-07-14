import torch
import torch.nn as nn


class Model(nn.Module):
    """
    KANAD: Kernel-based Anomaly Detection for Multivariate Time Series
    Updated to match original architecture while supporting multivariate datasets
    
    IMPORTANT: To match original KANAD performance, use these parameters:
    - batch_size: 1024 (not 128)
    - train_epochs: 100 (not 3-10)
    - learning_rate: 0.01 (not 0.0001)
    - order: 2 (default)
    - window: 96 (seq_len)
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # number of features

        if self.task_name == "anomaly_detection":
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # KANAD specific parameters - match original exactly
        self.window = configs.seq_len
        self.order = getattr(configs, "order", 2)  # default order=2
        self.channels = 2 * self.order + 1

        # Register periodic cosine buffer - exactly as in original
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine().unsqueeze(0),  # (1, order, window)
        )

        # KANAD architecture - exactly as in original
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, self.window, padding=0, stride=1, dilation=1)

    def _create_custom_periodic_cosine(self):
        """Create periodic cosine basis functions - exactly as in original"""
        pl = list(range(1, self.order + 1))
        result = torch.empty(self.order, self.window, dtype=torch.float32)
        for i, p in enumerate(pl):
            range_value = torch.arange(self.window, dtype=torch.float32)
            result[i, :] = torch.cos(2 * torch.pi * range_value * p / self.window)
        return result

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for different tasks
        x_enc: input tensor of shape (batch_size, seq_len, enc_in)
        """
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection forward pass - matches original architecture exactly
        x_enc: (batch_size, seq_len, enc_in)
        """
        batch_size, seq_len, n_features = x_enc.shape

        # Process each feature separately using original KANAD logic
        outputs = []
        for i in range(n_features):
            x_feature = x_enc[:, :, i]  # (batch_size, seq_len)

            # Original KANAD processing for single feature
            res = []
            res.append(x_feature.unsqueeze(1))  # (batch_size, 1, seq_len)

            # Create feature maps - exactly as in original
            ff = torch.cat(
                [self.orders.repeat(x_feature.size(0), 1, 1)]
                + [
                    torch.cos(order * x_feature.unsqueeze(1))
                    for order in range(1, self.order + 1)
                ]
                + [x_feature.unsqueeze(1)],
                dim=1,
            )  # (batch_size, channels, seq_len)

            res.append(ff)

            # Original KANAD network - exactly as in original
            ff = self.init_conv(ff)
            ff = self.bn1(ff)
            ff = self.act(ff)
            ff = self.inner_conv(ff) + res.pop()
            ff = self.bn2(ff)
            ff = self.act(ff)
            ff = self.out_conv(ff) + res.pop()
            ff = self.bn3(ff)
            ff = self.act(ff)
            ff = self.final_conv(ff)

            outputs.append(ff.squeeze(1))  # (batch_size, 1)

        # Stack all feature outputs to maintain multivariate compatibility
        output = torch.stack(outputs, dim=-1)  # (batch_size, 1, n_features)
        return output
