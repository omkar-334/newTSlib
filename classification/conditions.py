import torch.nn as nn


def condition(config):
    if config.use_cond == 1:
        return Condition1(config)
    if config.use_cond == 2:
        return Condition2(config)
    raise ValueError("Invalid condition network type specified in config.")


# Condition network
class Condition1(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dec = nn.Linear(config.seq_len, config.pred_len)

    def forward(self, x):
        out = self.dec(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out


# Condition network
class Condition2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                config.feature_dim,
                config.hidden_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=3,
                padding=4,
                dilation=4,
            ),
            nn.ReLU(),
        )
        self.project = nn.Linear(config.hidden_dim, config.feature_dim)

    def forward(self, x):
        out = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, H)
        return self.project(out)
