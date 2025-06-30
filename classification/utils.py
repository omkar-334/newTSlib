import math

import torch
import torch.nn as nn

from cndiff_utils.layers import (
    StepEmbedding,
)


class Tphi(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()

        param1 = (
            config.c_out if config.task_name != "classification" else config.feature_dim
        )
        param2 = config.pred_len

        self.w1 = nn.Parameter(torch.empty(param1, param1))
        self.b1 = nn.Parameter(torch.empty(param1))

        self.w2 = nn.Parameter(torch.empty(param2, param2))
        self.b2 = nn.Parameter(torch.empty(param2))
        self.act = nn.Tanh()
        self.time_emb = StepEmbedding(param1, freq_dim=256)

        self.init_weights(self.w1, self.b1)

    @staticmethod
    def init_weights(weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, batch_y, t):
        t_emb = self.time_emb(t).unsqueeze(1)
        out = batch_y + t_emb
        out = (out.permute(0, 2, 1) @ self.w2.T) + self.b2
        out = out.permute(0, 2, 1)

        out = (out @ self.w1.T) + self.b1
        out = self.act(out)

        # return out + batch_y
        # residual has no effect, all accuracies are same
        return out


def classifier(config):
    """
    Returns a classifier based on the configuration.
    """
    if config.classifier == 1:
        return classifier1(config)
    if config.classifier == 2:
        return classifier2(config)
    raise ValueError(f"Unknown classifier type: {config.classifier}")


def classifier1(config):
    param = config.feature_dim if config.model == "CnDiff" else config.d_model
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(param, config.num_class),
    )


def classifier2(config):
    param = config.pred_len if config.model == "CnDiff" else config.d_model
    return nn.Sequential(
        nn.Conv1d(param, config.hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(config.hidden_dim, config.num_class),
    )


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
