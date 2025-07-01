import math

import torch
import torch.nn as nn

from cndiff_utils.layers import StepEmbedding
from cndiff_utils.modules import Condition


class Tphi(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()

        param1 = config.feature_dim
        param2 = config.pred_len

        self.time_emb = StepEmbedding(param1, freq_dim=256)
        self.backward_time_emb = StepEmbedding(config.num_class, freq_dim=256)

        self.w1 = nn.Parameter(torch.empty(param1, param1))
        self.b1 = nn.Parameter(torch.empty(param1))

        self.w2 = nn.Parameter(torch.empty(param2, param2))
        self.b2 = nn.Parameter(torch.empty(param2))
        self.act = nn.Tanh()

        self.init_weights(self.w1, self.b1)

        self.backward_mapper = nn.Linear(128, param1, bias=False)

    @staticmethod
    def init_weights(weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, batch_y, t, forward=True):
        print("Tphi forward", batch_y.shape, t.shape)
        t_emb = self.time_emb(t).unsqueeze(1)
        print("t_emb shape:", t_emb.shape)
        out = batch_y + t_emb
        print(out.shape)

        out = (out.permute(0, 2, 1) @ self.w2.T) + self.b2
        out = out.permute(0, 2, 1)

        out = (out @ self.w1.T) + self.b1
        out = self.act(out)
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
        return Condition(config)
    if config.use_cond == 2:
        return Condition2(config)
    raise ValueError("Invalid condition network type specified in config.")


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
