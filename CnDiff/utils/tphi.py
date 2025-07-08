import math

import torch
import torch.nn as nn

from CnDiff.utils import StepEmbedding
from rational_kat_cu.kat_rational import KAT_Group


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


class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        feature_dim,
        pred_len,
        act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
        bias=True,
        drop=0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.act1 = KAT_Group(mode=act_cfg["act_init"][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode=act_cfg["act_init"][1])
        self.fc2 = nn.Linear(pred_len, pred_len, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x).permute(0, 2, 1)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)


class KanTphi(nn.Module):
    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self, config):
        super().__init__()
        print("Using KAN as T_phi network")
        param1 = (
            config.c_out if config.task_name != "classification" else config.feature_dim
        )
        self.model = KAN(config.c_out, config.pred_len)
        self.time_emb = StepEmbedding(param1, freq_dim=256)

    def forward(self, batch_y, t):
        t_emb = self.time_emb(t).unsqueeze(1)
        out = batch_y + t_emb
        out = self.model(out)
        return out
