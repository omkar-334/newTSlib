import torch.nn as nn

from rational_kat_cu.kat_rational import KAT_Group


class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=dict(type="KAT", act_init=["sigmoid", "sigmoid"]),
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(num_groups=1, mode=act_cfg["act_init"][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(num_groups=1, mode=act_cfg["act_init"][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        # x = self.fc1(x).permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        # x = x.permute(0, 2, 1)
        return x
