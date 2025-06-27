import torch.nn as nn


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
