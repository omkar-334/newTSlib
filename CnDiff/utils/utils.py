import torch


def extract(tensor, t, x):
    shape = x.shape
    out = torch.gather(tensor, 0, t.to(tensor.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def get_gammas(alphas, one_minus_alphas_bar_sqrt, t, y_t, squeeze=False):
    alpha_t = extract(alphas, t, y_t)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y_t)

    if squeeze:
        alpha_t = alpha_t.squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t_m_1 = (
            (sqrt_one_minus_alpha_bar_t_m_1).squeeze(1).squeeze(1)
        )

    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

    gamma_0 = (
        (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_1 = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        * (alpha_t.sqrt())
        / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square()
    )

    beta_t_hat = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        / (sqrt_one_minus_alpha_bar_t.square())
        * (1 - alpha_t)
    )
    return sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modify_gammas(sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat):
    return (
        sqrt_alpha_bar_t.unsqueeze(1).unsqueeze(2),
        gamma_0.unsqueeze(1).unsqueeze(2),
        gamma_1.unsqueeze(1).unsqueeze(2),
        gamma_2.unsqueeze(1).unsqueeze(2),
        beta_t_hat.unsqueeze(1).unsqueeze(2),
    )


# preprocessing in dlinear
def DL_normalize(device, x, y0=None):
    x = x.to(device)
    # x_mean = x[:, -1:, :].to(device)
    x_mean = x.mean(dim=1, keepdim=True)

    x_std = torch.ones_like(x_mean).to(device)
    x_norm = (x - x_mean) / x_std

    if y0 is not None:
        y0 = y0.to(device)
        y0_norm = (y0 - x_mean) / x_std
    else:
        y0_norm = None
    return x_norm, y0_norm, x_mean, x_std


def DL_denormalize(y0, mean, std, pred_len):
    B = mean.shape[0]
    n_samples = y0.shape[0] // B
    std = torch.repeat_interleave(std, n_samples, dim=0).repeat(1, pred_len, 1)
    mean = torch.repeat_interleave(mean, n_samples, dim=0).repeat(1, pred_len, 1)
    y0 = y0 * std + mean
    return y0


def NST_normalize(device, inp, mask=None):
    inp = inp.to(device)
    means = torch.sum(inp, dim=1) / torch.sum(mask == 1, dim=1)
    means = means.unsqueeze(1).detach()
    x_enc = inp.sub(means)
    x_enc = x_enc.masked_fill(mask == 0, 0)
    stdev = torch.sqrt(
        torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
    )
    stdev = stdev.unsqueeze(1).detach()
    inp = x_enc.div(stdev)
    return inp, means, stdev


def NST_denormalize(outputs, means, stdev, pred_len):
    dec_out = outputs.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
    outputs = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
    return outputs


def invalid(name, tensor):
    if torch.isnan(tensor).any():
        print(f"{name} is NaN")
        print(tensor)
        return True
    if torch.isinf(tensor).any():
        print(f"{name} is Inf")
        print(tensor)
        return True

    return False
