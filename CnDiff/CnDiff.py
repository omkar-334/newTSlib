from types import SimpleNamespace

import torch
import torch.nn as nn

from CnDiff.utils import (
    Condition,
    Denoiser,
    KanTphi,
    Tphi,
    TransformerCondition,
    extract,
    get_gammas,
    # invalid,
    make_beta_schedule,
)


class Model(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        self.config = config
        self.device = self.config.device
        config.d_model = config.hidden_dim * (2 if config.use_cond else 1)
        self.num_timesteps = config.timesteps
        # betas and alphas for diffusion
        betas = make_beta_schedule(
            schedule=self.config.beta_schedule,
            num_timesteps=self.config.timesteps,
            start=self.config.beta_start,
            end=self.config.beta_end,
        )

        betas = betas.float().to(self.device)
        alphas = 1.0 - betas
        self.alphas = alphas
        alphas_cumprod = alphas.to("cpu").cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if self.config.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= (
                0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            )
        if self.config.use_tphi == 1:
            self.t_phi = Tphi(config)
            # exit()
        elif self.config.use_tphi == 2:
            self.t_phi = KanTphi(config)

        # model initialisation for condition network
        self.diffusion_model = Denoiser(config)
        if self.config.use_cond == 1:
            self.condition_model = Condition(config)
        elif self.config.use_cond == 2:
            self.condition_model = TransformerCondition(config)

        if self.config.task_name == "classification":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # reduces sequence dimension
                nn.Flatten(),  # (batch, hidden_dim)
                nn.Linear(config.feature_dim, config.num_class),
            )
        self.parameter_dict = {
            "diffusion_model": sum(
                p.numel() for p in self.diffusion_model.parameters()
            ),
            "condition_model": sum(p.numel() for p in self.condition_model.parameters())
            if self.config.use_cond
            else 0,
            "t_phi": sum(p.numel() for p in self.t_phi.parameters())
            if hasattr(self, "t_phi")
            else 0,
            "total": sum(p.numel() for p in self.parameters()),
        }
        if hasattr(self, "classifier"):
            self.parameter_dict["classifier"] = sum(
                p.numel() for p in self.classifier.parameters()
            )
        print(self.parameter_dict)

    def q_sample(self, batch_y, t):
        """
        Forward process for conditional and learnable mean
        """
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        noise = torch.randn_like(batch_y)

        batch_y_trans = (
            self.t_phi(t=t, batch_y=batch_y) if self.config.use_tphi else batch_y
        )

        y_t = sqrt_alpha_bar_t * batch_y_trans + sqrt_one_minus_alpha_bar_t * noise

        if self.config.use_cond:
            y_t = y_t + (1 - sqrt_alpha_bar_t) * self.condition_info
        return y_t, noise

    def classification(self, x, t):
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        dec_out = self.classifier(dec_out.permute(0, 2, 1))
        # print(dec_out)
        # print(dec_out.shape)

        # print("classification over")
        return dec_out

    def anomaly_detection(self, x, t):
        y_t_batch, _ = self.q_sample(x, t)
        dec_out = self.diffusion_model(y_t_batch, t, self.condition_info)
        return dec_out

    def forward(self, x, original_x=None, y=None, mask=None):
        self.condition_info = self.condition_model(x) if self.config.use_cond else None

        n = x.size(0)
        t = torch.randint(
            low=1,
            high=self.config.timesteps,
            size=(n // 2 + 1,),
        ).to(self.device)
        self.t = t = torch.cat([t, self.config.timesteps - t], dim=0)[:n]

        if "forecast" in self.config.task_name:
            return self.forecast(y, t, forward=True)
        if self.config.task_name == "imputation":
            return self.imputation(original_x, mask, t)
        if self.config.task_name == "anomaly_detection":
            return self.anomaly_detection(x, t)
        if self.config.task_name == "classification":
            return self.classification(x, t)
        return None

    def imputation(self, original_x, mask, t):
        x_t_noisy, _ = self.q_sample(original_x, t)
        # Start from clean ground truth
        x_t = original_x.clone()

        # Replace missing positions with their noised version
        x_t[mask == 0] = x_t_noisy[mask == 0]

        pred_noise = self.diffusion_model(x_t, t, self.condition_info)
        return pred_noise

    def p_sample_loop(self, x, batch_y=None):
        """
        Inference for diffusion model
        """
        time_param = batch_y if "forecast" in self.config.task_name else x

        t = (
            torch.tensor([self.num_timesteps - 1])
            .repeat(time_param.shape[0])
            .to(self.device)
        )
        y_t = torch.randn_like(time_param)

        if self.config.use_cond:
            self.condition_info = self.condition_model(x)
            y_t = self.condition_info + y_t
        else:
            self.condition_info = None

        for t in reversed(range(1, self.num_timesteps)):
            y_t = self.p_sample(y_t, t)

        z = self.p_sample_t_1to0(y_t)
        return z

    def p_sample(self, y_t, t):
        t = torch.tensor([t]).to(self.device)

        sqrt_alpha_bar_t, gamma_0, gamma_1, gamma_2, beta_t_hat = get_gammas(
            self.alphas,
            self.one_minus_alphas_bar_sqrt,
            t,
            y_t,
        )
        y_0_reparam = self.forecast(y_t, t).to(self.device).detach()

        if self.config.use_tphi:
            z = torch.randn_like(y_0_reparam)
            t1 = ((gamma_1 * sqrt_alpha_bar_t) + gamma_0) * (
                self.t_phi(batch_y=y_0_reparam, t=t - 1)
            )
            t2 = (gamma_1 * sqrt_alpha_bar_t) * (self.t_phi(batch_y=y_0_reparam, t=t))

            y_t_m_1_hat = (gamma_1 * y_t) - (t2 - t1)

        else:
            z = torch.randn_like(y_t)
            y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t

        if self.config.use_cond:
            y_t_m_1_hat = y_t_m_1_hat + gamma_2 * self.condition_info

        y_t_m_1 = y_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(
            self.device
        ) * z.to(self.device)

        return y_t_m_1

    def p_sample_t_1to0(self, y_t):
        t = torch.tensor([0]).to(self.device)

        y_0_reparam = self.forecast(y_t, t).to(self.device).detach()

        y_t_m_1 = y_0_reparam.to(self.device)

        return y_t_m_1

    def forecast(self, y, t, forward=False):
        if forward:
            y, _ = self.q_sample(y, t)
        dec_out = self.diffusion_model(y, t, self.condition_info)
        return dec_out

    def get_prior(self, batch_y, cond_info=None):
        """
        Prior loss term in transformed forward process
        """
        T = (
            torch.tensor([self.num_timesteps - 1])
            .repeat(batch_y.shape[0])
            .to(self.device)
        )
        batch_y_mean = self.t_phi(t=T, batch_y=batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, T, batch_y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

        u = sqrt_alpha_bar_t * batch_y_mean

        if self.config.use_cond:
            u = u - (sqrt_alpha_bar_t) * cond_info

        return (1 / 2) * (torch.mean((u) ** 2, dim=(1, 2)))

    def get_mu_t_phi_loss(self, pred_noise, batch_y, t, condition_info=None, mask=None):
        gamma_0, gamma_1, gamma_2, sqrt_alpha_bar_t, beta_t_hat = get_gammas(
            self.alphas,
            self.one_minus_alphas_bar_sqrt,
            t,
            pred_noise,
        )

        term_1 = (gamma_1 * sqrt_alpha_bar_t) * (
            self.t_phi(batch_y=pred_noise, t=t) - self.t_phi(batch_y=batch_y, t=t)
        )
        term_2 = ((gamma_1 * sqrt_alpha_bar_t) + gamma_0) * (
            self.t_phi(batch_y=batch_y, t=t - 1)
            - self.t_phi(batch_y=pred_noise, t=t - 1)
        )

        diff_term = (torch.mean((term_1 + term_2) ** 2, dim=(1, 2), keepdim=True)) / (
            2 * beta_t_hat + 1e-6
        )

        prior_term = self.get_prior(batch_y=batch_y, cond_info=condition_info)

        if mask is not None:
            outputs = torch.where(mask.bool(), batch_y, pred_noise)
        else:
            outputs = pred_noise
        recon_term = torch.mean(
            (outputs.float() - batch_y.float()) ** 2, dim=(1, 2), keepdim=True
        )

        # for term_name, term in [
        #     ("diff_term", diff_term),
        #     ("prior_term", prior_term),
        #     ("recon_term", recon_term),
        # ]:
        #     if invalid(term_name, term):
        #         print("pred_noise:", pred_noise)
        #         exit()

        return torch.mean((diff_term) + (prior_term) + recon_term)
