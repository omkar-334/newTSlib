import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.adam import Adam

# from CnDiff.utils import NST_denormalize as denormalize
# from CnDiff.utils import NST_normalize as normalize
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.ad_plot import plot
from utils.metrics import save_results
from utils.tools import EarlyStopping, adjustment, get_loader_dims

warnings.filterwarnings("ignore")


def normalize(device, x_enc):
    """Batch-wise normalization: zero mean, unit variance."""
    x_enc = x_enc.to(device)
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc.sub(means)
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_enc = x_enc.div(stdev)
    return x_enc, means, stdev


def denormalize(dec_out, means, stdev, pred_len):
    """Inverse normalization using stored means & std."""
    dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
    dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
    return dec_out


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        self.train_data, self.train_loader = self._get_data(flag="TRAIN")
        self.test_data, self.test_loader = self._get_data(flag="TEST")
        self.vali_data, self.vali_loader = self._get_data(flag="TEST")

        self.args.seq_len, self.args.feature_dim = get_loader_dims(self.train_loader)

        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(self.vali_loader):
                batch_x = batch_x.float().to(self.device)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, x_mean, x_std = normalize(self.device, batch_x)

                    outputs = self.model(batch_x)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )
                else:
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                if self.args.tphi_loss:
                    loss = self.model.get_mu_t_phi_loss(
                        outputs,
                        batch_x.to(outputs.dtype),
                        self.model.t,
                        self.model.condition_info,
                    )
                    # print(loss)
                else:
                    loss = criterion(outputs, batch_x.to(outputs.dtype))

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(self.train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                    print("❌ NaN/Inf in input batch, skipping…")
                    continue

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, x_mean, x_std = normalize(self.device, batch_x)

                    outputs = self.model(batch_x)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )
                else:
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                if self.args.tphi_loss:
                    loss = self.model.get_mu_t_phi_loss(
                        outputs,
                        batch_x.to(outputs.dtype),
                        self.model.t,
                        self.model.condition_info,
                    )
                else:
                    loss = criterion(outputs, batch_x.to(outputs.dtype))

                if not torch.isfinite(loss):
                    print(f"❌ Non-finite loss detected: {loss.item()}")
                    # Optional diagnostics
                    print(
                        "outputs stats:",
                        outputs.detach().mean().item(),
                        outputs.detach().std().item(),
                    )
                    print(
                        "targets stats:",
                        batch_x.detach().mean().item(),
                        batch_x.detach().std().item(),
                    )
                    # Skip this batch
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                # (Optional) check grad norm after clipping
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    print(f"❌ Non-finite grad norm: {grad_norm}, skipping step")
                    model_optim.zero_grad(set_to_none=True)
                    continue

                model_optim.step()

                train_loss.append(loss.item())

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if not torch.isfinite(torch.tensor(grad_norm)):
                            print(f"🔥 Non-finite grad in {name}, norm = {grad_norm}")
                            raise SystemExit

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)

            vali_loss = self.vali(criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} "
            )

            if self.args.wandb:
                import wandb

                wandb.log({
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    # "test_loss": test_loss,
                })

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    @torch.inference_mode()
    def test(self, setting, test=0):
        PATH = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(PATH))

        train_energy = []
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction="none")

        # (1) Compute train energy
        print("Calculating train energy")
        for i, (batch_x, _) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.device)

            if "cndiff" in self.args.model.lower():
                if self.args.normalize:
                    batch_x, x_mean, x_std = normalize(self.device, batch_x)
                outputs = self.model.p_sample_loop(batch_x)
                if self.args.normalize:
                    outputs = denormalize(outputs, x_mean, x_std, self.args.pred_len)
            else:
                outputs = self.model(batch_x, None, None, None)

            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            train_energy.append(score.detach().cpu().numpy())
            torch.cuda.empty_cache()

        train_energy = np.concatenate(train_energy, axis=0).reshape(-1)

        # (2) Compute test energy
        test_energy, test_labels, test_data_for_viz = [], [], []
        print("Calculating test energy")
        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x = batch_x.float().to(self.device)
            if self.args.viz:
                test_data_for_viz.append(batch_x.detach().cpu().numpy())

            if "cndiff" in self.args.model.lower():
                if self.args.normalize:
                    batch_x, x_mean, x_std = normalize(self.device, batch_x)
                outputs = self.model.p_sample_loop(batch_x)
                if self.args.normalize:
                    outputs = denormalize(outputs, x_mean, x_std, self.args.pred_len)
            else:
                outputs = self.model(batch_x, None, None, None)

            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            test_energy.append(score.detach().cpu().numpy())
            test_labels.append(batch_y)

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold:", threshold)

        # (3) Evaluate
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        gt = test_labels.astype(int)

        gt, pred = adjustment(gt, pred)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average="binary"
        )

        metrics = {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "f1": float(f_score),
            "precision": float(precision),
            "parameters": getattr(self.model, "parameter_dict", None),
        }
        if self.args.wandb:
            import wandb

            wandb.log(metrics)

        filename = self.args.filename or "tryd"
        save_results(filename, setting, metrics)

        if self.args.viz:
            plot(
                test_data_for_viz,
                pred,
                gt,
                test_energy,
                setting,
                self.args.model,
                threshold,
            )

        return PATH
