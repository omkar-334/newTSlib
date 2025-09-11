import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from CnDiff.utils import NST_denormalize as denormalize
from CnDiff.utils import NST_normalize as normalize
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, save_results
from utils.tools import EarlyStopping, get_loader_dims

warnings.filterwarnings("ignore")


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.f_dim = -1 if self.args.features == "MS" else 0

    def _build_model(self):
        self.train_data, self.train_loader = self._get_data(flag="train")
        self.test_data, self.test_loader = self._get_data(flag="test")
        self.vali_data, self.vali_loader = self._get_data(flag="test")

        self.args.seq_len, self.args.feature_dim = get_loader_dims(self.train_loader)
        model = self.model_dict[self.args.model].Model(self.args).float()
        model = model.to(self.device)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _prepare_batch(self, batch_x):
        B, T, N = batch_x.shape
        mask = (torch.rand((B, T, N), device=self.device) > self.args.mask_rate).float()
        inp = batch_x * mask
        return inp, mask

    def vali(self, criterion):
        self.model.eval()
        total_loss = []

        for batch_x, _, batch_x_mark, _ in self.vali_loader:
            batch_x = batch_x.float().to(self.device, non_blocking=True)
            batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)

            inp, mask = self._prepare_batch(batch_x)

            if "cndiff" in self.args.model.lower():
                if self.args.normalize:
                    inp, x_mean, x_std = normalize(self.device, inp, mask)

                outputs = self.model(inp, original_x=batch_x, mask=mask)

                if self.args.normalize:
                    outputs = denormalize(outputs, x_mean, x_std, self.pred_len)
            else:
                outputs = self.model(inp, batch_x_mark, None, None, mask)

            outputs = outputs[:, :, self.f_dim :]
            batch_x = batch_x[:, :, self.f_dim :]
            mask = mask[:, :, self.f_dim :]

            if self.args.tphi_loss:
                loss = self.model.get_mu_t_phi_loss(
                    outputs, batch_x, self.model.t, self.model.condition_info, mask
                )
            else:
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])

            total_loss.append(loss.item())
        self.model.train()
        return np.mean(total_loss)

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            epoch_loss = []
            self.model.train()
            start_time = time.time()
            for batch_x, _, batch_x_mark, _ in self.train_loader:
                optimizer.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)

                inp, mask = self._prepare_batch(batch_x)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        inp, x_mean, x_std = normalize(self.device, inp, mask)
                    outputs = self.model(inp, original_x=batch_x, mask=mask)
                    if self.args.normalize:
                        outputs = denormalize(outputs, x_mean, x_std, self.pred_len)
                else:
                    outputs = self.model(inp, batch_x_mark, None, None, mask)

                outputs = outputs[:, :, self.f_dim :]
                batch_x = batch_x[:, :, self.f_dim :]
                mask = mask[:, :, self.f_dim :]

                if self.args.tphi_loss:
                    loss = self.model.get_mu_t_phi_loss(
                        outputs, batch_x, self.model.t, self.model.condition_info, mask
                    )
                else:
                    loss = criterion(outputs[mask == 0], batch_x[mask == 0])

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            train_loss = np.mean(epoch_loss)
            vali_loss = self.vali(criterion)
            print(
                f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"Train Loss: {train_loss:.6f} | Vali Loss: {vali_loss:.6f} | "
                f"Time: {time.time() - start_time:.1f}s"
            )

            if self.args.wandb:
                import wandb

                wandb.log({"train_loss": train_loss, "vali_loss": vali_loss})

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

        return self.model

    @torch.inference_mode()
    def test(self, setting, test=0):
        PATH = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
        if test:
            print("Loading model")
        self._load_checkpoint(PATH)

        preds, trues, masks = [], [], []
        self.model.eval()

        for batch_x, _, batch_x_mark, _ in self.test_loader:
            batch_x = batch_x.float().to(self.device, non_blocking=True)
            batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)

            inp, mask = self._prepare_batch(batch_x)

            if "cndiff" in self.args.model.lower():
                if self.args.normalize:
                    inp, x_mean, x_std = normalize(self.device, inp, mask)
                outputs = self.model.p_sample_loop(inp)
                if self.args.normalize:
                    outputs = denormalize(outputs, x_mean, x_std, self.pred_len)
            else:
                outputs = self.model(inp, batch_x_mark, None, None, mask)

            outputs = outputs[:, :, self.f_dim :]
            batch_x = batch_x[:, :, self.f_dim :]
            mask = mask[:, :, self.f_dim :]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_x.detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        preds, trues = preds[masks == 0], trues[masks == 0]
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        save_results(
            self.args.filename or "imputation",
            setting,
            {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "mspe": float(mspe),
                "parameters": getattr(self.model, "parameter_dict", None),
            },
        )

        return PATH

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
