import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, save_results
from utils.tools import EarlyStopping, get_loader_dims

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


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        self.train_data, self.train_loader = self._get_data(flag="train")
        self.test_data, self.test_loader = self._get_data(flag="test")
        self.vali_data, self.vali_loader = self._get_data(flag="val")
        _, self.args.feature_dim = get_loader_dims(self.train_loader)
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return Adam(self.model.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def _select_criterion():
        return nn.MSELoss()

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
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                self.train_loader
            ):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, x_mean, x_std = normalize(self.device, batch_x)
                    outputs = self.model(x=batch_x, y=batch_y)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len :, :]
                    ).float()
                    dec_inp = (
                        torch.cat(
                            [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                        )
                        .float()
                        .to(self.device)
                    )
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                if self.args.tphi_loss:
                    # print("Using t_phi loss")
                    loss = self.model.get_mu_t_phi_loss(
                        outputs, batch_y, self.model.t, self.model.condition_info
                    )
                else:
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            if epoch + 1 >= 5:
                train_loss = np.average(train_loss)
                vali_loss = self.vali(criterion)

                print(
                    f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}"
                )
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    @torch.inference_mode()
    def vali(self, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                self.vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, x_mean, x_std = normalize(self.device, batch_x)

                    outputs = self.model(x=batch_x, y=batch_y)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len :, :]
                    ).float()
                    dec_inp = (
                        torch.cat(
                            [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                        )
                        .float()
                        .to(self.device)
                    )
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.tphi_loss:
                    # print("Using t_phi loss")
                    loss = self.model.get_mu_t_phi_loss(
                        outputs, batch_y, self.model.t, self.model.condition_info
                    )
                else:
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    @torch.inference_mode()
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        PATH = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(PATH))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, x_mean, x_std = normalize(self.device, batch_x)
                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.float16
                    ):
                        outputs = self.model.p_sample_loop(batch_x, batch_y)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len :, :]
                    ).float()
                    dec_inp = (
                        torch.cat(
                            [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                        )
                        .float()
                        .to(self.device)
                    )
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])]
                        )
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        argsdict = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "parameters": getattr(self.model, "parameter_dict", None),
        }

        filename = self.args.filename or "LTF"
        save_results(filename, setting, argsdict)
        return PATH
