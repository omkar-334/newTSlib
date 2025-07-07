import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim.adam import Adam

from CnDiff.utils import denormalize, normalize
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, save_results
from utils.tools import EarlyStopping, adjust_learning_rate, get_loader_dims, visual

warnings.filterwarnings("ignore")


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        self.train_data, self.train_loader = self._get_data(flag="train")
        self.test_data, self.test_loader = self._get_data(flag="test")
        self.vali_data, self.vali_loader = self._get_data(flag="test")

        self.args.seq_len, self.args.feature_dim = get_loader_dims(self.train_loader)
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, batch_y_mark) in enumerate(
                self.vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        inp, _, x_mean, x_std = normalize(self.device, inp)

                    outputs = self.model(inp, original_x=batch_x)

                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )

                else:
                    outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                self.train_loader
            ):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        inp, _, x_mean, x_std = normalize(self.device, inp)

                    outputs = self.model(inp, original_x=batch_x)

                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )

                else:
                    outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                if self.args.tphi_loss:
                    # print("Using t_phi loss")
                    loss = self.model.get_mu_t_phi_loss(
                        outputs[mask == 0],
                        batch_x[mask == 0],
                        self.model.t,
                        self.model.condition_info,
                    )
                else:
                    loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(criterion)
            test_loss = self.vali(criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )
            if self.args.wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    # "vali_accuracy": val_accuracy,
                    "test_loss": test_loss,
                    # "test_accuracy": test_accuracy,
                })
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        PATH = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(PATH))

        preds = []
        trues = []
        masks = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                self.test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        inp, _, x_mean, x_std = normalize(self.device, inp)

                    outputs = self.model.p_sample_loop(inp)

                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )

                else:
                    outputs = self.model(inp, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + pred[
                        0, :, -1
                    ] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, setting, i)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        preds, trues = preds[masks == 0], trues[masks == 0]

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        argsdict = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
        }

        # save_preds(setting, preds, trues)
        save_results("impsweep", setting, argsdict)
        return PATH
