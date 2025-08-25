import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.radam import RAdam

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import save_results
from utils.tools import EarlyStopping, cal_accuracy, get_loader_dims

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


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        self.train_data, self.train_loader = self._get_data(flag="TRAIN")
        self.test_data, self.test_loader = self._get_data(flag="TEST")
        self.vali_data, self.vali_loader = self._get_data(flag="TEST")
        self.args.seq_len, self.args.feature_dim = get_loader_dims(self.train_loader)
        self.args.enc_in = self.train_data.feature_df.shape[1]
        self.args.num_class = len(self.train_data.class_names)

        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return RAdam(self.model.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def _select_criterion():
        return nn.CrossEntropyLoss()

    def vali(self, criterion):
        self.model.eval()
        total_loss, preds, trues = [], [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in self.vali_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, _, _ = normalize(self.device, batch_x)
                    outputs = self.model(batch_x)

                else:
                    outputs = self.model(batch_x, padding_mask, None, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = F.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()

            for batch_x, label, padding_mask in self.train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, _, _ = normalize(self.device, batch_x)
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None, None)

                # print(outputs.shape)
                # print(label.shape)
                # print(outputs)
                loss = criterion(outputs, label.long().squeeze())
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                train_loss.append(loss.item())

            print(f"Epoch {epoch + 1} | Time: {time.time() - epoch_time:.1f}s")
            train_loss_avg = np.average(train_loss)
            vali_loss, vali_acc = self.vali(criterion)
            test_loss, test_acc = self.vali(criterion)

            print(
                f"Epoch {epoch + 1} | Train Loss: {train_loss_avg:.4f} | "
                f"Vali Loss: {vali_loss:.4f} | Vali Acc: {vali_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

            if self.args.wandb:
                import wandb

                wandb.log({
                    "train_loss": train_loss_avg,
                    "vali_loss": vali_loss,
                    "vali_accuracy": vali_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                })

            early_stopping(-vali_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag="TEST")
        PATH = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(PATH))

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, _, _ = normalize(self.device, batch_x)
                    outputs = self.model.p_sample_loop(batch_x)
                    outputs = self.model.classifier(outputs.permute(0, 2, 1))
                else:
                    outputs = self.model(batch_x, padding_mask, None, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = F.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        metrics = {
            "accuracy": accuracy,
            "parameters": getattr(self.model, "parameter_dict", None),
        }

        save_results(
            "classification",
            setting,
            metrics,
            self.args.sweep,
        )
        return PATH
