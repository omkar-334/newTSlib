import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import Adam

from CnDiff.utils import denormalize, normalize
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import save_results
from utils.tools import EarlyStopping, get_loader_dims


class Exp_ReconClassification(Exp_Basic):
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
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = []
            for batch_x, label, padding_mask in self.train_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                labels_inp = F.one_hot(
                    label.squeeze(1).long(), num_classes=self.args.num_class
                ).float()
                # Expand to [batch, seq_len, num_class] for class conditioning
                labels_inp = labels_inp.unsqueeze(1).expand(-1, batch_x.shape[1], -1)

                if self.args.normalize:
                    batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                # Reconstruction-based: output should reconstruct batch_x
                output = self.model(batch_x, labels_inp, padding_mask)

                if self.args.normalize:
                    output = denormalize(output, x_mean, x_std, self.args.pred_len)

                # Debug prints for output stats
                print('output stats: min', output.min().item(), 'max', output.max().item(), 'mean', output.mean().item(), 'std', output.std().item())

                # MSE loss between output and batch_x
                loss = criterion(output, batch_x)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                epoch_loss.append(loss.item())

            train_loss = np.mean(epoch_loss)
            vali_loss, vali_acc = self.vali(criterion)
            test_loss, test_acc = self.vali(criterion)

            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Vali Loss: {vali_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

            early_stopping(-vali_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth")))
        return self.model

    def vali(self, criterion):
        self.model.eval()
        total_loss, preds, trues = [], [], []
        with torch.no_grad():
            for batch_x, label, padding_mask in self.vali_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                labels_inp = F.one_hot(
                    label.squeeze(1).long(), num_classes=self.args.num_class
                ).float()
                # Expand to [batch, seq_len, num_class] for class conditioning
                labels_inp = labels_inp.unsqueeze(1).expand(-1, batch_x.shape[1], -1)

                if self.args.normalize:
                    batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                # For validation, get reconstruction for each class
                # [B, num_class, seq_len, feature_dim]
                recons = self.model.p_sample_loop_classification(batch_x)
                # Compute MSE for each class
                # [B, num_class]
                mse = ((recons - batch_x.unsqueeze(1)) ** 2).mean(dim=(2, 3))
                # Predicted class is the one with lowest MSE
                pred_class = torch.argmin(mse, dim=1)
                true_class = label.squeeze().long().cpu()
                accuracy = accuracy_score(true_class, pred_class.cpu())
                # For loss, report mean MSE of the true class
                true_mse = mse[torch.arange(batch_x.size(0)), true_class]
                total_loss.append(true_mse.mean().item())

        return np.mean(total_loss), accuracy

    def test(self, setting):
        self.model.load_state_dict(
            torch.load(os.path.join(self.args.checkpoints, setting, "checkpoint.pth"))
        )
        self.model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for batch_x, label, padding_mask in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                labels_inp = F.one_hot(
                    label.squeeze(1).long(), num_classes=self.args.num_class
                ).float()
                # Expand to [batch, seq_len, num_class] for class conditioning
                labels_inp = labels_inp.unsqueeze(1).expand(-1, batch_x.shape[1], -1)
                if self.args.normalize:
                    batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                # [B, num_class, seq_len, feature_dim]
                recons = self.model.p_sample_loop_classification(batch_x)
                mse = ((recons - batch_x.unsqueeze(1)) ** 2).mean(dim=(2, 3))
                pred_class = torch.argmin(mse, dim=1)
                preds.append(pred_class.cpu())
                trues.append(label.squeeze().long().cpu())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        accuracy = accuracy_score(trues, preds)

        save_results(
            "classification_diffusion", setting, {"accuracy": accuracy}, self.args.sweep
        )
        return accuracy
