import torch.multiprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import wandb
from cndiff_utils.utils import denormalize, normalize
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import save_results
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment

torch.multiprocessing.set_sharing_strategy("file_system")
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from utils.tools import get_loader_dims

warnings.filterwarnings("ignore")


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        self.train_data, self.train_loader = self._get_data(flag="TRAIN")
        self.test_data, self.test_loader = self._get_data(flag="TEST")
        self.vali_data, self.vali_loader = self._get_data(flag="TEST")

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
            for i, (batch_x, _) in enumerate(self.vali_loader):
                batch_x = batch_x.float().to(self.device)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                    outputs = self.model(batch_x)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )

                else:
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
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
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

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
                    # print("Using t_phi loss")
                    loss = self.model.get_mu_t_phi_loss(
                        outputs, batch_x, self.model.t, self.model.condition_info
                    )
                else:
                    loss = criterion(outputs, batch_x)

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

        attens_energy = []

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(self.train_loader):
                batch_x = batch_x.float().to(self.device)

                # reconstruction
                if "cndiff" in self.args.model.lower():
                    if self.args.normalize:
                        batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                    outputs = self.model.p_sample_loop(batch_x, batch_x)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )

                else:
                    outputs = self.model(batch_x, None, None, None)

                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            if "cndiff" in self.args.model.lower():
                if self.args.normalize:
                    batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                outputs = self.model.p_sample_loop(batch_x, batch_x)
                if self.args.normalize:
                    outputs = denormalize(outputs, x_mean, x_std, self.args.pred_len)

            else:
                outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )

        metrics = {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "f1": float(f_score),
            "precision": float(precision),
        }
        if self.args.wandb:
            wandb.log(metrics)

        # save_preds(setting, pred, gt)
        save_results(
            f"{self.args.model_id}_anomaly_detection_diffusion", setting, metrics
        )
        return PATH
