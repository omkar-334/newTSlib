import torch.multiprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import wandb
from CnDiff.utils import denormalize, normalize
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

from utils.ad_plot import (
    plot_anomaly_scores_with_threshold,
    plot_combined_anomaly_view,
)
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
                elif "sssd" in self.args.model.lower():
                    outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                # loss = criterion(pred, true)
                if self.args.tphi_loss:
                    # print("Using t_phi loss")
                    loss = self.model.get_mu_t_phi_loss(
                        outputs, batch_x, self.model.t, self.model.condition_info
                    )
                else:
                    loss = criterion(outputs, batch_x)

                total_loss.append(loss.item())
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

        train_energy = []

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

                    outputs = self.model(batch_x)
                    if self.args.normalize:
                        outputs = denormalize(
                            outputs, x_mean, x_std, self.args.pred_len
                        )
                elif "sssd" in self.args.model.lower():
                    outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                train_energy.append(score)

        train_energy = np.array(np.concatenate(train_energy, axis=0).reshape(-1))

        # (2) find the threshold
        test_energy = []
        test_labels = []
        test_data_for_viz = []  # Store test data for visualization

        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x = batch_x.float().to(self.device)

            # Store original data for visualization (before any normalization)
            test_data_for_viz.append(batch_x.detach().cpu().numpy())

            # reconstruction
            if "cndiff" in self.args.model.lower():
                if self.args.normalize:
                    batch_x, _, x_mean, x_std = normalize(self.device, batch_x)

                outputs = self.model(batch_x)
                if self.args.normalize:
                    outputs = denormalize(outputs, x_mean, x_std, self.args.pred_len)

            elif "sssd" in self.args.model.lower():
                outputs = self.model(batch_x, None, None, None)
            else:
                outputs = self.model(batch_x, None, None, None)

            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            test_energy.append(score)
            test_labels.append(batch_y)

        test_energy = np.array(np.concatenate(test_energy, axis=0).reshape(-1))

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

        filename = self.args.filename or "AD_viZ"
        save_results(filename, setting, metrics)

        # Enhanced Visualization
        print("Creating enhanced anomaly visualizations...")

        # Prepare test data for visualization
        test_data_combined = np.concatenate(test_data_for_viz, axis=0)

        # Reshape if needed - handle different data shapes
        if test_data_combined.ndim == 3:
            # If shape is (batch, seq_len, features), flatten batch dimension
            original_shape = test_data_combined.shape
            test_data_combined = test_data_combined.reshape(-1, original_shape[-1])

        # Take a sample for visualization if data is too large
        max_viz_points = 2000
        if len(test_data_combined) > max_viz_points:
            step = len(test_data_combined) // max_viz_points
            sample_indices = np.arange(0, len(test_data_combined), step)
            viz_data = test_data_combined[sample_indices]
            viz_pred = (
                pred[sample_indices] if len(pred) == len(test_data_combined) else pred
            )
            viz_gt = gt[sample_indices] if len(gt) == len(test_data_combined) else gt
            viz_scores = (
                test_energy[sample_indices]
                if len(test_energy) == len(test_data_combined)
                else test_energy
            )
        else:
            viz_data = test_data_combined
            viz_pred = pred
            viz_gt = gt
            viz_scores = test_energy

        # Create visualizations directory
        viz_dir = f"./ADplots/{setting}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Plot anomaly scores with threshold
        plot_anomaly_scores_with_threshold(
            anomaly_scores=viz_scores,
            threshold=threshold,
            anomaly_mask=viz_pred,
            title=f"{self.args.model} - Anomaly Scores - {setting}",
            save_path=f"{viz_dir}/anomaly_scores.png",
        )

        # 2. Plot multivariate time series with predicted anomalies
        feature_names = [f"Feature {i + 1}" for i in range(viz_data.shape[1])]

        # plot_multivariate_anomalies(
        #     time_series_data=viz_data,
        #     anomaly_mask=viz_pred,
        #     feature_names=feature_names,
        #     title=f"{self.args.model} - Predicted Anomalies - {setting}",
        #     save_path=f"{viz_dir}/predicted_anomalies.png",
        # )

        # # 3. Plot multivariate time series with ground truth anomalies
        # plot_multivariate_anomalies(
        #     time_series_data=viz_data,
        #     anomaly_mask=viz_gt,
        #     feature_names=feature_names,
        #     title=f"{self.args.model} - Ground Truth Anomalies - {setting}",
        #     save_path=f"{viz_dir}/ground_truth_anomalies.png",
        #     anomaly_color="orange",
        # )

        # # 4. Create comparison plot
        # create_comparison_plot(
        #     viz_data,
        #     viz_gt,
        #     viz_pred,
        #     feature_names,
        #     f"{viz_dir}/comparison.png",
        #     setting,
        # )

        # # 5. Keep the original simple plot for compatibility
        # test_energy_norm = (test_energy - np.min(test_energy)) / (
        #     np.max(test_energy) - np.min(test_energy)
        # )

        # plot_anomaly_trace(
        #     signal=test_energy_norm,
        #     anomaly_mask=pred,
        #     title=f"{self.args.model} - {setting}",
        #     save_path=f"./ADplots/{setting}/anomaly_plot.png",
        # )
        plot_combined_anomaly_view(
            time_series_data=viz_data,
            anomaly_scores=viz_scores,
            threshold=threshold,
            pred_mask=viz_pred,
            true_mask=viz_gt,
            title=f"{self.args.model} - Combined Anomaly View - {setting}",
            save_path=f"{viz_dir}/combined_anomaly_view.png",
            feature_index=0,  # or None for mean signal
        )

        print(f"Enhanced visualizations saved to: {viz_dir}")

        return PATH
