import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.radam import RAdam

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import save_results
from utils.tools import EarlyStopping, cal_accuracy, get_loader_dims

warnings.filterwarnings("ignore")


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
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # MODIFICATION: The training task is now reconstruction of the added noise.
        # We use Mean Squared Error to measure the difference between predicted and true noise.
        return nn.MSELoss()

    def vali(self, data_loader):
        # MODIFICATION: This entire function is rewritten for the new evaluation logic.
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                # 1. Use the new classification method which returns reconstruction errors for each class.
                reconstruction_errors = self.model.classify_by_reconstruction(batch_x)

                # 2. The predicted class is the one with the minimum reconstruction error.
                predicted_labels = torch.argmin(reconstruction_errors, dim=1)

                # 3. The validation loss is the average of the minimum reconstruction errors.
                min_errors, _ = torch.min(reconstruction_errors, dim=1)
                loss = min_errors.mean()
                total_loss.append(loss.item())

                preds.append(predicted_labels.cpu())
                trues.append(label.squeeze().cpu())

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0).numpy()
        trues = torch.cat(trues, 0).numpy()
        accuracy = cal_accuracy(preds, trues)

        self.model.train()
        return total_loss, accuracy

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

            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # MODIFICATION: The training logic is completely changed.
                # We no longer one-hot encode labels or expect class logits as output.

                # 1. The model's forward pass now takes the raw integer label to select the correct decoder.
                # It returns the predicted noise and the actual noise that was added.
                pred_noise, true_noise = self.model(
                    x=batch_x,
                    original_x=batch_x,
                    padding_mask=padding_mask,
                    label=label,
                )

                # 2. The loss is the MSE between the predicted noise and the true noise.
                loss = criterion(pred_noise, true_noise)
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)

            # MODIFICATION: Call the updated validation logic.
            vali_loss, val_accuracy = self.vali(self.vali_loader)
            test_loss, test_accuracy = self.vali(self.test_loader)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Vali Acc: {val_accuracy:.3f} Test Acc: {test_accuracy:.3f}"
            )

            if self.args.wandb:
                import wandb

                wandb.log({
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "vali_accuracy": val_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                })
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        # MODIFICATION: The test logic is simplified to reuse the new validation function.
        PATH = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(PATH))

        # Use the same evaluation logic for the test set.
        test_loss, test_accuracy = self.vali(self.test_loader)
        print(f"Test Loss: {test_loss:.7f}, Test Accuracy: {test_accuracy:.3f}")

        # MODIFICATION: Correctly access parameter_dict when using DataParallel
        params = (
            self.model.module.parameter_dict
            if self.args.use_multi_gpu
            else self.model.parameter_dict
        )
        metrics = {
            "accuracy": test_accuracy,
            "parameters": params,
        }
        save_results(
            "classification_diffusion",
            setting,
            metrics,
            self.args.sweep,
        )
        return PATH
