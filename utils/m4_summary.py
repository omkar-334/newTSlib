# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright 2020 Element AI Inc. All rights reserved.

"""
M4 Summary
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from data_provider.m4 import M4Dataset, M4Meta


def group_values(values, groups, group_name):
    return [v[~np.isnan(v)] for v, g in zip(values, groups) if g == group_name]


def mase(forecast, insample, outsample, frequency):
    forecast = np.asarray(forecast)
    insample = np.asarray(insample)
    outsample = np.asarray(outsample)

    denom = np.mean(np.abs(insample[:-frequency] - insample[frequency:]))
    if denom == 0:
        return np.nan
    return np.mean(np.abs(forecast - outsample)) / denom


def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    denom = np.abs(target)
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, "submission-Naive2.csv")

    def evaluate(self):
        """
        Evaluate forecasts using M4 test dataset.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        grouped_owa = OrderedDict()

        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)

        naive2_forecasts = [v[~np.isnan(v)] for v in naive2_forecasts]

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}
        for group_name in M4Meta.seasonal_patterns:
            file_name = self.file_path + group_name + "_forecast.csv"
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values

            naive2_forecast = group_values(
                naive2_forecasts, self.test_set.groups, group_name
            )
            target = group_values(
                self.test_set.values, self.test_set.groups, group_name
            )
            insample = group_values(
                self.training_set.values, self.test_set.groups, group_name
            )

            frequency = self.training_set.frequencies[
                self.test_set.groups == group_name
            ][0]

            model_mases[group_name] = np.mean([
                mase(
                    forecast=model_forecast[i],
                    insample=insample[i],
                    outsample=target[i],
                    frequency=frequency,
                )
                for i in range(len(model_forecast))
            ])
            naive2_mases[group_name] = np.mean([
                mase(
                    forecast=naive2_forecast[i],
                    insample=insample[i],
                    outsample=target[i],
                    frequency=frequency,
                )
                for i in range(len(naive2_forecast))
            ])
            naive2_concat = (
                np.concatenate(naive2_forecast) if naive2_forecast else np.array([])
            )
            target_concat = np.concatenate(target) if target else np.array([])
            model_forecast_concat = model_forecast.flatten()

            naive2_smapes[group_name] = np.mean(smape_2(naive2_concat, target_concat))
            grouped_smapes[group_name] = np.mean(
                smape_2(model_forecast_concat, target_concat)
            )
            grouped_mapes[group_name] = np.mean(
                mape(model_forecast_concat, target_concat)
            )

        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        for k in grouped_model_mases.keys():
            grouped_owa[k] = (
                grouped_model_mases[k] / grouped_naive2_mases[k]
                + grouped_smapes[k] / grouped_naive2_smapes[k]
            ) / 2

        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        return (
            round_all(grouped_smapes),
            round_all(grouped_owa),
            round_all(grouped_mapes),
            round_all(grouped_model_mases),
        )

    def summarize_groups(self, scores):
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}
        for g in ["Yearly", "Quarterly", "Monthly"]:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ["Weekly", "Daily", "Hourly"]:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score["Others"] = others_score
        scores_summary["Others"] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary["Average"] = average

        return scores_summary
