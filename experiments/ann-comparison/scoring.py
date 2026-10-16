"""Scoring functions for choosing the best parameter setting of each ANN method.

Every function works on a results frame with one row per (config, method,
setting), like `results` in visualize_results.ipynb. Required columns:
`config`, `algorithm`, `param_value`, `latency_ms_per_query`, `speedup`
and `compression` (both relative to Exact FlatIP in the same config), plus
the accuracy metric (Recall@10 by default).

Two independent scores, highest wins:
- weighted_score: accuracy (mean of Recall and mAP @5/10/20), speed and
  memory in one weighted sum.
- knee_score: the point of diminishing returns on the accuracy vs latency
  curve of each sweep (speed and accuracy only).
"""

import numpy as np
import pandas as pd

ACC_METRIC = "recall_at_10"
# Accuracy term of the weighted score: Recall and mAP at 5 / 10 / 20, equally
# weighted (Precision@K always equals Recall@K here, so it would count recall twice).
ACC_METRICS = [f"{m}_at_{k}" for m in ("recall", "map") for k in (5, 10, 20)]
WEIGHTS = {"accuracy": 1 / 3, "speed": 1 / 3, "memory": 1 / 3}
RATIO_REF = 100   # a 100x speedup / compression scores 1


def log_ratio_score(ratio, ref=RATIO_REF):
    # log(ratio) / log(ref), clipped to [-1, 1]: same as exact search -> 0,
    # `ref` times better -> 1, slower or larger than exact search -> negative.
    return np.clip(np.log(ratio) / np.log(ref), -1, 1)


def weighted_score(df, weights=WEIGHTS, metrics=ACC_METRICS, ref=RATIO_REF):
    # w_acc * accuracy + w_speed * speed score + w_mem * memory score, per row.
    # The accuracy weight is split equally over `metrics` (their mean).
    # Weights are normalised to sum to 1, so the score stays on the same scale.
    total = sum(weights.values())
    return (weights["accuracy"] * df[list(metrics)].mean(axis=1)
            + weights["speed"] * log_ratio_score(df["speedup"], ref)
            + weights["memory"] * log_ratio_score(df["compression"], ref)) / total


def _min_max(values):
    # Scale to 0-1; a constant series maps to 0 instead of dividing by zero.
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else values * 0.0


def knee_score(df, metric=ACC_METRIC):
    # Kneedle-style, within each (config, algorithm) sweep: x = log2(latency)
    # and y = accuracy, both scaled to 0-1; score = y - x, the height above the
    # line from the fastest to the slowest setting. The maximum is the knee:
    # past it, each doubling of latency buys less accuracy than the average.
    groups = df.groupby(["config", "algorithm"])
    x = groups["latency_ms_per_query"].transform(lambda s: _min_max(np.log2(s)))
    y = groups[metric].transform(_min_max)
    return y - x


def best_per_config(df, score):
    # Highest-scoring row of each method in each configuration.
    swept = df[df["param_value"].notna()]
    return swept.loc[swept.groupby(["algorithm", "config"])[score].idxmax()]


def best_overall(df, score, metric=ACC_METRIC):
    # One setting per method: mean over configurations for each setting,
    # then the setting with the highest mean score.
    swept = df[df["param_value"].notna()]
    means = (swept.groupby(["algorithm", "param_value"])
             [[score, metric, "speedup", "compression"]].mean().reset_index())
    return means.loc[means.groupby("algorithm")[score].idxmax()].reset_index(drop=True)
