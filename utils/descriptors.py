"""Feature-level transferability descriptors used by Meta-FAN.

``s_i`` measures cross-domain functional stability and ``d_i`` measures the
source-target distribution shift. Together they form the two-value condition
vector consumed by the MetaGenerator for every input feature.
"""

from __future__ import annotations

from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import shap


def _shap_mean_abs(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Return the mean absolute SHAP value of every input feature."""
    sample_count, timestep_count, feature_count = x_values.shape
    x_flat = x_values.reshape(sample_count * timestep_count, feature_count)
    y_flat = np.repeat(y_values, timestep_count)

    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(x_flat, y_flat)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_flat)
    return np.abs(shap_values).mean(axis=0)


def compute_si(
    domain_data: List[Tuple[np.ndarray, np.ndarray]],
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the functional-stability score of every feature.

    Args:
        domain_data: ``(X, y)`` pairs from at least two source domains. Each
            ``X`` has shape ``(N, T, F)`` and each ``y`` has shape ``(N,)``.
        eps: Numerical guard used in the coefficient of variation.

    Returns:
        An array of shape ``(F,)`` with values in ``(0, 1]``.
    """
    if len(domain_data) < 2:
        raise ValueError("At least two source domains are required.")

    importance_by_domain = np.stack(
        [_shap_mean_abs(x_values, y_values) for x_values, y_values in domain_data]
    )
    mean_importance = importance_by_domain.mean(axis=0)
    std_importance = importance_by_domain.std(axis=0)
    coefficient_of_variation = std_importance / (mean_importance + eps)
    return np.exp(-coefficient_of_variation)


def _marginal_shift_per_feature(
    x_source: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    """Compute a normalized mean shift for every feature."""
    eps = 1e-8
    source_flat = x_source.reshape(-1, x_source.shape[-1])
    target_flat = x_target.reshape(-1, x_target.shape[-1])

    source_mean = source_flat.mean(axis=0)
    source_std = source_flat.std(axis=0) + eps
    target_mean = target_flat.mean(axis=0)
    return np.abs(target_mean - source_mean) / source_std


def _diurnal_curve(
    x_values: np.ndarray,
    timestamps: np.ndarray,
    n_bins: int = 96,
) -> np.ndarray:
    """Aggregate the last timestep of each window into time-of-day bins."""
    feature_count = x_values.shape[-1]
    last_timestep = x_values[:, -1, :]

    datetime_values = np.asarray(timestamps, dtype="datetime64[ns]")
    minutes = (
        datetime_values - datetime_values.astype("datetime64[D]")
    ).astype("timedelta64[m]").astype(int)
    bin_indices = np.clip(minutes * n_bins // (24 * 60), 0, n_bins - 1)

    curve = np.full((n_bins, feature_count), np.nan)
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if mask.any():
            curve[bin_index] = last_timestep[mask].mean(axis=0)

    for feature_index in range(feature_count):
        missing = np.isnan(curve[:, feature_index])
        if missing.all():
            curve[:, feature_index] = 0.0
        elif missing.any():
            curve[missing, feature_index] = np.nanmean(curve[:, feature_index])

    return curve


def _dtw_distance_1d(first: np.ndarray, second: np.ndarray) -> float:
    """Compute classic dynamic time-warping distance for two sequences."""
    row_count = len(first)
    column_count = len(second)
    costs = np.full((row_count + 1, column_count + 1), np.inf)
    costs[0, 0] = 0.0

    for row_index in range(1, row_count + 1):
        for column_index in range(1, column_count + 1):
            point_cost = (
                first[row_index - 1] - second[column_index - 1]
            ) ** 2
            costs[row_index, column_index] = point_cost + min(
                costs[row_index - 1, column_index],
                costs[row_index, column_index - 1],
                costs[row_index - 1, column_index - 1],
            )

    return float(np.sqrt(costs[row_count, column_count]))


def _temporal_shift_per_feature(
    source_curve: np.ndarray,
    target_curve: np.ndarray,
) -> np.ndarray:
    """Compute time-warping distance between domain curves per feature."""
    feature_count = source_curve.shape[1]
    return np.asarray(
        [
            _dtw_distance_1d(
                source_curve[:, feature_index],
                target_curve[:, feature_index],
            )
            for feature_index in range(feature_count)
        ]
    )


def compute_di(
    x_source: np.ndarray,
    x_target: np.ndarray,
    t_source: np.ndarray,
    t_target: np.ndarray,
    lambda1: float = 0.5,
    lambda2: float = 0.5,
) -> np.ndarray:
    """Compute the source-target distribution shift of every feature."""
    if x_source.shape[-1] != x_target.shape[-1]:
        raise ValueError("Source and target feature counts must match.")

    marginal_shift = _marginal_shift_per_feature(x_source, x_target)
    source_curve = _diurnal_curve(x_source, t_source)
    target_curve = _diurnal_curve(x_target, t_target)
    temporal_shift = _temporal_shift_per_feature(source_curve, target_curve)

    marginal_normalized = marginal_shift / (marginal_shift.max() + 1e-8)
    temporal_normalized = temporal_shift / (temporal_shift.max() + 1e-8)
    return lambda1 * marginal_normalized + lambda2 * temporal_normalized


def build_descriptors(stability: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Stack feature stability and domain shift into an ``(F, 2)`` array."""
    if stability.shape != shift.shape:
        raise ValueError("Stability and shift arrays must have the same shape.")
    return np.stack((stability, shift), axis=1)
