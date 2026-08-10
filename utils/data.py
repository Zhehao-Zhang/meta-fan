"""Data interfaces for the runnable, artifact-free Meta-FAN pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DomainWindows = Tuple[np.ndarray, np.ndarray, np.ndarray]


def load_windowed_npz(path: str | Path) -> Tuple[Dict[str, DomainWindows], List[str]]:
    """Load authorized, already-windowed arrays from an external NPZ file.

    Required arrays:
        X: ``(N, T, F)`` float-like input windows.
        y: ``(N,)`` energy targets.
        timestamps: ``(N,)`` datetime strings or NumPy datetimes.
        domains: ``(N,)`` string domain labels.

    Optional array:
        feature_names: ``(F,)`` string feature names.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"X", "y", "timestamps", "domains"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")

        x_values = np.asarray(archive["X"], dtype=np.float32)
        y_values = np.asarray(archive["y"], dtype=np.float32).reshape(-1)
        timestamps = np.asarray(archive["timestamps"]).astype("datetime64[ns]")
        labels = np.asarray(archive["domains"]).astype(str).reshape(-1)
        feature_names = (
            np.asarray(archive["feature_names"]).astype(str).tolist()
            if "feature_names" in archive.files
            else [f"feature_{index}" for index in range(x_values.shape[-1])]
        )

    if x_values.ndim != 3:
        raise ValueError(f"X must have shape (N, T, F); received {x_values.shape}.")
    sample_count = len(x_values)
    if not (sample_count == len(y_values) == len(timestamps) == len(labels)):
        raise ValueError("X, y, timestamps, and domains must have equal length.")
    if len(feature_names) != x_values.shape[-1]:
        raise ValueError("feature_names length must equal X.shape[-1].")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("X and y must contain only finite values.")

    grouped: Dict[str, DomainWindows] = {}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        order = np.argsort(timestamps[indices])
        selected = indices[order]
        grouped[label] = (
            x_values[selected],
            y_values[selected],
            timestamps[selected],
        )

    if len(grouped) < 3:
        raise ValueError("Meta-FAN requires at least two source domains and one target domain.")
    return grouped, feature_names


def make_synthetic_domains(
    domain_count: int = 3,
    samples_per_domain: int = 64,
    sequence_length: int = 12,
    feature_count: int = 6,
    seed: int = 7,
) -> Tuple[Dict[str, DomainWindows], List[str]]:
    """Create an in-memory dataset for an end-to-end smoke run."""
    if domain_count < 3:
        raise ValueError("Synthetic mode requires at least three domains.")
    if samples_per_domain < 8:
        raise ValueError("Each synthetic domain must contain at least eight samples.")
    if feature_count < 2:
        raise ValueError("Synthetic mode requires at least two features.")

    rng = np.random.default_rng(seed)
    domains: Dict[str, DomainWindows] = {}
    feature_names = [f"feature_{index}" for index in range(feature_count - 1)]
    feature_names.append("energy_last")

    for domain_index in range(domain_count):
        x_values = rng.normal(
            0.0,
            1.0,
            size=(samples_per_domain, sequence_length, feature_count),
        ).astype(np.float32)
        phase = np.arange(samples_per_domain * sequence_length).reshape(
            samples_per_domain,
            sequence_length,
        )
        daily_signal = np.sin(2.0 * np.pi * phase / 96.0)
        x_values[:, :, 0] += daily_signal.astype(np.float32)
        x_values += np.float32(domain_index * 0.12)

        target = (
            0.50 * x_values[:, -1, 0]
            + 0.25 * x_values[:, -1, 1]
            + 0.15 * x_values[:, :, -1].mean(axis=1)
            + rng.normal(0.0, 0.05, size=samples_per_domain)
        ).astype(np.float32)
        target = target - target.min() + np.float32(0.05)

        start = np.datetime64("2026-01-01T00:00") + np.timedelta64(
            domain_index * 30,
            "D",
        )
        timestamps = start + np.arange(samples_per_domain) * np.timedelta64(15, "m")
        domains[f"domain_{domain_index}"] = (x_values, target, timestamps)

    return domains, feature_names


def chronological_split(
    windows: DomainWindows,
    train_ratio: float,
) -> Tuple[DomainWindows, DomainWindows]:
    """Split one domain chronologically into train and validation subsets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")
    x_values, y_values, timestamps = windows
    split = min(len(x_values) - 1, max(1, int(len(x_values) * train_ratio)))
    return (
        (x_values[:split], y_values[:split], timestamps[:split]),
        (x_values[split:], y_values[split:], timestamps[split:]),
    )


def concatenate_domains(domains: Dict[str, DomainWindows]) -> DomainWindows:
    """Concatenate multiple domain tuples along the sample axis."""
    return (
        np.concatenate([values[0] for values in domains.values()], axis=0),
        np.concatenate([values[1] for values in domains.values()], axis=0),
        np.concatenate([values[2] for values in domains.values()], axis=0),
    )


def fit_standardizer(x_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one mean and standard deviation per input feature."""
    mean = x_values.mean(axis=(0, 1), keepdims=True)
    std = x_values.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(
    x_values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply source-domain statistics to input windows."""
    return ((x_values - mean) / std).astype(np.float32)


def split_target(
    windows: DomainWindows,
    support_samples: int,
) -> Tuple[DomainWindows, DomainWindows]:
    """Split the target chronologically into support and test subsets."""
    x_values, y_values, timestamps = windows
    if support_samples < 2:
        raise ValueError("support_samples must be at least two.")
    if support_samples >= len(x_values):
        raise ValueError(
            f"Target has {len(x_values)} samples; support_samples must be smaller."
        )
    return (
        (
            x_values[:support_samples],
            y_values[:support_samples],
            timestamps[:support_samples],
        ),
        (
            x_values[support_samples:],
            y_values[support_samples:],
            timestamps[support_samples:],
        ),
    )
