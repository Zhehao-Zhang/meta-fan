"""Run the complete Meta-FAN training and few-shot adaptation pipeline.

The entry point reads only an explicitly supplied windowed NPZ file, or creates
an in-memory synthetic dataset for a smoke run. It never saves datasets,
predictions, weights, figures, caches, or metric files.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from model.lstm import PureLSTM
from model.maml import EpisodeSampler, maml_train, precompute_si
from model.meta_fan import MetaFANModel, MetaGenerator
from utils.data import (
    DomainWindows,
    chronological_split,
    concatenate_domains,
    fit_standardizer,
    load_windowed_npz,
    make_synthetic_domains,
    split_target,
    standardize,
)
from utils.descriptors import compute_di
from utils.loss import nmse_loss
from utils.seed import set_seed
from utils.training import (
    finetune_backbone,
    initialize_and_adapt_fan,
    predict,
    pretrain_backbone,
    regression_metrics,
)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def find_skip_indices(feature_names: List[str], requested: str) -> List[int]:
    """Resolve comma-separated bypass feature names to tensor indices."""
    names = {name.strip() for name in requested.split(",") if name.strip()}
    return [index for index, name in enumerate(feature_names) if name in names]


def prepare_source_splits(
    domains: Dict[str, DomainWindows],
    source_keys: List[str],
    train_ratio: float,
) -> tuple[Dict[str, DomainWindows], Dict[str, DomainWindows]]:
    """Chronologically split every source domain."""
    training = {}
    validation = {}
    for key in source_keys:
        training[key], validation[key] = chronological_split(
            domains[key],
            train_ratio=train_ratio,
        )
    return training, validation


def run_pipeline(
    domains: Dict[str, DomainWindows],
    feature_names: List[str],
    target_domain: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Execute pretraining, descriptor learning, MAML, and target adaptation."""
    if target_domain not in domains:
        raise ValueError(
            f"Unknown target domain {target_domain!r}; available={sorted(domains)}"
        )
    source_keys = [key for key in domains if key != target_domain]
    if len(source_keys) < 2:
        raise ValueError("At least two source domains are required for Meta-FAN.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Source domains: {source_keys}")
    print(f"Target domain: {target_domain}")

    source_train, source_validation = prepare_source_splits(
        domains,
        source_keys,
        train_ratio=args.source_train_ratio,
    )
    x_source_train, y_source_train, t_source_train = concatenate_domains(source_train)
    x_source_validation, y_source_validation, _ = concatenate_domains(source_validation)

    source_mean, source_std = fit_standardizer(x_source_train)
    x_source_train_standardized = standardize(
        x_source_train,
        source_mean,
        source_std,
    )
    x_source_validation_standardized = standardize(
        x_source_validation,
        source_mean,
        source_std,
    )
    source_train_standardized = {
        key: (
            standardize(values[0], source_mean, source_std),
            values[1],
            values[2],
        )
        for key, values in source_train.items()
    }

    feature_count = x_source_train.shape[-1]
    if feature_count != len(feature_names):
        raise ValueError("Feature-name count does not match the input tensor.")
    skip_indices = find_skip_indices(feature_names, args.skip_features)
    skipped = [feature_names[index] for index in skip_indices]
    print(f"Input shape: T={x_source_train.shape[1]}, F={feature_count}")
    print(f"FAN bypass features: {skipped or 'none'}")

    print("\nStage 1/4: pretrain the source forecasting backbone")
    backbone = PureLSTM(
        input_size=feature_count,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    pretrain_backbone(
        model=backbone,
        x_train=x_source_train_standardized,
        y_train=y_source_train,
        x_validation=x_source_validation_standardized,
        y_validation=y_source_validation,
        device=device,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        patience=args.patience,
    )

    print("\nStage 2/4: compute feature stability descriptors")
    stability = precompute_si(source_train)
    for name, score in zip(feature_names, stability):
        print(f"  {name}: stability={score:.4f}")

    print("\nStage 3/4: episodic Meta-FAN training")
    meta_fan_model = MetaFANModel(
        backbone=backbone,
        num_features=feature_count,
        skip_indices=skip_indices,
    ).to(device)
    meta_generator = MetaGenerator(hidden_dim=args.meta_hidden_size).to(device)
    sampler = EpisodeSampler(
        source_train_standardized,
        support_ratio=args.meta_support_ratio,
    )
    maml_train(
        meta_fan_model=meta_fan_model,
        meta_generator=meta_generator,
        sampler=sampler,
        s_i=stability,
        device=device,
        criterion=nmse_loss,
        num_episodes=args.maml_episodes,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_learning_rate,
        outer_lr=args.outer_learning_rate,
        reg_lambda=args.regularization,
        first_order=not args.second_order,
        log_interval=args.log_interval,
        max_support=args.max_meta_support,
        max_query=args.max_meta_query,
    )

    print("\nStage 4/4: initialize, adapt, fine-tune, and evaluate the target")
    target_support, target_test = split_target(
        domains[target_domain],
        support_samples=args.support_samples,
    )
    x_support_raw, y_support, t_support = target_support
    x_test_raw, y_test, _ = target_test

    shift = compute_di(
        x_source=x_source_train,
        x_target=x_support_raw,
        t_source=t_source_train,
        t_target=t_support,
        lambda1=args.shift_marginal_weight,
        lambda2=args.shift_temporal_weight,
    )
    x_support = standardize(x_support_raw, source_mean, source_std)
    x_test = standardize(x_test_raw, source_mean, source_std)

    support_loss = initialize_and_adapt_fan(
        model=meta_fan_model,
        meta_generator=meta_generator,
        stability=stability,
        shift=shift,
        x_support=x_support,
        y_support=y_support,
        device=device,
        inner_steps=args.inner_steps,
        inner_learning_rate=args.inner_learning_rate,
    )
    print(f"FAN support NMSE after adaptation: {support_loss:.6f}")

    finetune_backbone(
        model=meta_fan_model,
        x_support=x_support,
        y_support=y_support,
        device=device,
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        learning_rate=args.finetune_learning_rate,
    )
    prediction = predict(meta_fan_model, x_test, device=device)
    metrics = regression_metrics(prediction, y_test)

    print("\nTarget test metrics")
    print(f"  NMSE: {metrics['nmse']:.6f}")
    print(f"  MAE : {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print("No predictions, weights, caches, plots, or metric files were written.")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complete Meta-FAN training with no generated artifacts.",
    )
    parser.add_argument(
        "--data",
        help="External windowed NPZ. Omit it to run on in-memory synthetic data.",
    )
    parser.add_argument(
        "--target-domain",
        help="Target label in the NPZ; defaults to the last sorted domain.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-features", default="energy_last")
    parser.add_argument("--source-train-ratio", type=float, default=0.8)
    parser.add_argument("--support-samples", type=int, default=16)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-learning-rate", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--meta-hidden-size", type=int, default=16)
    parser.add_argument("--maml-episodes", type=int, default=1000)
    parser.add_argument("--meta-support-ratio", type=float, default=0.5)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--inner-learning-rate", type=float, default=0.05)
    parser.add_argument("--outer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--second-order", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-meta-support", type=int, default=512)
    parser.add_argument("--max-meta-query", type=int, default=512)

    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-batch-size", type=int, default=128)
    parser.add_argument("--shift-marginal-weight", type=float, default=0.5)
    parser.add_argument("--shift-temporal-weight", type=float, default=0.5)

    parser.add_argument("--synthetic-domains", type=int, default=3)
    parser.add_argument("--synthetic-samples", type=int, default=64)
    parser.add_argument("--synthetic-sequence-length", type=int, default=12)
    parser.add_argument("--synthetic-features", type=int, default=6)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny training budgets for a fast end-to-end smoke run.",
    )
    return parser


def apply_quick_mode(args: argparse.Namespace) -> None:
    """Reduce compute while still executing all four stages."""
    if not args.quick:
        return
    args.hidden_size = min(args.hidden_size, 16)
    args.pretrain_epochs = 1
    args.maml_episodes = 2
    args.inner_steps = min(args.inner_steps, 2)
    args.finetune_epochs = 1
    args.patience = 0
    args.log_interval = 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_quick_mode(args)

    if args.data:
        domains, feature_names = load_windowed_npz(args.data)
        source_label = args.data
    else:
        domains, feature_names = make_synthetic_domains(
            domain_count=args.synthetic_domains,
            samples_per_domain=args.synthetic_samples,
            sequence_length=args.synthetic_sequence_length,
            feature_count=args.synthetic_features,
            seed=args.seed,
        )
        source_label = "in-memory synthetic data"

    target_domain = args.target_domain or sorted(domains)[-1]
    print(f"Data source: {source_label}")
    run_pipeline(
        domains=domains,
        feature_names=feature_names,
        target_domain=target_domain,
        args=args,
    )


if __name__ == "__main__":
    main()
