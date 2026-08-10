"""Episodic meta-training primitives for Meta-FAN.

The module performs no file I/O. Callers provide already-windowed arrays and
decide how, or whether, to persist trained parameters and loss histories.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from model.meta_fan import MetaFANModel, MetaGenerator
from utils.descriptors import compute_di, compute_si

WindowTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


class EpisodeSampler:
    """Sample pseudo-transfer episodes from multiple source domains.

    Each episode selects one domain as the pseudo-target, uses its earlier
    samples as support, and uses its remaining samples as the query set.
    """

    def __init__(
        self,
        domain_windows: Dict[str, WindowTuple],
        support_ratio: float = 0.5,
    ) -> None:
        if len(domain_windows) < 2:
            raise ValueError("At least two domains are required for episode sampling.")
        if not 0.0 < support_ratio < 1.0:
            raise ValueError("support_ratio must be strictly between 0 and 1.")
        if any(len(values[0]) < 2 for values in domain_windows.values()):
            raise ValueError("Every domain must contain at least two windows.")

        self.domain_keys = list(domain_windows)
        self.domain_windows = domain_windows
        self.support_ratio = support_ratio

    def sample(self) -> dict:
        """Return one support/query episode."""
        target_key = random.choice(self.domain_keys)
        source_keys = [key for key in self.domain_keys if key != target_key]

        x_target, y_target, t_target = self.domain_windows[target_key]
        split = min(
            len(x_target) - 1,
            max(1, int(len(x_target) * self.support_ratio)),
        )

        return {
            "target_key": target_key,
            "source_keys": source_keys,
            "x_support": x_target[:split],
            "y_support": y_target[:split],
            "t_support": t_target[:split],
            "x_query": x_target[split:],
            "y_query": y_target[split:],
            "t_query": t_target[split:],
        }


def precompute_si(domain_windows: Dict[str, WindowTuple]) -> np.ndarray:
    """Compute functional-stability scores once across the source domains."""
    domain_data = [(x_values, y_values) for x_values, y_values, _ in domain_windows.values()]
    return compute_si(domain_data)


def maml_train(
    meta_fan_model: MetaFANModel,
    meta_generator: MetaGenerator,
    sampler: EpisodeSampler,
    s_i: np.ndarray,
    device: torch.device,
    criterion,
    num_episodes: int = 3000,
    inner_steps: int = 5,
    inner_lr: float = 1e-2,
    outer_lr: float = 1e-3,
    reg_lambda: float = 0.01,
    first_order: bool = False,
    log_interval: int = 100,
    di_lambda1: float = 0.5,
    di_lambda2: float = 0.5,
    max_support: int = 512,
    max_query: int = 512,
) -> List[float]:
    """Train the MetaGenerator through episodic FAN adaptation.

    The backbone remains fixed during this stage. The inner loop adapts FAN
    ``gamma`` and ``beta`` on the support set; the outer loop updates the
    MetaGenerator from query-set loss.
    """
    meta_fan_model.to(device)
    meta_generator.to(device)
    meta_fan_model.freeze_backbone()

    outer_optimizer = torch.optim.Adam(meta_generator.parameters(), lr=outer_lr)
    s_i_tensor = torch.as_tensor(s_i, dtype=torch.float32, device=device)
    meta_losses: List[float] = []

    # Domain-shift descriptors depend only on the selected pseudo-target, so
    # compute them once in memory instead of repeating them every episode.
    di_by_target = {}
    for target_key in sampler.domain_keys:
        source_keys = [key for key in sampler.domain_keys if key != target_key]
        x_source = np.concatenate(
            [sampler.domain_windows[key][0] for key in source_keys],
            axis=0,
        )
        t_source = np.concatenate(
            [sampler.domain_windows[key][2] for key in source_keys],
            axis=0,
        )
        x_target, _, t_target = sampler.domain_windows[target_key]
        split = min(
            len(x_target) - 1,
            max(1, int(len(x_target) * sampler.support_ratio)),
        )
        d_i = compute_di(
            x_source,
            x_target[:split],
            t_source,
            t_target[:split],
            lambda1=di_lambda1,
            lambda2=di_lambda2,
        )
        di_by_target[target_key] = torch.as_tensor(
            d_i,
            dtype=torch.float32,
            device=device,
        )

    meta_fan_model.train()
    meta_generator.train()

    for episode_index in range(1, num_episodes + 1):
        episode = sampler.sample()
        d_i_tensor = di_by_target[episode["target_key"]]
        descriptors = torch.stack((s_i_tensor, d_i_tensor), dim=1)

        gamma_init, beta_init = meta_generator(descriptors)
        skip_indices = meta_fan_model.fan.skip_indices
        if skip_indices:
            gamma_init = gamma_init.clone()
            beta_init = beta_init.clone()
            for index in skip_indices:
                gamma_init[index] = 1.0
                beta_init[index] = 0.0

        gamma = gamma_init.clone().requires_grad_(True)
        beta = beta_init.clone().requires_grad_(True)

        x_support_np = episode["x_support"]
        y_support_np = episode["y_support"]
        x_query_np = episode["x_query"]
        y_query_np = episode["y_query"]

        if len(x_support_np) > max_support:
            indices = np.random.choice(len(x_support_np), max_support, replace=False)
            x_support_np = x_support_np[indices]
            y_support_np = y_support_np[indices]
        if len(x_query_np) > max_query:
            indices = np.random.choice(len(x_query_np), max_query, replace=False)
            x_query_np = x_query_np[indices]
            y_query_np = y_query_np[indices]

        x_support = torch.as_tensor(x_support_np, dtype=torch.float32, device=device)
        y_support = torch.as_tensor(y_support_np, dtype=torch.float32, device=device)
        x_query = torch.as_tensor(x_query_np, dtype=torch.float32, device=device)
        y_query = torch.as_tensor(y_query_np, dtype=torch.float32, device=device)

        for _ in range(inner_steps):
            support_prediction = meta_fan_model.backbone(gamma * x_support + beta)
            support_loss = criterion(support_prediction, y_support.reshape(-1, 1))
            gamma_grad, beta_grad = torch.autograd.grad(
                support_loss,
                (gamma, beta),
                create_graph=not first_order,
            )

            if skip_indices:
                gradient_mask = torch.ones_like(gamma_grad)
                for index in skip_indices:
                    gradient_mask[index] = 0.0
                gamma_grad = gamma_grad * gradient_mask
                beta_grad = beta_grad * gradient_mask

            gamma = gamma - inner_lr * gamma_grad
            beta = beta - inner_lr * beta_grad

        query_prediction = meta_fan_model.backbone(gamma * x_query + beta)
        query_loss = criterion(query_prediction, y_query.reshape(-1, 1))
        regularization = reg_lambda * (
            (gamma - gamma_init).square().sum()
            + (beta - beta_init).square().sum()
        )
        meta_loss = query_loss + regularization

        outer_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_generator.parameters(), max_norm=1.0)
        outer_optimizer.step()

        meta_losses.append(meta_loss.item())

        if log_interval > 0 and episode_index % log_interval == 0:
            recent_loss = np.mean(meta_losses[-log_interval:])
            print(
                f"Episode {episode_index}/{num_episodes} "
                f"meta_loss={recent_loss:.6f}"
            )

    return meta_losses
