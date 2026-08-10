"""Training and target-adaptation routines without artifact generation."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.meta_fan import MetaFANModel, MetaGenerator
from utils.descriptors import build_descriptors
from utils.loss import nmse_loss


def _loss_on_arrays(
    model: torch.nn.Module,
    x_values: np.ndarray,
    y_values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_values, dtype=torch.float32),
            torch.as_tensor(y_values, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            predictions.append(model(x_batch.to(device)))
            targets.append(y_batch.to(device))
    return float(
        nmse_loss(
            torch.cat(predictions, dim=0),
            torch.cat(targets, dim=0).reshape(-1, 1),
        ).item()
    )


def pretrain_backbone(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int = 5,
    max_grad_norm: float = 1.0,
) -> Dict[str, list]:
    """Pretrain the forecasting backbone and keep the best state in memory."""
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_nmse": [], "validation_nmse": []}
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0

    model.to(device)
    for epoch_index in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = nmse_loss(prediction, y_batch.reshape(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        validation_loss = _loss_on_arrays(
            model,
            x_validation,
            y_validation,
            device,
            batch_size,
        )
        history["train_nmse"].append(train_loss)
        history["validation_nmse"].append(validation_loss)
        print(
            f"Pretrain epoch {epoch_index}/{epochs}: "
            f"train_nmse={train_loss:.6f}, validation_nmse={validation_loss:.6f}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"Pretraining stopped early at epoch {epoch_index}.")
                break

    model.load_state_dict(best_state)
    return history


def initialize_and_adapt_fan(
    model: MetaFANModel,
    meta_generator: MetaGenerator,
    stability: np.ndarray,
    shift: np.ndarray,
    x_support: np.ndarray,
    y_support: np.ndarray,
    device: torch.device,
    inner_steps: int,
    inner_learning_rate: float,
) -> float:
    """Initialize FAN from descriptors and adapt it on target support data."""
    descriptor_tensor = torch.as_tensor(
        build_descriptors(stability, shift),
        dtype=torch.float32,
        device=device,
    )
    meta_generator.eval()
    with torch.no_grad():
        gamma_init, beta_init = meta_generator(descriptor_tensor)

    skip_indices = model.fan.skip_indices
    if skip_indices:
        gamma_init = gamma_init.clone()
        beta_init = beta_init.clone()
        for index in skip_indices:
            gamma_init[index] = 1.0
            beta_init[index] = 0.0

    gamma = gamma_init.detach().clone().requires_grad_(True)
    beta = beta_init.detach().clone().requires_grad_(True)
    x_tensor = torch.as_tensor(x_support, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_support, dtype=torch.float32, device=device)

    final_loss = float("nan")
    for _ in range(inner_steps):
        prediction = model.backbone(gamma * x_tensor + beta)
        loss = nmse_loss(prediction, y_tensor.reshape(-1, 1))
        gamma_grad, beta_grad = torch.autograd.grad(loss, (gamma, beta))

        if skip_indices:
            mask = torch.ones_like(gamma_grad)
            for index in skip_indices:
                mask[index] = 0.0
            gamma_grad = gamma_grad * mask
            beta_grad = beta_grad * mask

        gamma = (gamma - inner_learning_rate * gamma_grad).detach().requires_grad_(True)
        beta = (beta - inner_learning_rate * beta_grad).detach().requires_grad_(True)
        final_loss = float(loss.item())

    model.fan.set_params(gamma, beta)
    return final_loss


def finetune_backbone(
    model: MetaFANModel,
    x_support: np.ndarray,
    y_support: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> list:
    """Fine-tune the backbone while holding the adapted FAN fixed."""
    model.unfreeze_backbone()
    model.freeze_fan()
    optimizer = torch.optim.Adam(model.backbone_parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_support, dtype=torch.float32),
            torch.as_tensor(y_support, dtype=torch.float32),
        ),
        batch_size=min(batch_size, len(x_support)),
        shuffle=True,
    )
    losses = []

    for epoch_index in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = nmse_loss(prediction, y_batch.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)
        print(
            f"Target fine-tune epoch {epoch_index}/{epochs}: "
            f"support_nmse={epoch_loss:.6f}"
        )
    return losses


def predict(
    model: torch.nn.Module,
    x_values: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return predictions without saving them."""
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x_values, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for (x_batch,) in loader:
            predictions.append(model(x_batch.to(device)).cpu())
    return torch.cat(predictions, dim=0).numpy().reshape(-1)


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Compute the three scalar metrics printed by the entry point."""
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    return {
        "nmse": float(nmse_loss(prediction, target)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }
