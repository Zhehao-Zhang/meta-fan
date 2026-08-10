"""Loss functions used by Meta-FAN."""

import numpy as np
import torch


def nmse_loss(y_pred, y_true):
    """Normalized mean-squared error for NumPy arrays or PyTorch tensors."""
    if isinstance(y_pred, np.ndarray):
        prediction = y_pred.reshape(-1)
        target = np.asarray(y_true).reshape(-1)
        return np.mean((prediction - target) ** 2) / (np.mean(target**2) + 1e-8)
    if isinstance(y_pred, torch.Tensor):
        prediction = y_pred.reshape(-1)
        target = y_true.reshape(-1)
        return torch.mean((prediction - target) ** 2) / (torch.mean(target**2) + 1e-8)
    raise TypeError("Inputs must be NumPy arrays or PyTorch tensors.")
