import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from sklearn.base import clone

def scale_data(data, split_indices: list, concatenate=False, scaler=None):
    # Initialize the scaler
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        # Clone to ensure we have a fresh estimator
        try:
            scaler = clone(scaler)
        except Exception:
            pass

    # Combine all training simulations into a single NumPy array (for fitting)
    combined_data = np.concatenate(data[split_indices[0]], axis=0)

    scaler.fit(combined_data)

    scaled_X = np.array([scaler.transform(x) for x in data])

    def concat_func(x):
        if concatenate:
            return np.concatenate(x, axis=0)
        else:
            return x

    train_data = concat_func(scaled_X[split_indices[0]])

    additional_data = []

    for sims in split_indices[1:]:
        additional_data.append(concat_func(scaled_X[sims]))

    return scaler, train_data, *additional_data


class SimulationDataset(Dataset):
    def __init__(self, *args, device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = [torch.tensor(d, dtype=torch.float64).to(device) for d in args]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

    def append(self, *args) -> None:
        """
        Append new data to the existing dataset.
        """
        new_data_tensors = [torch.tensor(d, dtype=torch.float64).to(self.data[i].device) for i, d in enumerate(args)]
        self.data = [torch.cat((self.data[i], new_data_tensors[i])) for i in range(len(self.data))]


def cube_root_transform(x):
    return np.cbrt(x * 10)


def cube_root_inverse_transform(y):
    return y ** 3 / 10

class MinMaxScalerUnscaler:
    def __init__(self, min_val, scale_val):
        self.min_val = min_val
        self.scale_val = scale_val
    
    def __call__(self, x):
        return (x - self.min_val) / self.scale_val

class StandardScalerUnscaler:
    def __init__(self, mean_val, scale_val):
        self.mean_val = mean_val
        self.scale_val = scale_val
    
    def __call__(self, x):
        return x * self.scale_val + self.mean_val

class IdentityUnscaler:
    def __call__(self, x):
        return x

def get_torch_unscaler(scaler, device):
    """
    Returns a callable that takes a tensor and unscales it using the provided scaler's parameters.
    Supports sklearn MinMaxScaler and StandardScaler.
    """
    import torch
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if isinstance(scaler, MinMaxScaler):
        min_val = torch.tensor(scaler.min_, dtype=torch.float64, device=device)
        scale_val = torch.tensor(scaler.scale_, dtype=torch.float64, device=device)
        return MinMaxScalerUnscaler(min_val, scale_val)
    
    elif isinstance(scaler, StandardScaler):
        mean_val = torch.tensor(scaler.mean_, dtype=torch.float64, device=device)
        scale_val = torch.tensor(scaler.scale_, dtype=torch.float64, device=device)
        return StandardScalerUnscaler(mean_val, scale_val)
    
    else:
        return IdentityUnscaler()

def spils_loss(y_t_pred, y_t_target, i_t_pred, i_t_target, alpha=0.7, beta=0.3, penalty=1.2, delta=0.2, unscale_func=None):
    import torch.nn as nn
    
    # unscale y (needed to identify penalized values)
    if unscale_func is not None:
        unscaled_y_pred = unscale_func(y_t_pred)
        unscaled_y_target = unscale_func(y_t_target)
    else:
        unscaled_y_pred = y_t_pred
        unscaled_y_target = y_t_target

    # penalize error in the cube root domain -> prefer underestimation
    penalized_values = unscaled_y_pred.abs() - unscaled_y_target.abs() > 0

    # apply Huber Loss to unscaled space
    loss_y = nn.functional.huber_loss(unscaled_y_pred, unscaled_y_target, delta=delta, reduction="none")

    # apply penalty
    loss_y[penalized_values] *= penalty
    loss_F = loss_y.mean()

    # internal state loss
    loss_i = nn.MSELoss()(i_t_pred, i_t_target)

    # Weighted sum
    total_loss = alpha * loss_F + beta * loss_i
    return total_loss
