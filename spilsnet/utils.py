import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.base import clone


def build_mlp(in_size, hidden_sizes, out_size, drop_p=0.0):
    layers = []
    curr = in_size
    for h in hidden_sizes:
        layers.append(nn.Linear(curr, h, dtype=torch.float64))
        layers.append(nn.Tanh())
        if drop_p > 0:
            layers.append(nn.Dropout(drop_p))
        curr = h
    layers.append(nn.Linear(curr, out_size, dtype=torch.float64))
    return nn.Sequential(*layers)


def scale_data(data, split_indices: list, concatenate=False, scaler=None):
    # Initialize the scaler
    if scaler is None:
        scaler = NoScaler()
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
    def __init__(self, *args, device: str | torch.device | None = None):
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


class NoScaler:
    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class NoTransformer:
    @staticmethod
    def transform(x):
        return x

    @staticmethod
    def inverse_transform(y):
        return y


class CubeRootTransformer:
    @staticmethod
    def transform(x: float, scaling: float = 10):
        return np.cbrt(x * scaling)

    @staticmethod
    def inverse_transform(y: float, scaling: float = 10):
        return y ** 3 / scaling


def spils_loss(y_t_pred, y_t_target, i_t_pred, i_t_target, alpha=0.9, beta=0.1, gamma=0.0, n_nodes=51, dimension=2):

    loss_F = nn.functional.mse_loss(y_t_pred, y_t_target)

    # 3. Internal state loss
    loss_i = nn.functional.mse_loss(i_t_pred, i_t_target)

    total_elements = y_t_pred.numel()
    batch_size = total_elements // (n_nodes * dimension)

    # Reshape into [Batch, Nodes, Dim]
    pred_3d = y_t_pred.view(batch_size, n_nodes, dimension)
    target_3d = y_t_target.view(batch_size, n_nodes, dimension)

    error_3d = pred_3d - target_3d

    left_neighbors = error_3d[:, 0:-2, :]
    center_nodes = error_3d[:, 1:-1, :]
    right_neighbors = error_3d[:, 2:, :]

    laplacian_of_error = left_neighbors - (2 * center_nodes) + right_neighbors

    loss_s = torch.mean(laplacian_of_error ** 2)

    return (alpha * loss_F) + (beta * loss_i) + (gamma * loss_s)
