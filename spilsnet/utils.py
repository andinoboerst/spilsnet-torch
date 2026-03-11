import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple, Any, Optional, Union, Dict


def build_mlp(
    in_size: int,
    hidden_sizes: List[int],
    out_size: int,
    drop_p: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> nn.Sequential:
    """
    Build a Multi-Layer Perceptron (MLP).

    Args:
        in_size (int): Size of the input layer.
        hidden_sizes (List[int]): List of sizes for each hidden layer.
        out_size (int): Size of the output layer.
        drop_p (float): Dropout probability. Defaults to 0.0.
        dtype (torch.dtype): Data type for layers. Defaults to torch.float64.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    curr = in_size
    for h in hidden_sizes:
        layers.append(nn.Linear(curr, h, dtype=dtype))
        layers.append(nn.Tanh())
        if drop_p > 0:
            layers.append(nn.Dropout(drop_p))
        curr = h
    layers.append(nn.Linear(curr, out_size, dtype=dtype))
    return nn.Sequential(*layers)


def scale_data(
    data: np.ndarray,
    split_indices: List[np.ndarray],
    concatenate: bool = False,
    scaler: Any = None,
) -> Tuple[Any, np.ndarray]:
    """
    Scale data using a provided scaler and split into train/val/test sets.

    Args:
        data (np.ndarray): The data to scale.
        split_indices (List[np.ndarray]): List of indices for each split.
        concatenate (bool): Whether to concatenate the simulations in each split. Defaults to False.
        scaler (Any): Scikit-learn style scaler object. Defaults to None.

    Returns:
        Tuple[Any, np.ndarray, ...]: The fitted scaler and the scaled data sets.
    """
    # Initialize the scaler
    if scaler is None:
        scaler = NoScaler()

    # Check if scaler is already fit (Scikit-learn convention)
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(scaler)
        is_fit = True
    except Exception:
        is_fit = False

    if not is_fit:
        # Combine all training simulations into a single NumPy array (for fitting)
        combined_data = np.concatenate(data[split_indices[0]], axis=0)
        scaler.fit(combined_data)

    scaled_X = np.array([scaler.transform(x) for x in data])

    def concat_func(x: np.ndarray) -> np.ndarray:
        if concatenate:
            if len(x) == 0:
                # Return empty array with correct shape for concatenation
                return np.zeros((0, *scaled_X[0].shape[1:]))
            return np.concatenate(x, axis=0)
        return x

    train_data = concat_func(scaled_X[split_indices[0]])

    additional_data = []
    for sims in split_indices[1:]:
        additional_data.append(concat_func(scaled_X[sims]))

    return (scaler, train_data, *additional_data)


class SimulationDataset(Dataset):
    """
    A PyTorch Dataset for simulation data.
    """

    def __init__(self, *args: np.ndarray, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Initialize the dataset.

        Args:
            *args (np.ndarray): Data arrays.
            device (Optional[Union[str, torch.device]]): Device to move tensors to.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data = [torch.tensor(d, dtype=torch.float64).to(device) for d in args]

    def __len__(self) -> int:
        # dataset length should be based on the shortest tensor so that
        # all components can be indexed without error. this fixes
        # out-of-bounds errors when some inputs (e.g. internal states) are
        # shorter than the primary input array.
        return min(d.shape[0] for d in self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return tuple(d[idx] for d in self.data)

    def append(self, *args: np.ndarray) -> None:
        """
        Append new data to the existing dataset.

        Args:
            *args (np.ndarray): New data arrays.
        """
        new_data_tensors = [torch.tensor(d, dtype=torch.float64).to(self.data[i].device) for i, d in enumerate(args)]
        self.data = [torch.cat((self.data[i], new_data_tensors[i])) for i in range(len(self.data))]


class NoScaler:
    """
    A dummy scaler that does nothing.
    """

    def fit(self, X: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


class NoTransformer:
    """
    A dummy transformer that does nothing.
    """

    @staticmethod
    def transform(x: Any) -> Any:
        return x

    @staticmethod
    def inverse_transform(y: Any) -> Any:
        return y


class CubeRootTransformer:
    """
    A transformer that applies a cube root transformation.
    """

    @staticmethod
    def transform(x: np.ndarray, scaling: float = 10.0) -> np.ndarray:
        return np.cbrt(x * scaling)

    @staticmethod
    def inverse_transform(y: np.ndarray, scaling: float = 10.0) -> np.ndarray:
        return (y**3) / scaling


def serialize_scaler(scaler: Any) -> Dict[str, Any]:
    """
    Serialize a Scikit-learn scaler or internal dummy scaler to a JSON-compatible dictionary.

    Args:
        scaler (Any): The scaler object to serialize.

    Returns:
        Dict[str, Any]: Dictionary containing scaler type and parameters.
    """
    if scaler is None or isinstance(scaler, NoScaler):
        return {"type": "NoScaler"}

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if isinstance(scaler, StandardScaler):
        return {
            "type": "StandardScaler",
            "params": {
                "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
                "var": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
                "n_samples_seen": int(scaler.n_samples_seen_) if hasattr(scaler, "n_samples_seen_") else None,
                "with_mean": scaler.with_mean,
                "with_std": scaler.with_std,
            },
        }
    elif isinstance(scaler, MinMaxScaler):
        return {
            "type": "MinMaxScaler",
            "params": {
                "min": scaler.min_.tolist() if hasattr(scaler, "min_") else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
                "data_min": scaler.data_min_.tolist() if hasattr(scaler, "data_min_") else None,
                "data_max": scaler.data_max_.tolist() if hasattr(scaler, "data_max_") else None,
                "data_range": scaler.data_range_.tolist() if hasattr(scaler, "data_range_") else None,
                "feature_range": list(scaler.feature_range),
            },
        }
    elif isinstance(scaler, CubeRootTransformer) or (isinstance(scaler, type) and issubclass(scaler, CubeRootTransformer)):
        return {"type": "CubeRootTransformer"}
    elif isinstance(scaler, NoTransformer) or (isinstance(scaler, type) and issubclass(scaler, NoTransformer)):
        return {"type": "NoTransformer"}

    raise ValueError(f"Unsupported scaler type for non-pickle serialization: {type(scaler)}")


def deserialize_scaler(data: Dict[str, Any]) -> Any:
    """
    Reconstruct a scaler from a dictionary.

    Args:
        data (Dict[str, Any]): Dictionary containing scaler type and parameters.

    Returns:
        Any: The reconstructed scaler object, or None if no data.
    """
    if data is None:
        return None

    scaler_type = data.get("type")
    if scaler_type == "NoScaler":
        return NoScaler()
    elif scaler_type == "CubeRootTransformer":
        return CubeRootTransformer()
    elif scaler_type == "NoTransformer":
        return NoTransformer()

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if scaler_type == "StandardScaler":
        params = data["params"]
        scaler = StandardScaler(with_mean=params.get("with_mean", True), with_std=params.get("with_std", True))
        if params["mean"] is not None:
            scaler.mean_ = np.array(params["mean"])
            scaler.scale_ = np.array(params["scale"])
            scaler.var_ = np.array(params["var"])
            scaler.n_samples_seen_ = np.int64(params["n_samples_seen"])
        return scaler
    elif scaler_type == "MinMaxScaler":
        params = data["params"]
        scaler = MinMaxScaler(feature_range=tuple(params.get("feature_range", [0, 1])))
        if params["min"] is not None:
            scaler.min_ = np.array(params["min"])
            scaler.scale_ = np.array(params["scale"])
            scaler.data_min_ = np.array(params["data_min"])
            scaler.data_max_ = np.array(params["data_max"])
            scaler.data_range_ = np.array(params["data_range"])
        return scaler

    raise ValueError(f"Unknown scaler type: {scaler_type}")


def spils_loss(
    y_t_pred: torch.Tensor,
    y_t_target: torch.Tensor,
    i_t_pred: torch.Tensor,
    i_t_target: torch.Tensor,
    n_nodes: int,
    dimension: int = 2,
    alpha: float = 0.9,
    beta: float = 0.1,
    gamma: float = 0.0,
) -> torch.Tensor:
    """
    The SPILSNet loss function.

    Args:
        y_t_pred (torch.Tensor): Predicted nodal positions.
        y_t_target (torch.Tensor): Target nodal positions.
        i_t_pred (torch.Tensor): Predicted internal states.
        i_t_target (torch.Tensor): Target internal states.
        alpha (float): Weight for position loss.
        beta (float): Weight for internal state loss.
        gamma (float): Weight for smoothness loss.
        n_nodes (int): Number of nodes.
        dimension (int): Dimension of each node.

    Returns:
        torch.Tensor: The total loss.
    """

    loss_F = nn.functional.mse_loss(y_t_pred, y_t_target)
    loss_i = nn.functional.mse_loss(i_t_pred, i_t_target)

    if gamma > 0:
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
        loss_s = torch.mean(laplacian_of_error**2)
    else:
        loss_s = torch.tensor(0.0, device=y_t_pred.device, dtype=y_t_pred.dtype)

    return (alpha * loss_F) + (beta * loss_i) + (gamma * loss_s)
