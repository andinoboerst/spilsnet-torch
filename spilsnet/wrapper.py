import numpy as np
import logging
import json
import random
import os
from functools import partial
from typing import Dict, Any, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from safetensors.torch import save_file, load_file
from spilsnet.models import SPILSNetCore
from spilsnet.utils import (
    scale_data,
    SimulationDataset,
    NoTransformer,
    spils_loss,
    serialize_scaler,
    deserialize_scaler,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across torch, numpy, and random.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)


class SPILSNet:
    """
    Wrapper class for SPILSNet (Structure-Preserving Input-Output Learning System Network).

    Handles data scaling, model training, persistence, and inference using a Scikit-learn style API.
    """

    def __init__(
        self,
        save_path: str = "spilsnet_model",
        input_scaler_class: Any = None,
        internal_in_scaler_class: Any = None,
        internal_out_scaler_class: Any = None,
        output_scaler_class: Any = None,
        output_transformer: Any = NoTransformer,
        loss_fn: Callable = spils_loss,
        scheduler_class: type = ReduceLROnPlateau,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the SPILSNet wrapper.

        Args:
            save_path (str): Path to save the model weights and metadata.
            input_scaler_class (Any): Scaler class for input data (Scikit-learn style).
            internal_in_scaler_class (Any): Scaler class for incoming internal states.
            internal_out_scaler_class (Any): Scaler class for internal state deltas.
            output_scaler_class (Any): Scaler class for output data.
            output_transformer (Any): Transformer class for output data (e.g., Log, CubeRoot).
            loss_fn (Callable): Loss function to use. Defaults to spils_loss.
            hyperparameters (Optional[Dict[str, Any]]): Dictionary of training hyperparameters.
            model_config (Optional[Dict[str, Any]]): Dictionary of model architecture configuration.
        """
        if model_config is None:
            raise ValueError("model_config must be provided.")

        self.model_config = model_config
        self.input_size = model_config["input_size"]
        self.problem_dimension = model_config.get("dimension", 2)
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration for data processing
        self.input_scaler_class = input_scaler_class
        self.internal_in_scaler_class = internal_in_scaler_class
        self.internal_out_scaler_class = internal_out_scaler_class
        self.output_scaler_class = output_scaler_class
        self.output_transform = output_transformer.transform
        self.output_inverse_transform = output_transformer.inverse_transform
        self.loss_fn = loss_fn

        self.scheduler_class = scheduler_class
        # Default to your legacy parameters if none are provided
        self.scheduler_kwargs = scheduler_kwargs or {
            "mode": "min",
            "factor": 0.5,
            "patience": 20,
        }

        self.curr_epoch = 0
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.optimizer_state_dict = None

        self.set_hyperparameters(hyperparameters or {})

        self._model = SPILSNetCore(model_config)
        self._model.to(self.device)

    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Set training hyperparameters.

        Args:
            hyperparameters (Dict[str, Any]): Dictionary of hyperparameters.
        """
        self.learning_rate = hyperparameters.get("learning_rate", 0.01)
        self.num_epochs = hyperparameters.get("num_epochs", 5000)
        self.weight_decay = hyperparameters.get("weight_decay", 0.01)
        self.batch_size = hyperparameters.get("batch_size", 2048)
        self.early_stop_patience = hyperparameters.get("early_stop_patience", 50)

        self.loss_alpha = hyperparameters.get("loss_alpha", 0.995)
        self.loss_beta = hyperparameters.get("loss_beta", 0.005)
        self.loss_gamma = hyperparameters.get("loss_gamma", 0.0)

    def _setup_training(self) -> None:
        """
        Setup optimizer, scheduler, and loss criterion before training.
        """
        self.optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.optimizer_state_dict is not None:
            self.optimizer_state_dict["param_groups"][0]["initial_lr"] = self.learning_rate
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        self.scheduler = self.scheduler_class(
            self.optimizer,
            **self.scheduler_kwargs
        )

        self.n_nodes = self.input_size // self.problem_dimension
        self.loss_criterion = partial(self.loss_fn, n_nodes=self.n_nodes, dimension=self.problem_dimension, alpha=self.loss_alpha, beta=self.loss_beta, gamma=self.loss_gamma)

    def prep_data(
        self,
        X: np.ndarray,
        internal_states: np.ndarray,
        Y: np.ndarray,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        num_workers: int = 0,
    ) -> None:
        """
        Prepare and scale data for training, validation, and testing.

        Args:
            X (np.ndarray): Input features.
            internal_states (np.ndarray): Internal states.
            Y (np.ndarray): Target values.
            train_indices (Optional[np.ndarray]): Explicit indices for the training set.
                If ``None`` (or ``val_indices`` is ``None``) a random 70/15/15 split of
                the simulations will be generated automatically.
            val_indices (Optional[np.ndarray]): Explicit indices for the validation set.
            test_indices (Optional[np.ndarray]): Explicit indices for the test set. If
                omitted during fallback splitting it will be assigned the remaining
                simulations.
            num_workers (int): Number of subprocesses for data loading. Defaults to 0.
        """
        # legacy behavior: if either train or val indices are omitted, perform a
        # random 70/15/15 split of the simulations.  This keeps the public API more
        # forgiving (and matches existing test expectations).
        if train_indices is None or val_indices is None:
            n = X.shape[0]
            all_idx = np.arange(n)
            np.random.shuffle(all_idx)
            n_train = int(np.floor(0.7 * n))
            n_val = int(np.floor(0.15 * n))
            n_test = n - n_train - n_val

            train_indices = all_idx[:n_train]
            val_indices = all_idx[n_train : n_train + n_val]
            if test_indices is None:
                test_indices = all_idx[n_train + n_val :]
            elif test_indices.size == 0:
                # keep empty array if explicitly provided
                test_indices = np.array([], dtype=int)

            # note: warning removed -- tests expect fallback
        

        # Capture raw initial state for inference
        self.raw_initial_internal_state = internal_states[0, 0, :]

        # Determine if test_indices were provided; if not, pass an empty array to scale_data
        has_test_set = test_indices is not None
        _test_indices = test_indices if has_test_set else np.array([], dtype=int)
        splits = [train_indices, val_indices, _test_indices]

        Y = self.output_transform(Y)

        # 1. Scale Input Data
        input_scaler_to_use = getattr(self, "input_scaler", None) or self.input_scaler_class
        self.input_scaler, train_in, val_in, test_in = scale_data(
            X, splits, concatenate=True, scaler=input_scaler_to_use
        )

        # 2. Scale Internal States
        internal_scaler_to_use = getattr(self, "internal_scaler", None) or self.internal_in_scaler_class
        self.internal_scaler, train_int_in, val_int_in, test_int_in = scale_data(
            internal_states[:, :-1, :], splits, concatenate=True, scaler=internal_scaler_to_use
        )

        # 3. Calculate and Scale Internal State Deltas
        internal_state_out = np.array(
            [
                self.internal_scaler.transform(internal_states[i, 1:, :]) - self.internal_scaler.transform(internal_states[i, :-1, :])
                for i in range(len(internal_states))
            ]
        )

        internal_out_scaler_to_use = getattr(self, "internal_out_scaler", None) or self.internal_out_scaler_class
        self.internal_out_scaler, train_int_out, val_int_out, test_int_out = scale_data(
            internal_state_out, splits, concatenate=True, scaler=internal_out_scaler_to_use
        )

        # 4. Scale Target Data
        output_scaler_to_use = getattr(self, "output_scaler", None) or self.output_scaler_class
        self.output_scaler, train_target, val_target, test_target = scale_data(
            Y, splits, concatenate=True, scaler=output_scaler_to_use
        )

        # Create datasets
        train_dataset = SimulationDataset(train_in, train_int_in, train_target, train_int_out, device=self.device)
        val_dataset = SimulationDataset(val_in, val_int_in, val_target, val_int_out, device=self.device)

        # Create loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers)

        if has_test_set:
            test_dataset = SimulationDataset(test_in, test_int_in, test_target, test_int_out, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=num_workers)
        else:
            self.test_loader = None

        # Initialize the scaled initial state for sequential inference
        initial_state_scaled = self.internal_scaler.transform(self.raw_initial_internal_state.reshape(1, -1))
        self.initial_internal_state = torch.tensor(initial_state_scaled, dtype=torch.float64, device=self.device)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        internal_states: np.ndarray,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        num_workers: int = 0,
    ) -> None:
        """
        High-level API: Fit the SPILSNet model from raw NumPy arrays.

        Args:
            X (np.ndarray): Input nodal features.
            Y (np.ndarray): Target nodal features.
            internal_states (np.ndarray): Physical internal states.
            train_indices (np.ndarray): Indices for training simulations.
            val_indices (np.ndarray): Indices for validation simulations.
            test_indices (Optional[np.ndarray]): Indices for testing simulations.
            num_workers (int): Number of background workers for DataLoader.
        """
        self.prep_data(X, internal_states, Y, train_indices, val_indices, test_indices, num_workers=num_workers)
        self._setup_training()
        self._train_loop()

    def fit_from_loaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Mid-level API: Fit the model using pre-configured PyTorch DataLoaders.
        Note: The user is responsible for ensuring data is appropriately scaled
        and `self.input_scaler` etc. are manually set if `predict()` is needed.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self._setup_training()
        self._train_loop()

    def _train_loop(self) -> None:
        """
        Internal method handling the actual epoch iterations.
        """
        patience_counter = 0

        for epoch in range(self.curr_epoch, self.num_epochs):
            self.curr_epoch = epoch
            self._model.train()
            total_train_loss = 0.0

            for batch in self.train_loader:
                inputs, int_in, targets, int_out = batch
                self.optimizer.zero_grad()

                outputs, next_int = self._model(inputs, int_in)
                loss = self.loss_criterion(
                    outputs, targets, next_int, int_out,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item()

            stop_early, patience_counter, total_val_loss = self._validate_epoch(
                epoch, total_train_loss, patience_counter
            )

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(total_val_loss)
            else:
                self.scheduler.step()

            if total_train_loss > 1e6:
                logger.warning("Training diverged. Stopping.")
                break

            if stop_early:
                break

        logger.info("Training finished!")
        if getattr(self, "test_loader", None) is not None:
            self.calculate_test_metrics()

    def _validate_epoch(self, epoch: int, total_train_loss: float, patience_counter: int) -> Tuple[bool, int, float]:
        """
        Run validation for the current epoch.
        """
        self._model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, int_in, targets, int_out = batch
                outputs, next_int = self._model(inputs, int_in)
                total_val_loss += self.loss_criterion(
                    outputs, targets, next_int, int_out
                ).item()

        avg_train_loss = total_train_loss / len(self.train_loader)
        avg_val_loss = total_val_loss / len(self.val_loader)

        self.optimizer_state_dict = self.optimizer.state_dict()

        # Save current model
        self.save(self.save_path)

        stop = False
        if avg_val_loss < self.best_val_loss:
            patience_counter = 0
            self.best_val_loss = avg_val_loss
            self.best_epoch = epoch
            self.save(f"{self.save_path}_best")
        else:
            patience_counter += 1
            if patience_counter >= self.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                stop = True

        logger.info(
            f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val Loss: {self.best_val_loss:.6f} | Best Epoch: {self.best_epoch + 1}"
        )

        return stop, patience_counter, avg_val_loss

    def calculate_test_metrics(self) -> None:
        """
        Evaluate model on the test set and log results.
        Calculates one-step-ahead prediction errors in raw physical units.
        """
        self._model.eval()
        all_y_pred = []
        all_y_true = []
        all_i_pred_delta = []
        all_i_true_delta = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, int_in, targets, int_out = batch
                outputs, next_int_delta = self._model(inputs, int_in)

                all_y_pred.append(outputs.cpu().numpy())
                all_y_true.append(targets.cpu().numpy())
                all_i_pred_delta.append(next_int_delta.cpu().numpy())
                all_i_true_delta.append(int_out.cpu().numpy())

        y_pred_scaled = np.concatenate(all_y_pred, axis=0)
        y_true_scaled = np.concatenate(all_y_true, axis=0)
        i_pred_delta_scaled_out = np.concatenate(all_i_pred_delta, axis=0)
        i_true_delta_scaled_out = np.concatenate(all_i_true_delta, axis=0)

        # 1. Force Prediction Metrics (Raw Units)
        y_pred_raw = self.output_inverse_transform(self.output_scaler.inverse_transform(y_pred_scaled))
        y_true_raw = self.output_inverse_transform(self.output_scaler.inverse_transform(y_true_scaled))

        l1_F = np.mean(np.abs(y_pred_raw - y_true_raw))
        # Normalized L1 (relative error)
        rel_l1_F = l1_F / (np.mean(np.abs(y_true_raw)) + 1e-10)

        # 2. Internal State Prediction Metrics (Raw Units)
        # i_delta_scaled_out is in the domain of internal_out_scaler
        i_pred_delta_scaled_state = self.internal_out_scaler.inverse_transform(i_pred_delta_scaled_out)
        i_true_delta_scaled_state = self.internal_out_scaler.inverse_transform(i_true_delta_scaled_out)

        # Convert delta from scaled state domain to raw state domain
        # Formula: Delta_Raw = Delta_Scaled_State_Domain / internal_scaler.scale_
        # If internal_scaler is NoScaler, scale_ is 1.0.
        it_scale = getattr(self.internal_scaler, "scale_", 1.0)
        i_pred_delta_raw = i_pred_delta_scaled_state / it_scale
        i_true_delta_raw = i_true_delta_scaled_state / it_scale

        l1_i = np.mean(np.abs(i_pred_delta_raw - i_true_delta_raw))

        # 3. Training Domain Metrics (Scaled MSE for reference)
        mse_scaled = np.mean((y_pred_scaled - y_true_scaled) ** 2)

        logger.info(
            f"Test Set Metrics (Best Epoch: {self.best_epoch + 1}):\n"
            f"  MSE (Scaled): {mse_scaled:.8f}\n"
            f"  L1 Force (Raw): {l1_F:.8f} (Rel: {rel_l1_F:.2%})\n"
            f"  L1 Internal State Delta (Raw): {l1_i:.8f}"
        )

    def initialize_memory_variables(self) -> None:
        """
        Prepare for sequential inference by resetting hidden state and counters.
        """
        self._model.eval()
        self.hidden_state = self.initial_internal_state.detach().clone()
        self.num_steps_predicted = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict a single step forward and update internal state.

        Args:
            x (np.ndarray): Current input nodal features [input_size].

        Returns:
            np.ndarray: Predicted nodal forces for the next step.
        """
        self._model.eval()

        x_norm = self.input_scaler.transform(np.array(x).reshape(1, -1))
        x_tensor = torch.tensor(x_norm, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            pred_scaled, internal_delta_scaled = self._model(x_tensor, self.hidden_state)

            # Update state with unscaled delta
            delta_np = internal_delta_scaled.cpu().numpy()
            delta_unscaled = self.internal_out_scaler.inverse_transform(delta_np)
            self.hidden_state += torch.tensor(delta_unscaled, dtype=torch.float64, device=self.device)

            # Transform back to raw coordinates
            pred_np = pred_scaled.cpu().numpy().reshape(1, -1)
            y_unscaled = self.output_scaler.inverse_transform(pred_np)[0]
            y_final = self.output_inverse_transform(y_unscaled)

        self.num_steps_predicted += 1
        return y_final

    def get_trainable_params(self) -> int:
        """
        Return the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """
        Save model weights and metadata in a unified, non-pickle safetensors format.

        Args:
            path (str): Base path to save the model. Extension .safetensors will be added.
        """
        path = str(path)
        if not path.endswith(".safetensors"):
            path = f"{path}.safetensors"

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # 1. Prepare Metadata (JSON-serializable)
        metadata = {
            "model_config": json.dumps(self.model_config),
            "hyperparameters": json.dumps(
                {
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "early_stop_patience": self.early_stop_patience,
                    "loss_alpha": self.loss_alpha,
                    "loss_beta": self.loss_beta,
                    "loss_gamma": self.loss_gamma,
                }
            ),
            "training_state": json.dumps(
                {
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                    "curr_epoch": self.curr_epoch,
                }
            ),
            "scalers": json.dumps(
                {
                    "input": serialize_scaler(self.input_scaler) if hasattr(self, "input_scaler") else None,
                    "internal": serialize_scaler(self.internal_scaler) if hasattr(self, "internal_scaler") else None,
                    "internal_out": serialize_scaler(self.internal_out_scaler)
                    if hasattr(self, "internal_out_scaler")
                    else None,
                    "output": serialize_scaler(self.output_scaler) if hasattr(self, "output_scaler") else None,
                }
            ),
            "initial_state": json.dumps(
                self.raw_initial_internal_state.tolist() if hasattr(self, "raw_initial_internal_state") else None
            ),
        }

        # 2. Extract State Dict and Optimizer State
        tensors = {k: v.contiguous() for k, v in self._model.state_dict().items()}

        if hasattr(self, "optimizer"):
            opt_state = self.optimizer.state_dict()
            # We only store the tensors from the optimizer state
            for i, group in enumerate(opt_state["param_groups"]):
                for key, val in group.items():
                    if isinstance(val, torch.Tensor):
                        tensors[f"optimizer.param_groups.{i}.{key}"] = val.contiguous()

            for key, val in opt_state["state"].items():
                for subkey, subval in val.items():
                    if isinstance(subval, torch.Tensor):
                        # key is the parameter ID (integer)
                        tensors[f"optimizer.state.{key}.{subkey}"] = subval.contiguous()

        # 3. Save Unified File
        save_file(tensors, path, metadata=metadata)
        logger.info(f"Model saved to {path} (Safetensors format)")

    @classmethod
    def load(cls, path: str) -> "SPILSNet":
        """
        Load a SPILSNet instance. Supports both unified safetensors and legacy pth/pkl formats.

        Args:
            path (str): Path to the model file (with or without extension).

        Returns:
            SPILSNet: Loaded model instance.
        """
        import pickle
        from safetensors import safe_open

        path = str(path)
        # 1. Try Safetensors (New Format)
        st_path = path if path.endswith(".safetensors") else f"{path}.safetensors"
        if os.path.exists(st_path):
            tensors = load_file(st_path)
            with safe_open(st_path, framework="pt") as f_safe:
                metadata = f_safe.metadata()

            if not metadata:
                raise ValueError(f"No metadata found in {st_path}")

            model_config = json.loads(metadata["model_config"])
            hyperparameters = json.loads(metadata["hyperparameters"])
            training_state = json.loads(metadata["training_state"])
            scalers_data = json.loads(metadata["scalers"])
            initial_state_data = json.loads(metadata["initial_state"])

            instance = cls(
                save_path=os.path.splitext(st_path)[0],
                model_config=model_config,
                hyperparameters=hyperparameters,
            )

            # Separate model tensors and optimizer tensors
            model_tensors = {}
            optimizer_tensors = {}
            for k, v in tensors.items():
                if k.startswith("optimizer."):
                    optimizer_tensors[k] = v
                else:
                    model_tensors[k] = v

            instance._model.load_state_dict(model_tensors)

            instance.best_val_loss = training_state["best_val_loss"]
            instance.best_epoch = training_state["best_epoch"]
            instance.curr_epoch = training_state["curr_epoch"]

            instance.input_scaler = deserialize_scaler(scalers_data["input"])
            instance.internal_scaler = deserialize_scaler(scalers_data["internal"])
            instance.internal_out_scaler = deserialize_scaler(scalers_data["internal_out"])
            instance.output_scaler = deserialize_scaler(scalers_data["output"])

            # 4. Restore Optimizer State (if present)
            if optimizer_tensors:
                # Reconstruct optimizer state dict
                opt_state = {"param_groups": [], "state": {}}

                # Check for param groups
                group_indices = sorted(list(set([int(k.split(".")[2]) for k in optimizer_tensors if "param_groups" in k])))
                for i in group_indices:
                    group = {}
                    prefix = f"optimizer.param_groups.{i}."
                    for k, v in optimizer_tensors.items():
                        if k.startswith(prefix):
                            group[k[len(prefix):]] = v
                    opt_state["param_groups"].append(group)

                # Check for state
                state_ids = sorted(list(set([k.split(".")[2] for k in optimizer_tensors if "state" in k])))
                for sid in state_ids:
                    state_id = int(sid)
                    opt_state["state"][state_id] = {}
                    prefix = f"optimizer.state.{sid}."
                    for k, v in optimizer_tensors.items():
                        if k.startswith(prefix):
                            opt_state["state"][state_id][k[len(prefix):]] = v

                instance.optimizer_state_dict = opt_state

            if initial_state_data is not None:
                instance.raw_initial_internal_state = np.array(initial_state_data)
                initial_state_scaled = instance.internal_scaler.transform(
                    instance.raw_initial_internal_state.reshape(1, -1)
                )
                instance.initial_internal_state = torch.tensor(
                    initial_state_scaled, dtype=torch.float64, device=instance.device
                )

            logger.info(f"Model loaded from {st_path} (Safetensors)")
            return instance

        # 2. Try Legacy Fallback (pth/pkl)
        # Check for metadata.pkl
        pkl_path = f"{path}_metadata.pkl" if not path.endswith("_metadata.pkl") else path
        weights_path = pkl_path.replace("_metadata.pkl", "_weights.pth")

        if os.path.exists(pkl_path) and os.path.exists(weights_path):
            logger.warning(f"Safetensors not found. Falling back to legacy format: {pkl_path}")
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)

            instance = cls(
                save_path=pkl_path.replace("_metadata.pkl", ""),
                model_config=state["model_config"],
                hyperparameters=state["hyperparameters"],
            )

            # Map to correct device
            instance._model.load_state_dict(torch.load(weights_path, map_location=instance.device, weights_only=True))

            instance.best_val_loss = state.get("best_val_loss", float("inf"))
            instance.best_epoch = state.get("best_epoch", 0)
            instance.curr_epoch = state.get("curr_epoch", 0)

            scalers = state.get("scalers", {})
            instance.input_scaler = scalers.get("input")
            instance.internal_scaler = scalers.get("internal")
            instance.internal_out_scaler = scalers.get("internal_out")
            instance.output_scaler = scalers.get("output")

            initial_state = state.get("initial_state")
            if initial_state is not None:
                instance.raw_initial_internal_state = initial_state
                initial_state_scaled = instance.internal_scaler.transform(
                    instance.raw_initial_internal_state.reshape(1, -1)
                )
                instance.initial_internal_state = torch.tensor(
                    initial_state_scaled, dtype=torch.float64, device=instance.device
                )

            logger.info(f"Model loaded from {pkl_path} (Legacy)")
            return instance

        raise FileNotFoundError(f"Could not find model at {path} in either .safetensors or legacy .pth/.pkl format.")
