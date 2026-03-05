import numpy as np
import logging
import pickle
import random
from functools import partial

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spilsnet.models import SPILSNetCore
from spilsnet.utils import (
    scale_data,
    SimulationDataset,
    NoTransformer,
    spils_loss
)

logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)


# Set random seed for reproducibility
set_seed(8)


class SPILSNet():
    def __init__(
        self,
        problem_dimension: int = 2,
        save_path: str = "current_lstm_model",
        input_scaler_class=None,
        internal_scaler_class=None,
        output_scaler_class=None,
        output_transformer=NoTransformer,
        loss_fn=spils_loss,
        hyperparameters: dict = {},
        model_config: dict = {},
    ) -> None:

        """
        Initialize the SPILSNet wrapper.

        Args:
            save_path (str): Path to save the model.
            input_scaler (object): Scaler for input data.
            internal_scaler (object): Scaler for internal states.
            output_scaler (object): Scaler for output data.
            output_transform (callable): Function to transform output data before scaling.
            output_inverse_transform (callable): Function to inverse transform output data after unscaling.
            loss_fn (callable): Loss function to use. Defaults to spils_loss.
            **model_config: Configuration for the SPILSNetCore model.
        """

        self.input_size = model_config["input_size"]
        self.problem_dimension = model_config.get("problem_dimension", 2)
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dependency injection for scalers and transforms
        self.input_scaler_class = input_scaler_class
        self.internal_scaler_class = internal_scaler_class
        self.output_scaler_class = output_scaler_class
        self.output_transform = output_transformer.transform
        self.output_inverse_transform = output_transformer.inverse_transform
        self.loss_fn = loss_fn

        self.curr_epoch = 0
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.optimizer_state_dict = None

        self.set_hyperparameters(hyperparameters)

        self._model = SPILSNetCore(model_config)
        self._model.to(self.device)

    def set_hyperparameters(self, hyperparameters: dict) -> None:
        self.learning_rate = hyperparameters.get('learning_rate', 0.001)
        self.num_epochs = hyperparameters.get('num_epochs', 5000)
        self.weight_decay = hyperparameters.get('weight_decay', 0.01)
        self.batch_size = hyperparameters.get('batch_size', 2048)
        self.early_stop_patience = hyperparameters.get('early_stop_patience', 50)

        self.loss_alpha = hyperparameters.get('loss_alpha', 0.1)
        self.loss_beta = hyperparameters.get('loss_beta', 0.9)
        self.loss_gamma = hyperparameters.get('loss_gamma', 0.1)

    def set_fit_parameters(self) -> None:
        self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.optimizer_state_dict is not None:
            self.optimizer_state_dict['param_groups'][0]['initial_lr'] = self.learning_rate
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',       # Monitor validation loss (minimize)
            factor=0.1,       # Reduce LR by 10x when plateau detected
            patience=20,       # Wait 20 epochs before reducing LR
        )

        self.n_nodes = self.input_size // self.problem_dimension

        self.loss_criterion = partial(self.loss_fn, n_nodes=self.n_nodes, dimension=self.problem_dimension)

    def prep_data(self, X, internal_states, Y, train_indices=None, val_indices=None, test_indices=None) -> None:
        if train_indices is None or val_indices is None or test_indices is None:
            # Fallback to 70/15/15 split
            n_sims = len(X)
            indices = np.arange(n_sims)
            # np.random.shuffle(indices) # Should we shuffle? Maybe keep deterministic or rely on user to shuffle.
            # Let's keep it deterministic for now or use a fixed seed if we shuffle.
            # The user asked for a "basic" split. Sequential might be safer for time series if sims are ordered,
            # but usually sims are independent. Let's do sequential to be safe and reproducible without seed issues.

            n_train = int(0.7 * n_sims)
            n_val = int(0.15 * n_sims)
            # n_test = n_sims - n_train - n_val

            train_sims = indices[:n_train]
            val_sims = indices[n_train:n_train + n_val]
            test_sims = indices[n_train + n_val:]
        else:
            train_sims, val_sims, test_sims = train_indices, val_indices, test_indices

        Y = self.output_transform(Y)

        # Use injected scaler instances if provided
        # Slice X and Y to match the length of internal_states transitions (N-1)
        # Assuming X, Y, and internal_states have the same number of steps N
        self.input_scaler, train_input_data, val_input_data, test_input_data = scale_data(X[:, :-1, :], [train_sims, val_sims, test_sims], concatenate=True, scaler=self.input_scaler_class)
        self.internal_scaler, train_internal_data_in, val_internal_data_in, test_internal_data_in = scale_data(internal_states[:, :-1, :], [train_sims, val_sims, test_sims], concatenate=True, scaler=self.internal_scaler_class)
        internal_state_out = np.array([self.internal_scaler.transform(internal_states[i, 1:, :]) - self.internal_scaler.transform(internal_states[i, :-1, :]) for i in range(len(internal_states))])

        self.internal_out_scaler, train_internal_data_out, val_internal_data_out, test_internal_data_out = scale_data(internal_state_out, [train_sims, val_sims, test_sims], concatenate=True, scaler=self.internal_scaler_class)

        self.output_scaler, train_target_data, val_target_data, test_target_data = scale_data(Y[:, :-1, :], [train_sims, val_sims, test_sims], concatenate=True, scaler=self.output_scaler_class)

        train_dataset = SimulationDataset(train_input_data, train_internal_data_in, train_target_data, train_internal_data_out, device=self.device)
        val_dataset = SimulationDataset(val_input_data, val_internal_data_in, val_target_data, val_internal_data_out, device=self.device)
        test_dataset = SimulationDataset(test_input_data, test_internal_data_in, test_target_data, test_internal_data_out, device=self.device)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

    def fit(self, X, Y, internal_states, train_indices=None, val_indices=None, test_indices=None) -> None:
        self.initial_internal_state = internal_states[0, 0, :]

        self.prep_data(X, internal_states, Y, train_indices, val_indices, test_indices)

        self.initial_internal_state = self.internal_scaler.transform(self.initial_internal_state.reshape(1, -1))
        self.initial_internal_state = torch.tensor(self.initial_internal_state, dtype=torch.float64).to(self.device)

        self.set_fit_parameters()

        patience_counter = 0

        for epoch in range(self.curr_epoch, self.num_epochs):
            self.curr_epoch = epoch

            total_train_loss = 0
            self._model.train()
            for batch_inputs, batch_internal_states_in, batch_targets, batch_internal_states_out in self.train_loader:

                self.optimizer.zero_grad()

                # Forward pass: process the entire sequence for the batch
                outputs, internal_states_output = self._model(batch_inputs, batch_internal_states_in)

                # Compute loss: compare the entire output sequence with the target sequence
                loss = self.loss_criterion(outputs, batch_targets, internal_states_output, batch_internal_states_out, self.loss_alpha, self.loss_beta, self.loss_gamma)

                # Backward pass and optimization
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item()

            stop_early, patience_counter, total_val_loss = self.validate_epoch_custom(epoch, self.num_epochs, total_train_loss, self.optimizer, self.early_stop_patience, patience_counter)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(total_val_loss)
            else:
                self.scheduler.step()

            if total_train_loss > 1e5:
                break

            if stop_early:
                break

        logger.info('Training finished!')

        self.calc_test_loss()

    def validate_epoch_custom(self, epoch, num_epochs, total_train_loss, optimizer, early_stop_patience, patience_counter) -> tuple[bool, int, float]:
        # Validation phase
        self._model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, internal_val_in_batch, y_val_batch, internal_val_out_batch in self.val_loader:
                val_outputs, val_internal_outputs = self._model(X_val_batch, internal_val_in_batch)
                total_val_loss += self.loss_criterion(val_outputs, y_val_batch, val_internal_outputs, internal_val_out_batch, self.loss_alpha, self.loss_beta).item()

        total_train_loss /= len(self.train_loader)
        total_val_loss /= len(self.val_loader)

        self.optimizer_state_dict = optimizer.state_dict()

        # Determine base path without extension
        base_path = self.save_path
        if base_path.endswith(".pkl"):
            base_path = base_path[:-4]

        self.save(self.save_path)
        self.save_weights(f"{base_path}_weights.pth")

        # Early stopping (optional)
        stop = False
        if total_val_loss < self.best_val_loss:
            patience_counter = 0
            self.best_val_loss = total_val_loss
            self.best_epoch = epoch
            self.save(f"{base_path}_best.pkl")
            self.save_weights(f"{base_path}_best_weights.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                stop = True

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss (MSE): {total_train_loss:.8f}, Val Loss (MSE): {total_val_loss:.8f}, Best Epoch: {self.best_epoch + 1}, Best Val Loss: {self.best_val_loss:.8f}")

        torch.cuda.empty_cache()  # Free unused memory

        return stop, patience_counter, total_val_loss

    def calc_test_loss(self):
        self._model.eval()
        test_loss = 0.0
        internal_loss = 0.0
        with torch.no_grad():
            for X_test_batch, internal_test_in_batch, y_test_batch, internal_test_out_batch in self.test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(X_test_batch.device), y_test_batch.to(y_test_batch.device)
                test_outputs, test_internal_outputs = self._model(X_test_batch, internal_test_in_batch)

                test_outputs = self.output_scaler.inverse_transform(test_outputs.reshape(-1, test_outputs.shape[-1]).cpu().numpy())
                y_test_batch = self.output_inverse_transform(self.output_scaler.inverse_transform(y_test_batch.reshape(-1, y_test_batch.shape[-1]).cpu().numpy()))

                internal_loss += nn.functional.l1_loss(test_internal_outputs, internal_test_out_batch)

                test_loss = nn.functional.l1_loss(torch.from_numpy(test_outputs), torch.from_numpy(y_test_batch))

        test_loss /= len(self.test_loader)
        internal_loss /= len(self.test_loader)
        logger.info(f"Final Test Loss (L1): {test_loss:.8f}, Internal Loss (L1): {internal_loss:.8f}")

    def initialize_memory_variables(self) -> None:
        self._model.eval()
        self.hidden_state = self.initial_internal_state.detach().clone()  # type: ignore

        self.num_steps_predicted = 0

    def predict(self, x: np.ndarray, _=None) -> np.ndarray:

        self._model.eval()

        # Move input to correct device
        x_norm = self.input_scaler.transform(np.array(x).reshape(1, -1))
        x_tensor = torch.tensor(x_norm, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            # Inference
            pred_scaled, internal_delta_scaled = self._model(x_tensor, self.hidden_state)

            # Update state: Map the delta back to the 'internal_scaler' domain
            delta_np = internal_delta_scaled.cpu().numpy()
            delta_unscaled = self.internal_out_scaler.inverse_transform(delta_np)

            # Keep the update on the same device/type as hidden_state
            self.hidden_state += torch.tensor(delta_unscaled, dtype=torch.float64, device=self.device)

            # Denormalize spatial output
            pred_np = pred_scaled.cpu().numpy()[-1].reshape(1, -1)
            y_unscaled = self.output_scaler.inverse_transform(pred_np)[0]
            y_final = self.output_inverse_transform(y_unscaled)

        self.num_steps_predicted += 1
        return y_final

    def get_trainable_params(self):
        return sum(p.numel() for p in self._model.parameters())

    def save_weights(self, save_path: str) -> None:
        if not save_path.endswith(".pth"):
            save_path = f"{save_path}.pth"
        torch.save(self._model.state_dict(), save_path)

    def load_weights(self, load_path: str) -> None:
        if not load_path.endswith(".pth"):
            load_path = f"{load_path}.pth"
        self._model.load_state_dict(torch.load(load_path, weights_only=True))

    def save(self, path: str) -> None:
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SPILSNet":
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
