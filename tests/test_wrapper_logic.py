from __future__ import annotations

import torch
import numpy as np
import pytest
import os
from spilsnet.wrapper import SPILSNet

# Minimal valid SPILSNetCore config
BASE_MODEL_CONFIG = {
    "dimension": 2,
    "input_size": 10,
    "internal_state_size": 5,
    "encoder_structure": [
        {"out": 16, "k": 3, "s": 1, "p": 1},
        {"out": 1, "k": 3, "s": 1, "p": 1},
    ],
    "bottleneck_pool_size": 3,
    "latent_dim": 8,
    "gru_hidden_size": 16,
    "latent_encoder_mlp": [16],
    "internal_input_mlp": [16],
    "internal_output_mlp": [16],
    "dropout_rate": 0.2,
}

SMALL_MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    "input_size": 10,
    "internal_state_size": 5,
}


def _make_data(n_sims: int, n_steps: int, input_size: int, internal_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.random.randn(n_sims, n_steps, input_size)
    Y = np.random.randn(n_sims, n_steps, input_size)
    I = np.random.randn(n_sims, n_steps, internal_size)
    return X, Y, I


def test_wrapper_fit_predict_logic() -> None:
    n_sims, n_steps = 5, 20
    X, Y, I = _make_data(n_sims, n_steps, 10, 5)

    model = SPILSNet(model_config=SMALL_MODEL_CONFIG)
    model.num_epochs = 1
    model.batch_size = 10

    model.fit(X, Y, I, train_indices=np.array([0, 1, 2]), val_indices=np.array([3]), test_indices=np.array([4]))

    assert model.optimizer_state_dict is not None

    model.initialize_memory_variables()
    y_pred = model.predict(np.random.randn(10))
    assert y_pred.shape == (10,)
    assert model.num_steps_predicted == 1


def test_wrapper_fallback_split() -> None:
    n_sims, n_steps, input_size, internal_size = 100, 10, 10, 5
    X, Y, I = _make_data(n_sims, n_steps, input_size, internal_size)

    model = SPILSNet(model_config=SMALL_MODEL_CONFIG)
    model.num_epochs = 0
    model.batch_size = 10

    model.fit(X, Y, I)  # no indices -> 70/15/15 fallback

    expected_train_size = 70 * (n_steps - 1)
    expected_val_size = 15 * (n_steps - 1)
    expected_test_size = 15 * (n_steps - 1)

    assert len(model.train_loader.dataset) == expected_train_size
    assert len(model.val_loader.dataset) == expected_val_size
    assert len(model.test_loader.dataset) == expected_test_size


def test_wrapper_custom_scalers() -> None:
    from sklearn.preprocessing import StandardScaler

    model = SPILSNet(
        model_config=SMALL_MODEL_CONFIG,
        input_scaler_class=StandardScaler(),
        internal_in_scaler_class=StandardScaler(),
        internal_out_scaler_class=StandardScaler(),
        output_scaler_class=StandardScaler(),
    )

    assert isinstance(model.input_scaler_class, StandardScaler)
    assert isinstance(model.internal_in_scaler_class, StandardScaler)
    assert isinstance(model.internal_out_scaler_class, StandardScaler)
    assert isinstance(model.output_scaler_class, StandardScaler)


def test_wrapper_serialization(tmp_path) -> None:
    """Test that model can be saved and loaded correctly."""
    n_sims, n_steps = 10, 10  # Increased n_sims from 5 to 10
    X, Y, I = _make_data(n_sims, n_steps, 10, 5)

    save_path = str(tmp_path / "test_model")
    model = SPILSNet(model_config=SMALL_MODEL_CONFIG, save_path=save_path)
    model.num_epochs = 1
    model.fit(X, Y, I)

    # Initial prediction
    model.initialize_memory_variables()
    x_test = np.random.randn(10)
    y_pred_1 = model.predict(x_test)

    # Load new instance
    model_loaded = SPILSNet.load(save_path)

    # Prediction after loading should be somewhat consistent (given deterministic weights)
    model_loaded.initialize_memory_variables()
    y_pred_2 = model_loaded.predict(x_test)

    assert np.allclose(y_pred_1, y_pred_2)
    assert model_loaded.model_config == SMALL_MODEL_CONFIG


def test_legacy_loading(tmp_path):
    """
    Test that SPILSNet.load correctly falls back to legacy .pth/.pkl files.
    """
    model_config = {
        "dimension": 2,
        "input_size": 10,
        "encoder_structure": [{"out": 8, "k": 3, "s": 1, "p": 1}],
        "bottleneck_pool_size": 1,
        "latent_dim": 4,
        "gru_hidden_size": 16,
        "latent_encoder_mlp": [8],
        "internal_state_size": 2,
        "internal_input_mlp": [8],
        "internal_output_mlp": [8],
    }
    net = SPILSNet(model_config=model_config)
    save_path = str(tmp_path / "legacy_model")

    # Manually save in old format
    import pickle

    state = {
        "model_config": model_config,
        "hyperparameters": {
            "learning_rate": 0.01,
            "num_epochs": 100,
            "weight_decay": 0.0,
            "batch_size": 32,
            "early_stop_patience": 10,
            "loss_alpha": 0.5,
            "loss_beta": 0.5,
            "loss_gamma": 0.0,
        },
        "best_val_loss": 0.5,
        "best_epoch": 10,
        "curr_epoch": 20,
        "scalers": {
            "input": None,
            "internal": None,
            "internal_out": None,
            "output": None,
        },
    }
    with open(f"{save_path}_metadata.pkl", "wb") as f:
        pickle.dump(state, f)
    torch.save(net._model.state_dict(), f"{save_path}_weights.pth")

    # Load back using new load() - it should fallback to legacy
    loaded = SPILSNet.load(save_path)

    assert loaded.best_epoch == 10
    assert loaded.curr_epoch == 20
    assert loaded.model_config["latent_dim"] == 4

    # Verify weights were loaded (compare some parameter)
    for p1, p2 in zip(net._model.parameters(), loaded._model.parameters()):
        assert torch.allclose(p1, p2)
