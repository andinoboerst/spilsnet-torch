import torch
import pytest
from spilsnet.models import SPILSNetCore

# Minimal valid config for SPILSNetCore
BASE_CONFIG = {
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


def test_core_initialization() -> None:
    model = SPILSNetCore(BASE_CONFIG)
    assert isinstance(model, torch.nn.Module)
    assert model.dtype == torch.float64


def test_core_forward_pass() -> None:
    batch_size = 32
    config = {**BASE_CONFIG, "dropout_rate": 0.0}
    model = SPILSNetCore(config)

    x_in = torch.randn(batch_size, config["input_size"], dtype=torch.float64)
    internal_state = torch.randn(batch_size, config["internal_state_size"], dtype=torch.float64)

    output, next_internal_state = model(x_in, internal_state)

    assert output.shape == (batch_size, config["input_size"])
    assert next_internal_state.shape == (batch_size, config["internal_state_size"])


def test_core_missing_required_key() -> None:
    """SPILSNetCore should raise an error if required config keys are missing."""
    bad_config = {k: v for k, v in BASE_CONFIG.items() if k != "encoder_structure"}
    with pytest.raises((KeyError, TypeError)):
        SPILSNetCore(bad_config)


def test_core_dtype_config() -> None:
    """Test that dtype can be configured."""
    config_f32 = {**BASE_CONFIG, "dtype": "float32"}
    model_f32 = SPILSNetCore(config_f32)
    assert model_f32.dtype == torch.float32

    # Check that a layer is actually float32
    assert model_f32.encoder_stack[0][0].weight.dtype == torch.float32
