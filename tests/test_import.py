import torch
import pytest
from spilsnet import SPILSNetCore, SPILSNet

# Minimal valid config for SPILSNetCore
BASE_MODEL_CONFIG = {
    "dimension": 2,
    "input_size": 10,
    "internal_state_size": 5,
    "encoder_structure": [
        {"out": 16, "k": 3, "s": 1, "p": 1},
        {"out": 1,  "k": 3, "s": 1, "p": 1},
    ],
    "bottleneck_pool_size": 3,
    "latent_dim": 8,
    "gru_hidden_size": 16,
    "latent_encoder_mlp": [16],
    "internal_input_mlp": [16],
    "internal_output_mlp": [16],
    "dropout_rate": 0.2,
}


def test_spilsnet_core_initialization():
    model = SPILSNetCore(BASE_MODEL_CONFIG)
    assert isinstance(model, torch.nn.Module)


def test_spilsnet_wrapper_initialization():
    model = SPILSNet(model_config=BASE_MODEL_CONFIG)
    assert model is not None



def test_spilsnet_generic_scaling():
    from sklearn.preprocessing import StandardScaler

    model = SPILSNet(
        model_config=BASE_MODEL_CONFIG,
        input_scaler_class=StandardScaler(),
        internal_scaler_class=StandardScaler(),
        output_scaler_class=StandardScaler()
    )
    assert model is not None
