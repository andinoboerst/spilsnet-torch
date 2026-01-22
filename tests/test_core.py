import torch
import pytest
from spilsnet.models import SPILSNetCore

def test_core_initialization():
    model = SPILSNetCore(
        dimension=2,
        input_size=10,
        internal_state_size=5,
        spatial_linear_layers=[5],
        hidden_internal_size=16,
        conv_layers_out_channels=[16, 1],
        kernel_size=3,
        internal_layers_in=[],
        n_gru_cells=1,
        internal_layers_out=[16, 16],
        deconv_layers_out_channels=[16],
        dropout_rate=0.2
    )
    assert isinstance(model, torch.nn.Module)

def test_core_forward_pass():
    batch_size = 32
    input_size = 10
    internal_state_size = 5
    dimension = 2
    
    model = SPILSNetCore(
        dimension=dimension,
        input_size=input_size,
        internal_state_size=internal_state_size,
        spatial_linear_layers=[5],
        hidden_internal_size=16,
        conv_layers_out_channels=[16, 1],
        kernel_size=3,
        internal_layers_in=[],
        n_gru_cells=1,
        internal_layers_out=[16, 16],
        deconv_layers_out_channels=[16],
        dropout_rate=0.0 # Disable dropout for deterministic output shape check if needed, though shape is constant
    )
    
    # Create dummy input
    x_in = torch.randn(batch_size, input_size, dtype=torch.float64)
    internal_state = torch.randn(batch_size, internal_state_size, dtype=torch.float64)
    
    # Forward pass
    output, next_internal_state = model(x_in, internal_state)
    
    # Check shapes
    assert output.shape == (batch_size, input_size)
    assert next_internal_state.shape == (batch_size, internal_state_size)

def test_core_invalid_config():
    # Test invalid configuration raises ValueError
    with pytest.raises(ValueError):
        SPILSNetCore(
            dimension=2,
            input_size=10,
            internal_state_size=5,
            spatial_linear_layers=[], # Invalid: empty list
            hidden_internal_size=16
        )
