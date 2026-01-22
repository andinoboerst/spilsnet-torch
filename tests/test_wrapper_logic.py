import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock nn_predictors if not present
sys.modules['nn_predictors'] = MagicMock()
sys.modules['nn_predictors.misc'] = MagicMock()

from spilsnet.wrapper import SPILSNet

def test_wrapper_fit_predict_logic():
    # Mock data
    n_sims = 5
    n_steps = 20
    input_size = 10
    internal_size = 5
    output_size = 10 # Same as input for this model
    
    X = np.random.randn(n_sims, n_steps, input_size)
    Y = np.random.randn(n_sims, n_steps, output_size)
    internal_states = np.random.randn(n_sims, n_steps, internal_size)
    
    # Define explicit indices
    train_indices = [0, 1, 2]
    val_indices = [3]
    test_indices = [4]

    model = SPILSNet(
        dimension=2,
        input_size=input_size,
        internal_state_size=internal_size,
        spatial_linear_layers=[5],
        hidden_internal_size=16,
        conv_layers_out_channels=[16, 1],
        kernel_size=3,
        deconv_layers_out_channels=[16],
        dropout_rate=0.2
    )
    model.num_epochs = 1
    model.batch_size = 10

    # Fit with explicit indices
    model.fit(X, Y, internal_states, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

    # Check if model parameters are updated (not None)
    assert model.optimizer_state_dict is not None

    # Predict
    model.initialize_memory_variables()
    x_pred = np.random.randn(input_size)
    y_pred = model.predict(x_pred)
    
    assert y_pred.shape == (output_size,)
    assert model.num_steps_predicted == 1

def test_wrapper_fallback_split():
    # Test that fallback split works (70/15/15)
    n_sims = 100
    n_steps = 10
    input_size = 6
    internal_size = 2
    output_size = 6
    
    X = np.random.randn(n_sims, n_steps, input_size)
    Y = np.random.randn(n_sims, n_steps, output_size)
    internal_states = np.random.randn(n_sims, n_steps, internal_size)
    
    model = SPILSNet(
        dimension=2,
        input_size=input_size,
        internal_state_size=internal_size,
        spatial_linear_layers=[6],
        hidden_internal_size=16,
        conv_layers_out_channels=[16, 1],
        kernel_size=3,
        deconv_layers_out_channels=[16],
        dropout_rate=0.2
    )
    model.num_epochs = 0 # Don't train, just prep data
    model.batch_size = 10
    
    # Fit without indices -> should trigger fallback
    model.fit(X, Y, internal_states)
    
    # Check dataset sizes
    # Train: 70% of 100 = 70
    # Val: 15% of 100 = 15
    # Test: 15% of 100 = 15
    # Note: SimulationDataset stores flattened data. 
    # Each sim has n_steps-1 transitions.
    # So train size should be 70 * (n_steps-1)
    
    expected_train_size = 70 * (n_steps - 1)
    expected_val_size = 15 * (n_steps - 1)
    expected_test_size = 15 * (n_steps - 1)
    
    assert len(model.train_loader.dataset) == expected_train_size
    assert len(model.val_loader.dataset) == expected_val_size
    assert len(model.test_loader.dataset) == expected_test_size

def test_wrapper_custom_scalers():
    from sklearn.preprocessing import StandardScaler
    
    model = SPILSNet(
        dimension=2,
        input_size=10,
        internal_state_size=5,
        spatial_linear_layers=[5],
        input_scaler=StandardScaler(),
        internal_scaler=StandardScaler(),
        output_scaler=StandardScaler()
    )
    
    assert isinstance(model.input_scaler, StandardScaler)
    assert isinstance(model.internal_scaler, StandardScaler)
    assert isinstance(model.output_scaler, StandardScaler)
    
    # Manually fit the scaler (mocking)
    model.output_scaler.mean_ = np.array([10.0])
    model.output_scaler.scale_ = np.array([2.0])
    
    # Verify that the loss function is set up correctly with the unscaler
    model.set_fit_parameters()
    assert model.loss_criterion.func == model.loss_fn
    assert 'unscale_func' in model.loss_criterion.keywords
    
    # Test that the unscale function works as expected (StandardScaler unscaling)
    unscale_func = model.loss_criterion.keywords['unscale_func']
    
    # Create dummy tensor
    x = torch.tensor([0.0], dtype=torch.float64, device=model.device)
    
    # Unscale: x * scale + mean = 0 * 2 + 10 = 10
    unscaled_x = unscale_func(x)
    assert torch.isclose(unscaled_x, torch.tensor([10.0], dtype=torch.float64, device=model.device))
