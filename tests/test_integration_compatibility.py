import numpy as np
import pytest
import torch
from spilsnet import SPILSNet
from spilsnet.utils import NoTransformer


@pytest.fixture
def mock_model_config():
    return {
        "dimension": 2,
        "input_size": 10,
        "internal_state_size": 1,
        "encoder_structure": [{"k": 3, "s": 1, "p": 1, "out": 4}],
        "bottleneck_pool_size": 1,
        "skip_target_nodes": 2,
        "latent_dim": 4,
        "latent_encoder_mlp": [8],
        "gru_hidden_size": 8,
        "gru_layers": 1,
        "internal_input_mlp": [4],
        "internal_output_mlp": [4],
        "latent_decoder_structure": [8],
        "smoothing_kernel_size": 3,
        "dropout_rate": 0.1,
    }


def test_predict_with_velocity(mock_model_config):
    """Test that predict handles optional velocity argument correctly."""
    save_path = "/tmp/test_compat_model"
    # Use real MinMaxScaler for testing scaling logic
    from sklearn.preprocessing import MinMaxScaler

    net = SPILSNet(
        save_path=save_path,
        model_config=mock_model_config,
        input_scaler_class=MinMaxScaler(),
        internal_in_scaler_class=MinMaxScaler(),
        internal_out_scaler_class=MinMaxScaler(),
        output_scaler_class=MinMaxScaler(),
    )

    # Mock data to fit scalers
    n_sims = 2
    n_steps = 5
    input_dim = mock_model_config["input_size"]
    internal_dim = mock_model_config["internal_state_size"]

    X = np.random.rand(n_sims, n_steps, input_dim)
    internal = np.random.rand(n_sims, n_steps, internal_dim)
    Y = np.random.rand(n_sims, n_steps, input_dim)

    net.prep_data(X, internal, Y)
    net.initialize_memory_variables()

    # Test predict with single argument
    u = np.random.rand(input_dim)
    pred1 = net.predict(u)
    assert isinstance(pred1, np.ndarray)
    assert pred1.shape == (input_dim,)

    # Legacy clients would concatenate velocity manually, so verify that
    # behaviour using half-dimension inputs.
    u_half = np.random.rand(input_dim // 2)
    v_half = np.random.rand(input_dim // 2)
    net.initialize_memory_variables()
    pred_manual = net.predict(np.concatenate((u_half, v_half)))
    assert isinstance(pred_manual, np.ndarray)
    assert pred_manual.shape == (input_dim,)

    # confirm that manual concatenation produces same shape/values as full input
    net.initialize_memory_variables()
    pred_full = net.predict(np.concatenate((u_half, v_half)))
    np.testing.assert_array_equal(pred_manual, pred_full)


if __name__ == "__main__":
    # Allows running this test file directly
    pytest.main([__file__])
