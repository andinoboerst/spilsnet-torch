import numpy as np
import torch
import pytest
from spilsnet import SPILSNetGraph, SPILSNetGraphCore, connectivity_to_edge_index, MinMaxScaler

def test_connectivity_to_edge_index():
    connectivity = np.array([[0, 1, 2], [1, 2, 3]])
    edge_index = connectivity_to_edge_index(connectivity)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0

def test_spilsnet_graph_core_forward():
    config = {
        "dimension": 3,
        "input_size": 30,  # 10 nodes * 3 dims
        "internal_state_size": 1,
        "encoder_structure": [{"out": 16}, {"out": 32}],
        "latent_dim": 16,
        "latent_encoder_mlp": [16],
        "gru_hidden_size": 16,
        "gru_layers": 1,
        "internal_input_mlp": [8],
        "internal_output_mlp": [8],
        "latent_decoder_structure": [32],
        "use_decoder_conv": True,
        "dropout_rate": 0.0,
    }
    model = SPILSNetGraphCore(config)
    x = torch.randn(4, 30, dtype=torch.float64)
    internal = torch.randn(4, 1, dtype=torch.float64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    out, next_int = model(x, internal, edge_index=edge_index)
    assert out.shape == (4, 30)
    assert next_int.shape == (4, 1)

def test_spilsnet_graph_wrapper_fit_predict(tmp_path):
    connectivity = np.array([[0, 1, 2], [1, 2, 3]])
    config = {
        "dimension": 3,
        "input_size": 12,  # 4 nodes * 3 dims
        "internal_state_size": 1,
        "encoder_structure": [{"out": 16}],
        "latent_dim": 16,
        "latent_encoder_mlp": [16],
        "gru_hidden_size": 16,
        "gru_layers": 1,
        "internal_input_mlp": [8],
        "internal_output_mlp": [8],
        "latent_decoder_structure": [16],
        "use_decoder_conv": True,
    }
    hyperparams = {
        "learning_rate": 0.01,
        "batch_size": 10,
        "num_epochs": 2,
        "weight_decay": 0.0,
        "early_stop_patience": 5,
        "loss_alpha": 1.0,
        "loss_beta": 0.1,
        "loss_gamma": 0.0,
    }

    save_file = str(tmp_path / "gnn_test_model")
    reg = SPILSNetGraph(
        connectivity=connectivity,
        save_path=save_file,
        model_config=config,
        hyperparameters=hyperparams,
        input_scaler_class=MinMaxScaler(),
        internal_in_scaler_class=MinMaxScaler(),
        internal_out_scaler_class=MinMaxScaler(),
        output_scaler_class=MinMaxScaler(),
    )

    X = np.random.randn(5, 20, 12)
    Y = np.random.randn(5, 20, 12)
    internal = np.random.randn(5, 21, 1)

    reg.fit(X, Y, internal, train_indices=np.array([0, 1, 2]), val_indices=np.array([3]), test_indices=np.array([4]))
    reg.initialize_memory_variables()

    pred = reg.predict(X[0, 0])
    assert pred.shape == (12,)

    # Test save and load
    loaded = SPILSNetGraph.load(save_file + "_best")
    loaded.initialize_memory_variables()
    pred_loaded = loaded.predict(X[0, 0])
    assert pred_loaded.shape == (12,)
