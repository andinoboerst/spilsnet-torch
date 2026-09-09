import numpy as np
import pytest
import torch
import torch.nn as nn
from spilsnet.utils import spils_loss, spils_loss_3d, connectivity_to_edge_index
from spilsnet import SPILSNet, SPILSNetGraph, MinMaxScaler


def test_spils_loss_1d_original():
    """
    Test 1D spils_loss using finite-difference stencil.
    """
    torch.manual_seed(42)
    batch_size = 2
    n_nodes = 5
    dim = 2

    y_pred = torch.randn(batch_size, n_nodes * dim, dtype=torch.float32)
    y_target = torch.randn(batch_size, n_nodes * dim, dtype=torch.float32)
    i_pred = torch.randn(batch_size, 1, dtype=torch.float32)
    i_target = torch.randn(batch_size, 1, dtype=torch.float32)

    loss = spils_loss(
        y_t_pred=y_pred,
        y_t_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=n_nodes,
        dimension=dim,
        alpha=0.7,
        beta=0.3,
        gamma=0.2,
    )

    err = (y_pred - y_target).view(batch_size, n_nodes, dim)
    lap1d = err[:, 0:-2, :] - (2 * err[:, 1:-1, :]) + err[:, 2:, :]
    expected_s = torch.mean(lap1d**2)
    expected_f = nn.functional.mse_loss(y_pred, y_target)
    expected_i = nn.functional.mse_loss(i_pred, i_target)
    expected_total = 0.7 * expected_f + 0.3 * expected_i + 0.2 * expected_s

    assert torch.allclose(loss, expected_total, atol=1e-6)


def test_spils_loss_3d_mode_1_uniform():
    """
    Test 3D Mode 1: Uniform mesh (pos is None) with edge_index.
    Verifies vectorized Graph Laplacian L(e_i) = sum_{j in N(i)} (e_j - e_i).
    """
    torch.manual_seed(42)
    batch_size = 3
    n_nodes = 4
    dim = 3

    y_pred = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64, requires_grad=True)
    y_target = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64)
    i_pred = torch.randn(batch_size, 2, dtype=torch.float64, requires_grad=True)
    i_target = torch.randn(batch_size, 2, dtype=torch.float64)

    # Simple undirected chain: 0-1-2-3
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)

    alpha, beta, gamma = 0.5, 0.2, 0.3
    loss = spils_loss_3d(
        y_t_pred=y_pred,
        y_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=n_nodes,
        dimension=dim,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        edge_index=edge_index,
        pos=None,
    )

    # Manual analytical calculation
    err = (y_pred - y_target).view(batch_size, n_nodes, dim)
    expected_laplacian = torch.zeros_like(err)
    expected_laplacian[:, 0, :] = err[:, 1, :] - err[:, 0, :]
    expected_laplacian[:, 1, :] = (err[:, 0, :] - err[:, 1, :]) + (err[:, 2, :] - err[:, 1, :])
    expected_laplacian[:, 2, :] = (err[:, 1, :] - err[:, 2, :]) + (err[:, 3, :] - err[:, 2, :])
    expected_laplacian[:, 3, :] = err[:, 2, :] - err[:, 3, :]

    expected_loss_s = torch.mean(expected_laplacian**2)
    expected_loss_F = nn.functional.mse_loss(y_pred, y_target)
    expected_loss_i = nn.functional.mse_loss(i_pred, i_target)
    expected_total = (alpha * expected_loss_F) + (beta * expected_loss_i) + (gamma * expected_loss_s)

    assert torch.allclose(loss, expected_total, atol=1e-10)

    loss.backward()
    assert y_pred.grad is not None
    assert torch.all(torch.isfinite(y_pred.grad))


def test_spils_loss_3d_mode_2_static_pos():
    """
    Test 3D Mode 2: Non-uniform mesh with static pos [n_nodes, 3].
    Verifies distance-normalized surface Laplacian:
    w_ij = 1 / (||p_i - p_j||_2 + eps)
    L_w(e_i) = (1 / sum w_ij) * sum w_ij * (e_j - e_i)
    """
    torch.manual_seed(42)
    batch_size = 2
    n_nodes = 4
    dim = 3
    eps = 1e-8

    y_pred = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64, requires_grad=True)
    y_target = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64)
    i_pred = torch.randn(batch_size, 1, dtype=torch.float64)
    i_target = torch.randn(batch_size, 1, dtype=torch.float64)

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)

    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ], dtype=torch.float64)

    loss = spils_loss_3d(
        y_t_pred=y_pred,
        y_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=n_nodes,
        dimension=dim,
        alpha=0.8,
        beta=0.1,
        gamma=0.5,
        edge_index=edge_index,
        pos=pos,
    )

    err = (y_pred - y_target).view(batch_size, n_nodes, dim)
    expected_lap = torch.zeros_like(err)

    d01 = 1.0 + eps
    d12 = 2.0 + eps
    d23 = 3.0 + eps
    w01, w12, w23 = 1.0 / d01, 1.0 / d12, 1.0 / d23

    expected_lap[:, 0, :] = err[:, 1, :] - err[:, 0, :]
    expected_lap[:, 1, :] = (w01 * (err[:, 0, :] - err[:, 1, :]) + w12 * (err[:, 2, :] - err[:, 1, :])) / (w01 + w12)
    expected_lap[:, 2, :] = (w12 * (err[:, 1, :] - err[:, 2, :]) + w23 * (err[:, 3, :] - err[:, 2, :])) / (w12 + w23)
    expected_lap[:, 3, :] = err[:, 2, :] - err[:, 3, :]

    expected_loss_s = torch.mean(expected_lap**2)
    expected_loss_F = nn.functional.mse_loss(y_pred, y_target)
    expected_loss_i = nn.functional.mse_loss(i_pred, i_target)
    expected_total = (0.8 * expected_loss_F) + (0.1 * expected_loss_i) + (0.5 * expected_loss_s)

    assert torch.allclose(loss, expected_total, atol=1e-10)

    loss.backward()
    assert y_pred.grad is not None
    assert torch.all(torch.isfinite(y_pred.grad))


def test_spils_loss_3d_mode_2_batched_pos():
    """
    Test 3D Mode 2: Non-uniform mesh with batched pos [Batch, N_nodes, 3].
    """
    torch.manual_seed(42)
    batch_size = 2
    n_nodes = 3
    dim = 3
    eps = 1e-8

    y_pred = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64, requires_grad=True)
    y_target = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64)
    i_pred = torch.randn(batch_size, 1, dtype=torch.float64)
    i_target = torch.randn(batch_size, 1, dtype=torch.float64)

    # Triangle 0-1-2
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],
        [1, 0, 2, 1, 0, 2]
    ], dtype=torch.long)

    pos_batched = torch.randn(batch_size, n_nodes, 3, dtype=torch.float64)

    loss = spils_loss_3d(
        y_t_pred=y_pred,
        y_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=n_nodes,
        dimension=dim,
        alpha=1.0,
        beta=0.0,
        gamma=0.5,
        edge_index=edge_index,
        pos=pos_batched,
    )

    err = (y_pred - y_target).view(batch_size, n_nodes, dim)
    expected_lap = torch.zeros_like(err)

    for b in range(batch_size):
        p = pos_batched[b]
        for i in range(n_nodes):
            w_sum = 0.0
            accum = torch.zeros(dim, dtype=torch.float64)
            for j in [(i - 1) % n_nodes, (i + 1) % n_nodes]:
                d = torch.linalg.vector_norm(p[i] - p[j]).item() + eps
                w = 1.0 / d
                w_sum += w
                accum += w * (err[b, j] - err[b, i])
            expected_lap[b, i] = accum / w_sum

    expected_loss_s = torch.mean(expected_lap**2)
    expected_total = nn.functional.mse_loss(y_pred, y_target) + 0.5 * expected_loss_s

    assert torch.allclose(loss, expected_total, atol=1e-10)


def test_spils_loss_3d_gamma_zero():
    """
    When gamma=0, smoothness loss is 0.0 without edge/stencil calculation overhead.
    """
    y_pred = torch.randn(4, 12)
    y_target = torch.randn(4, 12)
    i_pred = torch.randn(4, 2)
    i_target = torch.randn(4, 2)

    loss = spils_loss_3d(
        y_t_pred=y_pred,
        y_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=4,
        dimension=3,
        alpha=1.0,
        beta=0.5,
        gamma=0.0,
    )

    expected = nn.functional.mse_loss(y_pred, y_target) + 0.5 * nn.functional.mse_loss(i_pred, i_target)
    assert torch.allclose(loss, expected)


def test_spils_loss_3d_legacy_keyword_compatibility():
    """
    Test backward compatibility with y_t_target legacy parameter name in spils_loss_3d.
    """
    y_pred = torch.randn(2, 6)
    y_target = torch.randn(2, 6)
    i_pred = torch.randn(2, 1)
    i_target = torch.randn(2, 1)

    loss = spils_loss_3d(
        y_t_pred=y_pred,
        y_t_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=2,
        dimension=3,
        gamma=0.0,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_spils_loss_3d_isolated_nodes():
    """
    Test numerical stability when mesh contains isolated nodes (degree 0).
    """
    batch_size = 2
    n_nodes = 4
    dim = 3

    y_pred = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64)
    y_target = torch.randn(batch_size, n_nodes * dim, dtype=torch.float64)
    i_pred = torch.randn(batch_size, 1, dtype=torch.float64)
    i_target = torch.randn(batch_size, 1, dtype=torch.float64)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    pos = torch.randn(n_nodes, 3, dtype=torch.float64)

    loss = spils_loss_3d(
        y_t_pred=y_pred,
        y_target=y_target,
        i_t_pred=i_pred,
        i_t_target=i_target,
        n_nodes=n_nodes,
        dimension=dim,
        gamma=0.5,
        edge_index=edge_index,
        pos=pos,
    )

    assert torch.isfinite(loss)


def test_spilsnet_graph_training_with_spils_loss_3d(tmp_path):
    """
    Test SPILSNetGraph wrapper integration: uses spils_loss_3d with edge_index and node_coordinates.
    """
    connectivity = np.array([[0, 1, 2], [1, 2, 3]])
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    config = {
        "dimension": 3,
        "input_size": 12,
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
        "batch_size": 5,
        "num_epochs": 2,
        "weight_decay": 0.0,
        "early_stop_patience": 5,
        "loss_alpha": 1.0,
        "loss_beta": 0.1,
        "loss_gamma": 0.5,
    }

    reg = SPILSNetGraph(
        connectivity=connectivity,
        node_coordinates=node_coords,
        save_path=str(tmp_path / "gnn_smooth_model"),
        model_config=config,
        hyperparameters=hyperparams,
        input_scaler_class=MinMaxScaler(),
        internal_in_scaler_class=MinMaxScaler(),
        internal_out_scaler_class=MinMaxScaler(),
        output_scaler_class=MinMaxScaler(),
    )

    assert reg.loss_fn is spils_loss_3d

    X = np.random.randn(4, 10, 12)
    Y = np.random.randn(4, 10, 12)
    internal = np.random.randn(4, 11, 1)

    reg.fit(X, Y, internal, train_indices=np.array([0, 1]), val_indices=np.array([2]), test_indices=np.array([3]))
    reg.initialize_memory_variables()

    pred = reg.predict(X[0, 0])
    assert pred.shape == (12,)
    assert np.all(np.isfinite(pred))
