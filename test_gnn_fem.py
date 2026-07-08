"""
Test script for SPILSNetGNN with FEM mesh connectivity.
Demonstrates:
1. Creating synthetic tetrahedral meshes
2. Instantiating models with different graph types
3. Running forward passes
4. Comparing FEM vs k-NN connectivity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

import sys
sys.path.insert(0, '/Users/andinoboerst/Code/spilsnet-torch')

from spilsnet.models import (
    SPILSNetGNN,
    build_fem_graph_from_elements,
    build_knn_graph,
    build_line_graph,
)
from config_gnn_fem import CONFIG_GNN_FEM_SMALL, CONFIG_GNN_KNN_SMALL, CONFIG_GNN_LINE_SMALL


# ============================================================================
# SYNTHETIC FEM MESH GENERATION
# ============================================================================

def create_tetrahedral_mesh(n_nodes: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple tetrahedral mesh on a regular grid.
    
    Args:
        n_nodes: Approximate number of nodes (will adjust to fit grid)
    
    Returns:
        positions: [num_nodes, 3] - node coordinates
        elements: [num_elements, 4] - tetrahedral connectivity
    """
    # Create a cubic lattice of nodes
    n_per_side = int(np.ceil(n_nodes ** (1/3)))
    x = np.linspace(0, 1, n_per_side)
    y = np.linspace(0, 1, n_per_side)
    z = np.linspace(0, 1, n_per_side)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Create tetrahedral elements from cubic lattice
    # Each unit cube is divided into 6 tetrahedra
    elements = []
    n = n_per_side
    
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                # Node indices of the cube
                n000 = i * n * n + j * n + k
                n100 = (i + 1) * n * n + j * n + k
                n010 = i * n * n + (j + 1) * n + k
                n110 = (i + 1) * n * n + (j + 1) * n + k
                n001 = i * n * n + j * n + (k + 1)
                n101 = (i + 1) * n * n + j * n + (k + 1)
                n011 = i * n * n + (j + 1) * n + (k + 1)
                n111 = (i + 1) * n * n + (j + 1) * n + (k + 1)
                
                # 6 tetrahedra per cube (standard decomposition)
                elements.extend([
                    [n000, n100, n010, n001],
                    [n100, n110, n010, n101],
                    [n010, n110, n011, n101],
                    [n001, n100, n101, n000],
                    [n010, n011, n001, n000],
                    [n100, n010, n101, n001],
                ])
    
    positions = torch.from_numpy(positions).float()
    elements = torch.tensor(elements, dtype=torch.long)
    
    print(f"✓ Created tetrahedral mesh:")
    print(f"  - {positions.shape[0]} nodes")
    print(f"  - {elements.shape[0]} elements")
    print(f"  - Bounding box: [{positions.min():.2f}, {positions.max():.2f}]")
    
    return positions, elements


def create_line_mesh(n_nodes: int = 100, closed: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple line mesh (1D chain of nodes).
    
    Args:
        n_nodes: number of nodes
        closed: if True, connect last node back to first (circular topology)
    
    Returns:
        positions: [num_nodes, 3] - node coordinates along a curve
        connectivity: [num_nodes] - node ordering for line connectivity
    """
    # Create nodes along a parametric curve (helix)
    t = torch.linspace(0, 4 * np.pi, n_nodes)
    x = torch.cos(t)
    y = torch.sin(t)
    z = t / (4 * np.pi)  # Normalized to [0, 1]
    
    positions = torch.stack([x, y, z], dim=1).float()
    
    # Connectivity is just sequential ordering
    connectivity = torch.arange(n_nodes, dtype=torch.long)
    
    print(f"✓ Created line mesh:")
    print(f"  - {n_nodes} nodes in sequence")
    print(f"  - {'Circular' if closed else 'Open'} topology")
    print(f"  - Node degree: max 2 (linear chain)")
    
    return positions, connectivity


def print_model_info(model: nn.Module, name: str, config: dict):
    """Print detailed model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"Graph type: {config.get('graph_type', 'N/A')}")
    if config.get('graph_type') == 'knn':
        print(f"k-neighbors: {config.get('k_neighbors')}")
    print(f"GNN type: {config.get('gnn_type', 'N/A')}")
    print(f"GNN hidden sizes: {config.get('gnn_hidden_sizes')}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 8 / 1e6:.2f} MB (float64)")


def benchmark_forward_pass(
    model: nn.Module,
    x: torch.Tensor,
    internal_state: torch.Tensor,
    n_runs: int = 3
):
    """Benchmark forward pass performance."""
    import time
    
    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(x, internal_state)
    
    # Timing
    times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(x, internal_state)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")


# ============================================================================
# TEST RUNS
# ============================================================================

def test_fem_connectivity():
    """Test 1: FEM mesh with element connectivity."""
    print("\n" + "=" * 70)
    print("TEST 1: FEM ELEMENT CONNECTIVITY")
    print("=" * 70)
    
    # Create synthetic mesh
    positions, elements = create_tetrahedral_mesh(n_nodes=100)
    actual_n_nodes = positions.shape[0]
    
    # Build graph
    edge_index = build_fem_graph_from_elements(positions, elements, connectivity_type="element")
    print(f"\nGraph connectivity:")
    print(f"  - Edges: {edge_index.shape[1]}")
    print(f"  - Avg degree: {2 * edge_index.shape[1] / positions.shape[0]:.1f}")
    
    # Create config with FEM elements and correct input size
    config = CONFIG_GNN_FEM_SMALL.copy()
    config["input_size"] = actual_n_nodes * config["dimension"]  # Adjust to actual mesh size
    config["fem_elements"] = elements
    
    # Create and test model
    model = SPILSNetGNN(config)
    print_model_info(model, "SPILSNetGNN (FEM Element)", config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config["input_size"], dtype=torch.float64)
    internal_state = torch.zeros(batch_size, config["internal_state_size"], dtype=torch.float64)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Internal state shape: {internal_state.shape}")
    
    output, next_internal = model(x, internal_state)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Next internal state shape: {next_internal.shape}")
    print(f"✓ Forward pass successful!")
    
    # Benchmark
    benchmark_forward_pass(model, x, internal_state)


def test_knn_connectivity():
    """Test 2: k-NN graph (dynamic construction)."""
    print("\n" + "=" * 70)
    print("TEST 2: k-NEAREST NEIGHBORS CONNECTIVITY")
    print("=" * 70)
    
    # Create synthetic positions
    positions = torch.randn(100, 3)
    
    # Build graph
    edge_index = build_knn_graph(positions, k=4)
    print(f"\nGraph connectivity (k=4):")
    print(f"  - Edges: {edge_index.shape[1]}")
    print(f"  - Avg degree: {2 * edge_index.shape[1] / positions.shape[0]:.1f}")
    
    # Create config
    config = CONFIG_GNN_KNN_SMALL.copy()
    
    # Create and test model
    model = SPILSNetGNN(config)
    print_model_info(model, "SPILSNetGNN (k-NN)", config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config["input_size"], dtype=torch.float64)
    internal_state = torch.zeros(batch_size, config["internal_state_size"], dtype=torch.float64)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    
    output, next_internal = model(x, internal_state)
    
    print(f"  Output shape: {output.shape}")
    print(f"✓ Forward pass successful!")
    
    # Benchmark
    benchmark_forward_pass(model, x, internal_state)


def test_comparison():
    """Test 3: Detailed comparison between FEM and k-NN."""
    print("\n" + "=" * 70)
    print("TEST 3: FEM vs k-NN COMPARISON")
    print("=" * 70)
    
    positions, elements = create_tetrahedral_mesh(n_nodes=100)
    actual_n_nodes = positions.shape[0]
    
    # Build both graphs
    fem_edge_index = build_fem_graph_from_elements(positions, elements, connectivity_type="element")
    knn_edge_index = build_knn_graph(positions, k=6)
    
    print(f"\nGraph Statistics:")
    print(f"{'Metric':<30} {'FEM Element':<20} {'k-NN (k=6)':<20}")
    print("-" * 70)
    print(f"{'Number of edges':<30} {fem_edge_index.shape[1]:<20} {knn_edge_index.shape[1]:<20}")
    print(f"{'Avg node degree':<30} {2*fem_edge_index.shape[1]/actual_n_nodes:<20.2f} {2*knn_edge_index.shape[1]/actual_n_nodes:<20.2f}")
    print(f"{'Graph density':<30} {2*fem_edge_index.shape[1]/(actual_n_nodes*(actual_n_nodes-1)):<20.4f} {2*knn_edge_index.shape[1]/(actual_n_nodes*(actual_n_nodes-1)):<20.4f}")
    
    # Create configs with corrected input sizes
    config_fem = CONFIG_GNN_FEM_SMALL.copy()
    config_fem["input_size"] = actual_n_nodes * config_fem["dimension"]
    config_fem["fem_elements"] = elements
    
    config_knn = CONFIG_GNN_KNN_SMALL.copy()
    config_knn["input_size"] = actual_n_nodes * config_knn["dimension"]
    
    # Create models
    model_fem = SPILSNetGNN(config_fem)
    model_knn = SPILSNetGNN(config_knn)
    
    # Count parameters
    params_fem = sum(p.numel() for p in model_fem.parameters())
    params_knn = sum(p.numel() for p in model_knn.parameters())
    
    print(f"\n{'Model':<30} {'Parameters':<20} {'Size (MB)':<20}")
    print("-" * 70)
    print(f"{'FEM Element':<30} {params_fem:<20,} {params_fem*8/1e6:<20.2f}")
    print(f"{'k-NN':<30} {params_knn:<20,} {params_knn*8/1e6:<20.2f}")
    
    # Test both models
    x = torch.randn(2, config_fem["input_size"], dtype=torch.float64)
    internal_state = torch.zeros(2, config_fem["internal_state_size"], dtype=torch.float64)
    
    print(f"\nInference Timing:")
    print("-" * 70)
    print("FEM Element model:")
    benchmark_forward_pass(model_fem, x, internal_state, n_runs=3)
    print("\nk-NN model:")
    benchmark_forward_pass(model_knn, x, internal_state, n_runs=3)


def test_batch_processing():
    """Test 4: Batch processing capability."""
    print("\n" + "=" * 70)
    print("TEST 4: BATCH PROCESSING")
    print("=" * 70)
    
    positions, elements = create_tetrahedral_mesh(n_nodes=100)
    actual_n_nodes = positions.shape[0]
    
    config = CONFIG_GNN_FEM_SMALL.copy()
    config["input_size"] = actual_n_nodes * config["dimension"]
    config["fem_elements"] = elements
    
    model = SPILSNetGNN(config)
    model.eval()
    
    print("\nTesting different batch sizes:")
    print(f"{'Batch Size':<15} {'Input Shape':<25} {'Output Shape':<25} {'Status':<20}")
    print("-" * 85)
    
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, config["input_size"], dtype=torch.float64)
        internal_state = torch.zeros(batch_size, config["internal_state_size"], dtype=torch.float64)
        
        try:
            with torch.no_grad():
                output, _ = model(x, internal_state)
            status = "✓ Success"
        except Exception as e:
            status = f"✗ Failed: {str(e)[:15]}"
        
        print(f"{batch_size:<15} {str(x.shape):<25} {str(output.shape):<25} {status:<20}")


def test_line_mesh():
    """Test 5: Line mesh with sequential connectivity."""
    print("\n" + "=" * 70)
    print("TEST 5: LINE MESH CONNECTIVITY")
    print("=" * 70)
    
    # Create line mesh
    positions, connectivity = create_line_mesh(n_nodes=100)
    actual_n_nodes = positions.shape[0]
    
    # Build graph
    edge_index = build_line_graph(positions, connectivity)
    print(f"\nGraph connectivity:")
    print(f"  - Edges: {edge_index.shape[1]}")
    print(f"  - Avg degree: {2 * edge_index.shape[1] / positions.shape[0]:.1f}")
    print(f"  - Expected: ~2.0 (linear chain)")
    
    # Create config with line connectivity
    config = CONFIG_GNN_LINE_SMALL.copy()
    config["input_size"] = actual_n_nodes * config["dimension"]
    config["line_connectivity"] = connectivity
    
    # Create and test model
    model = SPILSNetGNN(config)
    print_model_info(model, "SPILSNetGNN (Line Mesh)", config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config["input_size"], dtype=torch.float64)
    internal_state = torch.zeros(batch_size, config["internal_state_size"], dtype=torch.float64)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Internal state shape: {internal_state.shape}")
    
    output, next_internal = model(x, internal_state)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Next internal state shape: {next_internal.shape}")
    print(f"✓ Forward pass successful!")
    
    # Benchmark
    benchmark_forward_pass(model, x, internal_state)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SPILSNetGNN with FEM Mesh Connectivity - Test Suite")
    print("=" * 70)
    
    # Check PyTorch Geometric installation
    try:
        import torch_geometric
        print(f"\n✓ PyTorch Geometric {torch_geometric.__version__} available")
    except ImportError:
        print("\n✗ PyTorch Geometric not found. Install with:")
        print("  pip install torch-geometric")
        sys.exit(1)
    
    # Run tests
    test_fem_connectivity()
    test_knn_connectivity()
    test_comparison()
    test_batch_processing()
    test_line_mesh()
    
    print("\n" + "=" * 70)
    print("  All tests completed!")
    print("=" * 70 + "\n")
