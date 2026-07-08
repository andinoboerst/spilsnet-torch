"""
Example configuration for SPILSNetGNN with FEM mesh connectivity.
Demonstrates three graph construction strategies:
1. FEM element connectivity (all nodes in element connected)
2. k-nearest neighbors (distance-based)
3. FEM face connectivity (only face nodes connected)
"""

import torch

# ============================================================================
# BASE CONFIG (Common to all variants)
# ============================================================================

BASE_CONFIG = {
    # Input/output specifications
    "input_size": 1200,          # 400 nodes * 3D coordinates
    "dimension": 3,              # 3D spatial coordinates
    
    # Internal state
    "internal_state_size": 128,
    "internal_input_mlp": [64, 64],
    "internal_output_mlp": [64, 64],
    
    # Bottleneck (latent representation)
    "bottleneck_pool_size": 16,
    "latent_dim": 64,
    "latent_encoder_mlp": [128, 128],
    
    # Physics core (GRU)
    "gru_hidden_size": 128,
    "gru_layers": 2,
    
    # Decoder
    "latent_decoder_structure": [256, 512],
    
    # Regularization
    "dropout_rate": 0.1,
    
    # Output smoothing
    "smoothing_kernel_size": 5,
    
    # Data type
    "dtype": "float64",
    
    # Skip connections
    "skip_target_nodes": 8,
}


# ============================================================================
# VARIANT 1: FEM ELEMENT CONNECTIVITY
# ============================================================================

CONFIG_GNN_FEM_ELEMENT = {
    **BASE_CONFIG,
    
    # Graph construction from FEM mesh
    "graph_type": "fem_element",  # Options: "fem_element", "fem_face", "fem_edge", "knn"
    
    # FEM elements (tetrahedral mesh with 400 nodes -> ~650 elements)
    # This should be provided at runtime, but we show the shape here
    # "fem_elements": torch.tensor([...]),  # Shape: [num_elements, 4]
    
    # GNN encoder configuration
    "gnn_hidden_sizes": [64, 128, 256],
    "gnn_type": "gcn",            # Options: "gat", "gcn", "sage"
    "gnn_heads": 4,               # Only for GAT (if used)
}


# ============================================================================
# VARIANT 2: k-NEAREST NEIGHBORS (DYNAMIC)
# ============================================================================

CONFIG_GNN_KNN = {
    **BASE_CONFIG,
    
    # Dynamic k-NN graph construction
    "graph_type": "knn",
    "k_neighbors": 6,             # Each node connects to 6 nearest neighbors
    
    # GNN encoder configuration
    "gnn_hidden_sizes": [64, 128, 256],
    "gnn_type": "gcn",
    "gnn_heads": 4,
}


# ============================================================================
# VARIANT 3: FEM FACE CONNECTIVITY (SPARSER)
# ============================================================================

CONFIG_GNN_FEM_FACE = {
    **BASE_CONFIG,
    
    # Graph from FEM faces (fewer connections than element)
    "graph_type": "fem_face",
    
    # GNN encoder configuration
    "gnn_hidden_sizes": [64, 128, 256],
    "gnn_type": "gcn",
    "gnn_heads": 4,
}


# ============================================================================
# VARIANT 4: LINE MESH (1D CHAIN)
# ============================================================================

CONFIG_GNN_LINE = {
    **BASE_CONFIG,
    
    # Line mesh: each node connects to at most 2 neighbors
    "graph_type": "line",
    
    # Optional: specify connection order. If None, sorts by first coordinate
    # "line_connectivity": torch.tensor([0, 1, 2, 3, ...]),
    
    # GNN encoder configuration
    "gnn_hidden_sizes": [64, 128, 256],
    "gnn_type": "gcn",
    "gnn_heads": 4,
}


# ============================================================================
# LIGHTWEIGHT VARIANTS FOR TESTING
# ============================================================================

CONFIG_GNN_FEM_SMALL = {
    **BASE_CONFIG,
    
    "input_size": 300,              # 100 nodes * 3D
    
    "bottleneck_pool_size": 8,
    "latent_dim": 32,
    "latent_encoder_mlp": [64],
    
    "gru_hidden_size": 64,
    
    "latent_decoder_structure": [128],
    
    "graph_type": "fem_element",
    "gnn_hidden_sizes": [32, 64],
    "gnn_type": "gcn",
    "gnn_heads": 4,
}


CONFIG_GNN_KNN_SMALL = {
    **BASE_CONFIG,
    
    "input_size": 300,
    
    "bottleneck_pool_size": 8,
    "latent_dim": 32,
    "latent_encoder_mlp": [64],
    
    "gru_hidden_size": 64,
    
    "latent_decoder_structure": [128],
    
    "graph_type": "knn",
    "k_neighbors": 4,
    "gnn_hidden_sizes": [32, 64],
    "gnn_type": "gcn",
    "gnn_heads": 4,
}


CONFIG_GNN_LINE_SMALL = {
    **BASE_CONFIG,
    
    "input_size": 300,
    
    "bottleneck_pool_size": 8,
    "latent_dim": 32,
    "latent_encoder_mlp": [64],
    
    "gru_hidden_size": 64,
    
    "latent_decoder_structure": [128],
    
    "graph_type": "line",
    "gnn_hidden_sizes": [32, 64],
    "gnn_type": "gcn",
    "gnn_heads": 4,
}
