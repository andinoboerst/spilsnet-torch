import torch
from spilsnet import SPILSNetCore


test_config = {
    "dimension": 2,
    "input_size": 42,
    "internal_state_size": 1,

    # --- 1. The "Aggressive" Encoder ---
    # We stride heavily (s=2) immediately to throw away local details.
    "encoder_structure": [
        # 21 -> 11 nodes.
        # Kernel 7 catches 33% of the beam in the first layer.
        {"k": 3, "s": 2, "p": 2, "out": 4},  # 21 -> 11 nodes
        {"k": 3, "s": 2, "p": 2, "out": 5},  # 11 -> 5 nodes
    ],

    "bottleneck_pool_size": 1,
    "skip_target_nodes": 5,

    # --- 2. The "Choke Point" ---
    # Latent Dim = 4.
    # This is the most critical change. It restricts the physics to
    # only 4 degrees of freedom (e.g., x-trans, y-trans, rotation, bend).
    # Noise requires high dimensions to exist; this kills it.
    "latent_dim": 16,

    # Keep this simple
    "latent_encoder_mlp": [8, 16],

    # --- 3. The Dynamics ---
    "gru_hidden_size": 16,
    "gru_layers": 1,

    "internal_input_mlp": [8],
    "internal_output_mlp": [16, 8],

    # --- 4. The "Linear" Decoder ---
    # EMPTY LIST [] means NO HIDDEN LAYERS.
    # It becomes a single Linear matrix: [Latent(4) + Pooled(16) -> Output(42)].
    # A single matrix multiplication is perfectly smooth.
    # It cannot create "kinks" or local spikes because it lacks ReLUs.
    "latent_decoder_structure": [32, 16],

    # --- 5. The "Sledgehammer" Smoothing ---
    # Kernel 9 covers ~45% of the mesh.
    # Node 10 is now the average of Nodes 6 through 14.
    # Independent oscillation is mathematically impossible.
    "smoothing_kernel_size": 3,

    "dropout_rate": 0.2,
}

# Define model parameters
model = SPILSNetCore(test_config)

# Create dummy input
batch_size = 32
x_in = torch.randn(batch_size, 42, dtype=torch.float64)
internal_state = torch.randn(batch_size, 1, dtype=torch.float64)

print(x_in.shape)

# Forward pass
output, next_internal_state = model(x_in, internal_state)

print("Output shape:", output.shape)
print("Next internal state shape:", next_internal_state.shape)
