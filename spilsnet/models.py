import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from spilsnet.utils import build_mlp


class SPILSNetCore(nn.Module):
    """
    Core PyTorch implementation of the SPILSNet architecture.

    SPILSNet (Structure-Preserving Input-Output Learning System Network) consists of:
    1. An encoder stack (The Eye) for spatial feature extraction.
    2. A learned spatial downsampler for skip connections.
    3. An AdaptiveAvgPool1d bottleneck (The Brain).
    4. A physics core using GRU for temporal dynamics.
    5. A global decoder forodal vector projection.
    6. A smoothing layer for noise reduction.

    Attributes:
        n_nodes (int): Number of spatial nodes in the input.
        dim (int): Dimension of each node (e.g., 2 for 2D coordinates).
        drop_p (float): Dropout probability.
        dtype (torch.dtype): Data type for model parameters and computations.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SPILSNetCore model.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model hyperparameters.
                Required keys:
                - "input_size" (int): Total size of the input nodal vector (n_nodes * dimension).
                - "dimension" (int): Dimension of each node.
                - "encoder_structure" (List[Dict[str, Any]]): List of conv layer parameters (out, k, s, p).
                - "bottleneck_pool_size" (int): Size of the adaptive average pooling.
                - "latent_dim" (int): Dimension of the latent representation.
                - "gru_hidden_size" (int): Size of the GRU hidden state.
                - "latent_encoder_mlp" (List[int]): Hidden sizes for encoder MLP.
                - "internal_state_size" (int): Size of the internal states.
                - "internal_input_mlp" (List[int]): Hidden sizes for internal input MLP.
                - "internal_output_mlp" (List[int]): Hidden sizes for internal output MLP.
                Optional keys:
                - "dropout_rate" (float): Dropout probability. Defaults to 0.0.
                - "skip_target_nodes" (int): Number of nodes for skip connection. Defaults to 3.
                - "gru_layers" (int): Number of GRU layers. Defaults to 1.
                - "latent_decoder_structure" (List[int]): Hidden sizes for decoder MLP. Defaults to [512, 1024].
                - "smoothing_kernel_size" (int): Kernel size for smoothing convolution. Defaults to 3.
                - "dtype" (str): Data type ('float32' or 'float64'). Defaults to 'float64'.
        """
        super().__init__()

        self.n_nodes = config["input_size"] // config["dimension"]
        self.dim = config["dimension"]
        self.drop_p = config.get("dropout_rate", 0.0)

        dtype_str = config.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_str == "float64" else torch.float32

        # --- 1. ENCODER (The Eye) ---
        self.encoder_stack = nn.ModuleList()
        current_in = self.dim

        for layer_cfg in config["encoder_structure"]:
            block = nn.Sequential(
                nn.Conv1d(
                    current_in,
                    layer_cfg["out"],
                    kernel_size=layer_cfg["k"],
                    stride=layer_cfg["s"],
                    padding=layer_cfg["p"],
                    padding_mode="replicate",
                    dtype=self.dtype,
                ),
                nn.Tanh(),
                nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
            )
            self.encoder_stack.append(block)
            current_in = layer_cfg["out"]

        # --- DYNAMIC SIZE DETECTION (The Dummy Pass Trick) ---
        dummy_input = torch.zeros(1, self.dim, self.n_nodes, dtype=self.dtype)

        with torch.no_grad():
            dummy_out = dummy_input
            for layer in self.encoder_stack:
                dummy_out = layer(dummy_out)

        spatial_nodes_out = dummy_out.size(2)

        self.skip_target_nodes = config.get("skip_target_nodes", 3)

        # The Learned Spatial Downsampler!
        self.spatial_downsampler = nn.Sequential(
            nn.Linear(spatial_nodes_out, self.skip_target_nodes, dtype=self.dtype),
            nn.Tanh(),
        )

        skip_connection_size = current_in * self.skip_target_nodes

        # --- 2. BOTTLENECK (The Brain) ---
        self.pool_size = config["bottleneck_pool_size"]
        self.pooling_layer = nn.AdaptiveAvgPool1d(self.pool_size)

        last_layer_out = config["encoder_structure"][-1]["out"]
        flat_size = last_layer_out * self.pool_size

        latent_dim = config["latent_dim"]
        self.gru_hidden = config["gru_hidden_size"]
        self.gru_layers = config.get("gru_layers", 1)

        # Map Spatial Features -> GRU Input
        self.latent_enc = build_mlp(flat_size, config["latent_encoder_mlp"], latent_dim, drop_p=0.0, dtype=self.dtype)

        # --- 3. PHYSICS CORE (GRU) ---
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            dropout=0.0,
            dtype=self.dtype,
        )

        # Internal State Handling
        self.total_hidden_params = self.gru_layers * self.gru_hidden
        self.internal_in = build_mlp(
            config["internal_state_size"], config["internal_input_mlp"], self.total_hidden_params, drop_p=0.0, dtype=self.dtype
        )
        self.internal_out = build_mlp(
            self.gru_hidden, config["internal_output_mlp"], config["internal_state_size"], drop_p=0.0, dtype=self.dtype
        )

        # --- 4. GLOBAL DECODER (The Projector) ---
        global_in_size = self.gru_hidden + skip_connection_size

        self.latent_decoder = build_mlp(
            in_size=global_in_size,
            hidden_sizes=config.get("latent_decoder_structure", [512, 1024]),
            out_size=self.n_nodes * self.dim,
            drop_p=self.drop_p,
            dtype=self.dtype,
        )

        # --- 5. SMOOTHING LAYER ---
        self.smoothing_layer = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=config.get("smoothing_kernel_size", 3),
            padding="same",
            padding_mode="replicate",
            dtype=self.dtype,
        )

    def forward(self, x_in: torch.Tensor, internal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SPILSNetCore model.

        Args:
            x_in (torch.Tensor): Input nodal vector of shape [Batch, Nodes * Dim].
            internal_state (torch.Tensor): Internal state of shape [Batch, InternalStateSize].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - out_flat (torch.Tensor): Predicted nodal vector of shape [Batch, Nodes * Dim].
                - internal_next (torch.Tensor): Next internal state of shape [Batch, InternalStateSize].
        """
        # 1. Reshape Input: [Batch, Nodes*Dim] -> [Batch, Dim, Nodes]
        x = x_in.view(-1, self.n_nodes, self.dim).permute(0, 2, 1)
        batch_size = x.size(0)

        # 2. Encoder Pass
        curr = x
        for layer in self.encoder_stack:
            curr = layer(curr)

        # 3. Bottleneck
        pooled = self.pooling_layer(curr).flatten(1)
        gru_input = self.latent_enc(pooled).unsqueeze(0)

        learned_skip = self.spatial_downsampler(curr)

        # Flatten to [Batch, Channels * skip_target_nodes]
        skip_connection = learned_skip.flatten(1)

        # 4. GRU Initialization
        h_flat = torch.tanh(self.internal_in(internal_state))
        h_0 = h_flat.view(batch_size, self.gru_layers, self.gru_hidden).permute(1, 0, 2).contiguous()

        # 5. Physics Step
        _, h_n = self.gru(gru_input, h_0)
        h_last = h_n[-1]  # The state of the top layer

        internal_next = self.internal_out(h_last)

        # 6. Global Projection (The Skip Connection)
        global_input = torch.cat([h_last, skip_connection], dim=1)

        # MLP Output: [Batch, Nodes * Dim]
        raw_force = self.latent_decoder(global_input)

        # 1. View as [Batch, Nodes, Dim] -> Restores (x,y) pairs
        # 2. Permute -> [Batch, Dim, Nodes] -> Ready for Conv1d
        force_spatial = raw_force.view(batch_size, self.n_nodes, self.dim).permute(0, 2, 1)

        # 7. Smoothing Pass
        final_spatial = self.smoothing_layer(force_spatial)

        # 8. Flatten for Output
        out_flat = final_spatial.permute(0, 2, 1).reshape(x_in.shape)

        return out_flat, internal_next
