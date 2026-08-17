import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

try:
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from spilsnet.utils import build_mlp

class SPILSNetGraphCore(nn.Module):
    """
    Graph Neural Network based PyTorch implementation of the SPILSNet architecture 
    for unstructured meshes (e.g. 2D interface in 3D domain).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for SPILSNetGraphCore")

        self.n_nodes = config["input_size"] // config["dimension"]
        self.dim = config["dimension"]
        self.drop_p = config.get("dropout_rate", 0.0)

        dtype_str = config.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_str == "float64" else torch.float32

        # --- 1. ENCODER (Graph Convolutions) ---
        self.encoder_stack = nn.ModuleList()
        current_in = self.dim
        
        # E.g. [{"out": 32}, {"out": 64}]
        for layer_cfg in config.get("encoder_structure", [{"out": 32}, {"out": 64}]):
            block = nn.ModuleList([
                SAGEConv(current_in, layer_cfg["out"]),
                nn.Tanh(),
                nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
            ])
            self.encoder_stack.append(block)
            current_in = layer_cfg["out"]

        self.skip_target_nodes = config.get("skip_target_nodes", 3)

        # Learned Spatial Downsampler
        self.spatial_downsampler = nn.Sequential(
            nn.Linear(self.n_nodes, self.skip_target_nodes, dtype=self.dtype),
            nn.Tanh(),
        )

        skip_connection_size = current_in * self.skip_target_nodes

        # --- 2. BOTTLENECK (The Brain) ---
        latent_dim = config["latent_dim"]
        self.gru_hidden = config["gru_hidden_size"]
        self.gru_layers = config.get("gru_layers", 1)

        # Map Spatial Features -> GRU Input
        self.latent_enc = build_mlp(current_in, config["latent_encoder_mlp"], latent_dim, drop_p=0.0, dtype=self.dtype)

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

    def forward(self, x_in: torch.Tensor, edge_index: torch.Tensor, internal_state: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using PyTorch Geometric.

        Args:
            x_in: Node features of shape [Batch * Nodes, Dim] or [Batch, Nodes*Dim]. 
                  If [Batch, Nodes*Dim], will be reshaped.
            edge_index: Graph connectivity [2, Num_Edges].
            internal_state: Physics core hidden states [Batch, InternalStateSize].
            batch: Batch vector [Batch * Nodes]. 

        Returns:
            out_flat: Predicted nodal vector [Batch, Nodes*Dim]
            internal_next: Next internal state [Batch, InternalStateSize]
        """
        batch_size = internal_state.size(0)
        
        # Reshape [Batch, Nodes*Dim] -> [Batch * Nodes, Dim]
        if x_in.dim() == 2 and x_in.size(1) == self.n_nodes * self.dim:
            x = x_in.view(batch_size * self.n_nodes, self.dim)
            if batch is None:
                # Create a simple batch index if not provided
                batch = torch.arange(batch_size, device=x.device).repeat_interleave(self.n_nodes)
        else:
            x = x_in
            if batch is None:
                raise ValueError("batch tensor must be provided if x_in is not [Batch, Nodes*Dim]")

        # 1. GNN Encoder Pass
        curr = x
        for conv, act, drop in self.encoder_stack:
            curr = conv(curr, edge_index)
            curr = act(curr)
            curr = drop(curr)

        # 2. Bottleneck
        # Global mean pool across nodes in each graph -> [Batch, current_in]
        pooled = global_mean_pool(curr, batch)
        gru_input = self.latent_enc(pooled).unsqueeze(0)  # [1, Batch, latent_dim]

        # Learned Skip Connection
        # Reshape features to [Batch, current_in, Nodes]
        curr_reshaped = curr.view(batch_size, self.n_nodes, -1).permute(0, 2, 1)
        learned_skip = self.spatial_downsampler(curr_reshaped)  # [Batch, current_in, skip_target_nodes]
        skip_connection = learned_skip.flatten(1)  # [Batch, current_in * skip_target_nodes]

        # 3. GRU Initialization & Step
        h_flat = torch.tanh(self.internal_in(internal_state))
        h_0 = h_flat.view(batch_size, self.gru_layers, self.gru_hidden).permute(1, 0, 2).contiguous()

        _, h_n = self.gru(gru_input, h_0)
        h_last = h_n[-1]

        internal_next = self.internal_out(h_last)

        # 4. Global Projection
        global_input = torch.cat([h_last, skip_connection], dim=1)
        
        # Output [Batch, Nodes * Dim]
        raw_force = self.latent_decoder(global_input)

        return raw_force, internal_next
