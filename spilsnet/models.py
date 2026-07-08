import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import DenseSAGEConv
from torch_geometric.utils import to_dense_adj

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


class GraphSPILSNetCore(nn.Module):
    """
    GNN PyTorch implementation of the SPILSNet architecture for fixed unstructured meshes.

    Attributes:
        n_nodes (int): Number of spatial nodes in the input.
        dim (int): Dimension of each node (e.g., 3 for 3D coordinates).
        edge_index (torch.Tensor): The connectivity of the fixed interface mesh [2, E].
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.n_nodes = config["input_size"] // config["dimension"]
        self.dim = config["dimension"]
        self.drop_p = config.get("dropout_rate", 0.0)

        dtype_str = config.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_str == "float64" else torch.float32

        self.edge_index = torch.tensor(config.get("edge_index", [[0, 1], [1, 0]]))

        # Register the fixed mesh topology as a buffer (moves to GPU automatically)
        # Expected shape: [2, Num_Edges]
        self.register_buffer("spilsnet_edge_index", self.edge_index.to(self.dtype).long())

        # --- 1. ENCODER (The Eye) ---
        # Replacing Conv1d with SAGEConv (Graph Sample and Aggregate)
        self.encoder_stack = nn.ModuleList()
        current_in = self.dim

        for layer_cfg in config["encoder_structure"]:
            block = nn.ModuleDict({
                "conv": SAGEConv(current_in, layer_cfg["out"]).to(self.dtype),
                "act": nn.Tanh(),
                "drop": nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity()
            })
            self.encoder_stack.append(block)
            current_in = layer_cfg["out"]

        # --- 2. BOTTLENECK (The Brain) & SKIP CONNECTION ---
        # Instead of AdaptiveAvgPool1d, we use global graph pooling.
        # To create a rich skip connection equivalent to your spatial downsampler,
        # we can concatenate both the Mean and Max pool of the entire graph.

        flat_size = current_in  # Size after global pooling
        skip_connection_size = current_in * 2  # Mean pool + Max pool

        last_out_channels = config["encoder_structure"][-1]["out"]
        self.skip_target_nodes = config["skip_target_nodes"]

        # 1. The Localized Skip Connection Downsampler (21 nodes -> 5 nodes)
        self.spatial_downsampler = nn.Sequential(
            nn.Linear(self.n_nodes, self.skip_target_nodes, dtype=self.dtype),
            nn.Tanh(),
        )

        # The flat size of the skip connection is now exactly what you had before:
        skip_connection_size = last_out_channels * self.skip_target_nodes

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

        self.total_hidden_params = self.gru_layers * self.gru_hidden
        self.internal_in = build_mlp(
            config["internal_state_size"], config["internal_input_mlp"], self.total_hidden_params, drop_p=0.0, dtype=self.dtype
        )
        self.internal_out = build_mlp(
            self.gru_hidden, config["internal_output_mlp"], config["internal_state_size"], drop_p=0.0, dtype=self.dtype
        )

        # --- 4. GLOBAL DECODER (The Projector) ---
        # This Node-wise MLP maps the broadcasted global state back to forces

        # The global state size (GRU hidden + global skip connection)
        global_in_size = self.gru_hidden + skip_connection_size
        self.latent_decoder = build_mlp(
            in_size=global_in_size,
            hidden_sizes=config.get("latent_decoder_structure", [32, 16]),
            out_size=self.n_nodes * self.dim,  # <-- CRITICAL: Back to full spatial output
            drop_p=self.drop_p,
            dtype=self.dtype,
        )

        # --- 5. SMOOTHING LAYER ---
        # 1 message passing layer to enforce physical continuity across the interface
        self.smoothing_layer = SAGEConv(self.dim, self.dim).to(self.dtype)

    def forward(self, x_in: torch.Tensor, internal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_in.size(0)
        device = x_in.device

        # --- GRAPH BATCHING PREP ---
        x = x_in.view(batch_size, self.n_nodes, self.dim).view(-1, self.dim)
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(self.n_nodes)

        num_edges = self.edge_index.size(1)
        batched_edge_index = self.edge_index.repeat(1, batch_size)
        edge_offsets = (torch.arange(batch_size, device=device) * self.n_nodes).repeat_interleave(num_edges)
        batched_edge_index = batched_edge_index + edge_offsets.unsqueeze(0)

        # --- 1. ENCODER PASS ---
        curr = x
        for layer in self.encoder_stack:
            curr = layer["conv"](curr, batched_edge_index)
            curr = layer["act"](curr)
            curr = layer["drop"](curr)

        # --- 2. BOTTLENECK & SKIP CONNECTION ---
        # PATH A: GLOBAL SUMMARY FOR GRU
        pooled_mean = global_mean_pool(curr, batch_idx)
        gru_input = self.latent_enc(pooled_mean).unsqueeze(0)

        # PATH B: LOCALIZED COARSE SKIP CONNECTION
        curr_reshaped = curr.view(batch_size, self.n_nodes, -1).permute(0, 2, 1)
        learned_skip = self.spatial_downsampler(curr_reshaped)
        skip_connection = learned_skip.flatten(1)

        # --- 3. PHYSICS STEP (GRU) ---
        h_flat = torch.tanh(self.internal_in(internal_state))
        h_0 = h_flat.view(batch_size, self.gru_layers, self.gru_hidden).permute(1, 0, 2).contiguous()

        _, h_n = self.gru(gru_input, h_0)
        h_last = h_n[-1]
        internal_next = self.internal_out(h_last)

        # --- 4. GLOBAL DECODER ---
        # Combine GRU memory with the coarse spatial skip connection
        global_state = torch.cat([h_last, skip_connection], dim=1)  # [Batch, Global_Dim]

        # The MLP maps the global state directly to the full Nodal Force Vector!
        raw_force_flat = self.latent_decoder(global_state)  # [Batch, Nodes * Dim]

        # --- 5. SMOOTHING PASS ---
        # Reshape the flat force vector into a PyG graph format: [Batch * Nodes, Dim]
        raw_force_graph = raw_force_flat.view(-1, self.dim)

        # Smooth the predictions using the physical connectivity of the interface
        final_spatial = self.smoothing_layer(raw_force_graph, batched_edge_index)

        # --- 6. FORMAT OUTPUT ---
        out_flat = final_spatial.view(batch_size, self.n_nodes * self.dim)

        return out_flat, internal_next


class GraphUNetSPILSNetCore(nn.Module):
    """
    Graph U-Net variant of SPILSNetCore for FEM subdomain surrogate modelling on
    a fixed unstructured interface mesh.

    Architecture overview
    ---------------------
    1. Hierarchical encoder  — DenseSAGEConv + learnable soft-pooling (DiffPool style)
                               downsamples the interface from N nodes to a small coarse graph.
    2. Latent encoder        — Flattens the full coarse graph [coarse_nodes × coarse_channels]
                               and maps to latent_dim via MLP.  No mean-pooling: the spatial
                               arrangement of the coarse nodes is fully preserved.
    3. GRU physics core      — Single-step GRU that carries the subdomain's internal state.
                               Only layer-0 of the GRU is seeded from the learned internal_in
                               projection; deeper layers start from zero and let the GRU's own
                               gating infer their initial state.
    4. Bottleneck mixer      — Concatenates the GRU output (temporal) with the flattened coarse
                               graph (spatial) and projects back to the coarse graph shape via
                               MLP.  The MLP sees both channels jointly, not as a broadcast.
    5. Hierarchical decoder  — Symmetric unpooling with skip connections, followed by
                               DenseSAGEConv at each level.
    6. Final projection      — Per-node nn.Linear → dim (e.g. 2 for 2-D force vectors).

    Efficiency note
    ---------------
    During training the coarsened adjacency matrices are recomputed from the learnable pool
    parameters every forward pass.  After training, call ``freeze_and_cache_adjacencies()``
    once to pre-compute and register those matrices as buffers.  Subsequent inference passes
    skip the O(N²) pooling einsums entirely.  Re-call the method (or set ``_pooling_cached``
    back to False) if you ever fine-tune the pool parameters.

    Config keys (in addition to the shared SPILSNet keys)
    -------------------------------------------------------
    encoder_structure : list of dicts with keys ``out`` (int) and ``nodes`` (int)
    decoder_structure : list of dicts with key  ``out`` (int)
    latent_encoder_mlp, bottleneck_mlp, internal_input_mlp, internal_output_mlp : List[int]
    latent_dim, gru_hidden_size, gru_layers, internal_state_size : int
    edge_index : [[src, ...], [dst, ...]]  — interface mesh connectivity
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.dim = config["dimension"]
        self.n_nodes = config["input_size"] // self.dim
        self.drop_p = config.get("dropout_rate", 0.0)

        dtype_str = config.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_str == "float64" else torch.float32

        self.gru_hidden = config["gru_hidden_size"]
        self.gru_layers = config["gru_layers"]
        self.internal_state_size = config["internal_state_size"]
        self._n_enc_levels = len(config["encoder_structure"])
        # Set to True after freeze_and_cache_adjacencies(); must be restored after
        # loading a state_dict if adjacency buffers were saved with it.
        self._pooling_cached = False

        # Fixed interface mesh → dense adjacency buffer
        edge_index = torch.tensor(config["edge_index"], dtype=torch.long)
        dense_adj = to_dense_adj(edge_index, max_num_nodes=self.n_nodes)[0]
        self.register_buffer("base_adj", dense_adj.to(self.dtype))

        # ── 1. HIERARCHICAL ENCODER ──────────────────────────────────────────
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ParameterList()

        in_channels = self.dim
        current_nodes = self.n_nodes
        self.enc_channel_history: List[int] = []

        for layer_cfg in config["encoder_structure"]:
            out_ch = layer_cfg["out"]
            target_nodes = layer_cfg["nodes"]

            self.encoder_convs.append(nn.ModuleDict({
                "conv": DenseSAGEConv(in_channels, out_ch).to(self.dtype),
                "norm": nn.LayerNorm(out_ch, dtype=self.dtype),
                "act":  nn.Tanh(),
                "drop": nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
            }))
            # Soft-assignment matrix: [N_current → N_target]
            self.encoder_pools.append(
                nn.Parameter(
                    torch.randn(current_nodes, target_nodes, dtype=self.dtype) * 0.01
                )
            )

            self.enc_channel_history.append(out_ch)
            in_channels = out_ch
            current_nodes = target_nodes

        self.coarse_nodes = current_nodes
        self.coarse_channels = in_channels
        self.coarse_flat_size = self.coarse_nodes * self.coarse_channels

        # ── 2. LATENT ENCODER (structure-preserving) ─────────────────────────
        # Flatten [B, coarse_nodes, coarse_channels] → [B, coarse_flat_size]
        # then project to latent_dim.  The full spatial layout is retained.
        latent_dim = config["latent_dim"]
        self.latent_enc = build_mlp(
            self.coarse_flat_size,
            config["latent_encoder_mlp"],
            latent_dim,
            drop_p=0.0,
            dtype=self.dtype,
        )

        # ── 3. GRU PHYSICS CORE ───────────────────────────────────────────────
        # internal_in projects the stored internal state to a single GRU hidden
        # vector that seeds only layer 0.  Deeper layers start at zero; the GRU's
        # own gating handles their initialisation implicitly.
        self.internal_in = build_mlp(
            self.internal_state_size,
            config["internal_input_mlp"],
            self.gru_hidden,        # intentionally one layer's worth
            drop_p=0.0,
            dtype=self.dtype,
        )
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=False,
            dropout=0.0,
            dtype=self.dtype,
        )
        self.internal_out = build_mlp(
            self.gru_hidden,
            config["internal_output_mlp"],
            self.internal_state_size,
            drop_p=0.0,
            dtype=self.dtype,
        )

        # ── 4. BOTTLENECK MIXER ───────────────────────────────────────────────
        # [h_last ‖ coarse_flat] → MLP → coarse_flat_size
        # The MLP sees both the temporal GRU memory and the full spatial coarse
        # graph jointly and learns how to fuse them — no uniform broadcasting.
        bottleneck_in = self.gru_hidden + self.coarse_flat_size
        self.bottleneck_mixer = build_mlp(
            bottleneck_in,
            config["bottleneck_mlp"],
            self.coarse_flat_size,
            drop_p=self.drop_p,
            dtype=self.dtype,
        )

        # ── 5. HIERARCHICAL DECODER ───────────────────────────────────────────
        self.decoder_convs = nn.ModuleList()
        dec_in_channels = self.coarse_channels
        reversed_enc_channels = list(reversed(self.enc_channel_history))

        for idx, layer_cfg in enumerate(config["decoder_structure"]):
            out_ch = layer_cfg["out"]
            skip_ch = reversed_enc_channels[idx]
            conv_in = dec_in_channels + skip_ch   # unpooled features + skip

            self.decoder_convs.append(nn.ModuleDict({
                "conv": DenseSAGEConv(conv_in, out_ch).to(self.dtype),
                "norm": nn.LayerNorm(out_ch, dtype=self.dtype),
                "act":  nn.Tanh(),
                "drop": nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity(),
            }))
            dec_in_channels = out_ch

        # ── 6. FINAL PROJECTION ───────────────────────────────────────────────
        self.final_projection = nn.Linear(dec_in_channels, self.dim, dtype=self.dtype)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _soft_assignments(self) -> List[torch.Tensor]:
        """Return softmax-normalised assignment matrices for all encoder levels."""
        return [torch.softmax(p, dim=-1) for p in self.encoder_pools]

    def _get_adj_levels(
        self, batch_size: int, s_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Return the (batched) adjacency matrix at each encoder level.

        If ``freeze_and_cache_adjacencies()`` has been called, returns pre-stored
        buffers (just a cheap .expand — no computation).  Otherwise recomputes the
        coarsened adjacencies from scratch (needed during training when the pool
        parameters are still being updated).
        """
        if self._pooling_cached:
            return [
                getattr(self, f"_adj_level_{i}")
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                for i in range(self._n_enc_levels)
            ]

        # Training path: coarsen adj level by level
        adj_levels: List[torch.Tensor] = []
        adj = self.base_adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        for s in s_list:
            adj_levels.append(adj)
            # S^T A S  with S = [N_fine, N_coarse]
            adj = torch.einsum("ni, bnm, mj -> bij", s, adj, s)
        return adj_levels

    def freeze_and_cache_adjacencies(self) -> None:
        """
        Pre-compute and register all coarsened adjacency matrices as buffers.

        Call once after training is complete (pool parameters must be frozen or
        at least stable).  Each inference forward pass will then skip the
        O(N²) coarsening einsums.

        Note: the cached buffers are included in ``state_dict()``.  After loading
        a checkpoint that was saved with caching enabled, set
        ``model._pooling_cached = True`` to reactivate the fast path, or simply
        call this method again.
        """
        with torch.no_grad():
            s_list = self._soft_assignments()
            adj = self.base_adj.clone()             # [N, N] — unbatched
            for i, s in enumerate(s_list):
                self.register_buffer(f"_adj_level_{i}", adj.clone())
                # Unbatched S^T A S
                adj = torch.einsum("ni, nm, mj -> ij", s, adj, s)
        self._pooling_cached = True

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self, x_in: torch.Tensor, internal_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_in           : [B, n_nodes * dim]          — interface nodal values
            internal_state : [B, internal_state_size]     — learned subdomain memory

        Returns:
            out_flat       : [B, n_nodes * dim]           — predicted nodal output
            internal_next  : [B, internal_state_size]     — updated memory
        """
        batch_size = x_in.size(0)
        device = x_in.device

        x = x_in.view(batch_size, self.n_nodes, self.dim)  # [B, N, dim]

        # Compute soft assignments once; share between encoder and adj computation
        s_list = self._soft_assignments()
        adj_levels = self._get_adj_levels(batch_size, s_list)

        # ── 1. ENCODER ───────────────────────────────────────────────────────
        encoder_history: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for layer_dict, s, adj in zip(self.encoder_convs, s_list, adj_levels):
            x = layer_dict["conv"](x, adj)
            x = layer_dict["norm"](x)
            x = layer_dict["act"](x)
            x = layer_dict["drop"](x)

            encoder_history.append((x, adj, s))    # save fine state before pooling

            x = torch.einsum("nk, bnc -> bkc", s, x)   # soft-pool nodes

        coarse_graph = x    # [B, coarse_nodes, coarse_channels]

        # ── 2. LATENT ENCODING (no mean-pool) ────────────────────────────────
        coarse_flat = coarse_graph.reshape(batch_size, self.coarse_flat_size)
        gru_input = self.latent_enc(coarse_flat).unsqueeze(0)   # [1, B, latent_dim]

        # ── 3. GRU PHYSICS STEP ──────────────────────────────────────────────
        # Seed only layer 0; layers 1..L-1 start from zero
        h_first = torch.tanh(self.internal_in(internal_state))  # [B, gru_hidden]
        h_0 = torch.zeros(
            self.gru_layers, batch_size, self.gru_hidden,
            dtype=self.dtype, device=device
        )
        h_0[0] = h_first

        _, h_n = self.gru(gru_input, h_0)
        h_last = h_n[-1]                            # [B, gru_hidden]
        internal_next = self.internal_out(h_last)

        # ── 4. BOTTLENECK MIXING ──────────────────────────────────────────────
        # Fuse temporal (GRU) and spatial (coarse graph) information jointly,
        # then reshape back to [B, coarse_nodes, coarse_channels]
        fused = torch.cat([h_last, coarse_flat], dim=-1)    # [B, gru_hidden + coarse_flat_size]
        x_dec = self.bottleneck_mixer(fused)                # [B, coarse_flat_size]
        x_dec = x_dec.view(batch_size, self.coarse_nodes, self.coarse_channels)

        # ── 5. DECODER ───────────────────────────────────────────────────────
        for layer_dict, (x_enc, adj_fine, s) in zip(
            self.decoder_convs, reversed(encoder_history)
        ):
            # Unpool: coarse → fine via transpose of assignment matrix S
            x_dec = torch.einsum("nk, bkc -> bnc", s, x_dec)

            # Concatenate skip connection from symmetric encoder level
            x_dec = torch.cat([x_dec, x_enc], dim=-1)

            # Convolve on the physical fine-level adjacency
            x_dec = layer_dict["conv"](x_dec, adj_fine)
            x_dec = layer_dict["norm"](x_dec)
            x_dec = layer_dict["act"](x_dec)
            x_dec = layer_dict["drop"](x_dec)

        # ── 6. FINAL PROJECTION ───────────────────────────────────────────────
        out_graph = self.final_projection(x_dec)            # [B, n_nodes, dim]
        out_flat = out_graph.reshape(batch_size, self.n_nodes * self.dim)

        return out_flat, internal_next


import random
import torch
import torch.nn as nn
import pymetis
from torch_scatter import scatter_mean
from torch_geometric.nn import NNConv
from torch_geometric.utils import remove_self_loops
from typing import Dict, Any, Tuple, List

from spilsnet.utils import build_mlp


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_coarse_edges(
    edge_index: torch.Tensor, cluster_map: torch.Tensor
) -> torch.Tensor:
    """Project fine-level edges into the coarse graph and deduplicate."""
    row, col = edge_index
    coarse_edges = torch.stack([cluster_map[row], cluster_map[col]], dim=0)
    coarse_edges, _ = remove_self_loops(coarse_edges)
    return torch.unique(coarse_edges, dim=1)


def build_adjacency_list(edge_index: torch.Tensor, n_nodes: int) -> List[List[int]]:
    """Convert a PyG edge_index to an undirected adjacency list for METIS."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in edge_index.t().tolist():
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SparseHierarchicalSPILSNet(nn.Module):
    """
    Sparse hierarchical GNN surrogate for FEM subdomains with large, potentially
    unstructured interface meshes (hundreds of thousands of nodes).

    Key design choices vs. the dense GraphUNet variant
    ---------------------------------------------------
    * **Sparse ops throughout** — NNConv + torch_scatter instead of dense
      adjacency matrices.  Memory scales with E (edges) not N² (nodes).
    * **METIS partitioning** — graph-aware, deterministic hard clustering.
      Avoids soft-pooling whose cost is quadratic in N.
    * **Live edge distances** — every NNConv call (encoder *and* decoder)
      receives the current Euclidean edge length derived from deformed node
      positions.  The distance MLP learns stiffness-like responses that adapt
      as geometry changes (critical for crash/contact simulation).
    * **Correct batching** — fine and coarse edge indices and cluster maps are
      offset per batch item so no two graphs share indices.
    * **Structure-preserving latent** — the coarse graph is *flattened* (not
      mean-pooled) before the GRU, preserving spatial arrangement.
    * **Bottleneck mixer** — GRU hidden state and coarse spatial features are
      fused jointly via MLP; no uniform broadcast to coarse nodes.
    * **Layer-0-only GRU seeding** — ``internal_in`` projects the stored
      subdomain state to a single hidden vector that seeds only layer 0 of the
      GRU; deeper layers are initialised to zero.

    Architecture
    ------------
    Input [B, N*dim]
        ↓  Encoder  (NNConv + METIS pool) × n_levels  — saves skip features & positions
        ↓  Flatten coarse graph → latent MLP → GRU
        ↓  Bottleneck mixer  [h_last ‖ coarse_flat] → MLP → coarse graph
        ↓  Decoder  (unpool + concat skip + NNConv) × n_levels
        ↓  Linear projection → [B, N*dim]

    Config keys
    -----------
    dimension, input_size, internal_state_size : int
    encoder_structure : list of {"out": int, "nodes": int}
    decoder_structure : list of {"out": int}
    latent_dim, gru_hidden_size, gru_layers : int
    latent_encoder_mlp, bottleneck_mlp,
    internal_input_mlp, internal_output_mlp : List[int]
    edge_distance_hidden : int  (hidden width of the edge MLP, default 16)
    dropout_rate : float        (applied to encoder/decoder norms, default 0.0)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        metis_seed: int = 42,
    ) -> None:
        super().__init__()

        self.dim = config["dimension"]
        self.n_nodes = pos.size(0)
        self.dtype = torch.float32
        self._n_enc_levels = len(config["encoder_structure"])
        self.drop_p = config.get("dropout_rate", 0.0)
        edge_hidden = config.get("edge_distance_hidden", 16)

        self.gru_hidden = config["gru_hidden_size"]
        self.gru_layers = config["gru_layers"]
        self.internal_state_size = config["internal_state_size"]

        # Reference (undeformed) positions — moves to GPU with the model
        self.register_buffer("base_pos", pos.to(self.dtype))

        # ── 1. HIERARCHICAL GRAPH STRUCTURE (METIS) ──────────────────────────
        # node_counts[i] = #nodes at encoder level i  (0 = finest = n_nodes)
        # edges_{i}     = edge_index at level i        (buffer)
        # clusters_{i}  = cluster_map fine[i]→coarse[i+1]  (buffer)
        self.node_counts: List[int] = [self.n_nodes]

        curr_edges = edge_index
        curr_nodes = self.n_nodes

        for i, layer_cfg in enumerate(config["encoder_structure"]):
            target_nodes = layer_cfg["nodes"]

            random.seed(metis_seed + i)   # per-level seed → deterministic
            adj_list = build_adjacency_list(curr_edges, curr_nodes)
            _, cluster_map = pymetis.part_graph(target_nodes, adjacency=adj_list)
            cluster_map = torch.tensor(cluster_map, dtype=torch.long)

            self.register_buffer(f"edges_{i}", curr_edges)
            self.register_buffer(f"clusters_{i}", cluster_map)

            curr_edges = build_coarse_edges(curr_edges, cluster_map)
            curr_nodes = target_nodes
            self.node_counts.append(curr_nodes)

        # Coarsest level edges (used by the first decoder conv)
        self.register_buffer(f"edges_{self._n_enc_levels}", curr_edges)

        self.coarse_nodes = curr_nodes

        # ── 2. ENCODER ────────────────────────────────────────────────────────
        # NNConv: edge MLP maps scalar distance → weight matrix [in_ch × out_ch].
        # This lets the convolution adapt to deformed geometry at every step.
        self.enc_convs = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        self.enc_drops = nn.ModuleList()
        in_ch = self.dim
        self.enc_channel_history: List[int] = []

        for layer_cfg in config["encoder_structure"]:
            out_ch = layer_cfg["out"]
            self.enc_convs.append(NNConv(
                in_ch, out_ch,
                nn=self._edge_mlp(1, edge_hidden, in_ch * out_ch),
                aggr="mean",
            ))
            self.enc_norms.append(nn.LayerNorm(out_ch, dtype=self.dtype))
            self.enc_drops.append(
                nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity()
            )
            self.enc_channel_history.append(out_ch)
            in_ch = out_ch

        self.coarse_channels = in_ch
        self.coarse_flat_size = self.coarse_nodes * self.coarse_channels

        # ── 3. LATENT ENCODER ────────────────────────────────────────────────
        # Full flatten preserves the spatial layout of the coarse partition.
        latent_dim = config["latent_dim"]
        self.latent_enc = build_mlp(
            self.coarse_flat_size,
            config["latent_encoder_mlp"],
            latent_dim,
            drop_p=0.0,
            dtype=self.dtype,
        )

        # ── 4. GRU PHYSICS CORE ───────────────────────────────────────────────
        # internal_in → seeds h_0[0] only; deeper layers start at zero.
        self.internal_in = build_mlp(
            self.internal_state_size,
            config["internal_input_mlp"],
            self.gru_hidden,          # one layer's worth, intentionally
            drop_p=0.0,
            dtype=self.dtype,
        )
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=False,
            dtype=self.dtype,
        )
        self.internal_out = build_mlp(
            self.gru_hidden,
            config["internal_output_mlp"],
            self.internal_state_size,
            drop_p=0.0,
            dtype=self.dtype,
        )

        # ── 5. BOTTLENECK MIXER ───────────────────────────────────────────────
        # [h_last ‖ coarse_flat] → MLP → coarse_flat.
        # GRU memory and spatial coarse graph are fused jointly; no broadcasting.
        bottleneck_in = self.gru_hidden + self.coarse_flat_size
        self.bottleneck_mixer = build_mlp(
            bottleneck_in,
            config["bottleneck_mlp"],
            self.coarse_flat_size,
            drop_p=self.drop_p,
            dtype=self.dtype,
        )

        # ── 6. DECODER ────────────────────────────────────────────────────────
        # Mirrors encoder in reverse order.  Each level:
        #   unpool (index coarse → fine) → cat skip → NNConv with fine distances.
        self.dec_convs = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        self.dec_drops = nn.ModuleList()

        dec_in_ch = self.coarse_channels
        reversed_enc_ch = list(reversed(self.enc_channel_history))

        for idx, layer_cfg in enumerate(config["decoder_structure"]):
            out_ch = layer_cfg["out"]
            skip_ch = reversed_enc_ch[idx]
            conv_in = dec_in_ch + skip_ch     # unpooled + skip connection

            self.dec_convs.append(NNConv(
                conv_in, out_ch,
                nn=self._edge_mlp(1, edge_hidden, conv_in * out_ch),
                aggr="mean",
            ))
            self.dec_norms.append(nn.LayerNorm(out_ch, dtype=self.dtype))
            self.dec_drops.append(
                nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity()
            )
            dec_in_ch = out_ch

        # ── 7. FINAL PROJECTION ───────────────────────────────────────────────
        self.final_projection = nn.Linear(dec_in_ch, self.dim, dtype=self.dtype)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _edge_mlp(self, in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
        """Small MLP mapping a scalar edge feature → NNConv weight matrix."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden, out_dim, dtype=self.dtype),
        )

    @staticmethod
    def _edge_distances(edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Euclidean distance for each edge: [E, 1]."""
        src, dst = edge_index
        return torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)

    def _batch_edges(
        self, edge_index: torch.Tensor, batch_size: int, n_nodes: int
    ) -> torch.Tensor:
        """
        Tile edge_index for a full batch.

        Fine node b*n_nodes + i should only be connected to nodes in the same
        graph item b.  We achieve this by adding b*n_nodes to every index for
        batch item b.

        [2, E] → [2, B*E]
        """
        offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes
        return (
            edge_index.unsqueeze(0)         # [1, 2, E]
            + offsets.view(-1, 1, 1)        # [B, 1, 1]
        ).permute(1, 0, 2).reshape(2, -1)   # [2, B*E]

    def _batch_clusters(
        self,
        clusters: torch.Tensor,
        batch_size: int,
        n_fine: int,
        n_coarse: int,
    ) -> torch.Tensor:
        """
        Build a flat cluster assignment for the entire batch.

        Fine node (b*n_fine + i) should map to coarse node
        (b*n_coarse + clusters[i]), keeping each batch item's graph isolated.

        clusters : [N_fine] with values in [0, N_coarse)
        returns  : [B*N_fine] with values in [0, B*N_coarse)
        """
        coarse_offsets = (
            torch.arange(batch_size, device=clusters.device) * n_coarse
        ).repeat_interleave(n_fine)               # [B * N_fine]
        return clusters.repeat(batch_size) + coarse_offsets

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        displacements: torch.Tensor,    # [B, n_nodes * dim]
        internal_state: torch.Tensor,   # [B, internal_state_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out_flat      : [B, n_nodes * dim]
        internal_next : [B, internal_state_size]
        """
        batch_size = displacements.size(0)
        device = displacements.device

        # Node features (displacements) and deformed positions, both flat
        x = displacements.view(batch_size * self.n_nodes, self.dim)
        p_level = (
            self.base_pos
            .unsqueeze(0).expand(batch_size, -1, -1)
            .reshape(-1, self.dim)
        ) + x                   # [B*N, dim]  — deformed positions

        # ── ENCODER ──────────────────────────────────────────────────────────
        # encoder_history[i] = (x_fine, p_fine, clusters_b)
        #   x_fine    : [B*N_fine, C]   — post-conv features before pooling
        #   p_fine    : [B*N_fine, dim] — deformed positions before pooling
        #   clusters_b: [B*N_fine]      — batched assignments (fine → coarse)
        encoder_history: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for i, (conv, norm, drop) in enumerate(
            zip(self.enc_convs, self.enc_norms, self.enc_drops)
        ):
            n_fine = self.node_counts[i]
            n_coarse = self.node_counts[i + 1]

            edges_b = self._batch_edges(
                getattr(self, f"edges_{i}"), batch_size, n_fine
            )
            clusters_b = self._batch_clusters(
                getattr(self, f"clusters_{i}"), batch_size, n_fine, n_coarse
            )
            dist = self._edge_distances(edges_b, p_level)   # live distances

            x = conv(x, edges_b, dist)
            x = drop(torch.tanh(norm(x)))

            encoder_history.append((x, p_level, clusters_b))

            # Hard pool: scatter fine nodes into coarse clusters (features + positions)
            total_coarse = batch_size * n_coarse
            x = scatter_mean(x, clusters_b, dim=0, dim_size=total_coarse)
            p_level = scatter_mean(p_level, clusters_b, dim=0, dim_size=total_coarse)

        # ── LATENT ENCODING ───────────────────────────────────────────────────
        # Flatten the full coarse graph — no mean pooling, spatial layout preserved
        coarse_flat = x.view(batch_size, self.coarse_flat_size)         # [B, coarse_flat]
        gru_input = self.latent_enc(coarse_flat).unsqueeze(0)           # [1, B, latent_dim]

        # ── GRU PHYSICS STEP ──────────────────────────────────────────────────
        h_first = torch.tanh(self.internal_in(internal_state))          # [B, gru_hidden]
        h_0 = torch.zeros(
            self.gru_layers, batch_size, self.gru_hidden,
            dtype=self.dtype, device=device,
        )
        h_0[0] = h_first        # seed only layer 0; deeper layers inferred by GRU

        _, h_n = self.gru(gru_input, h_0)
        h_last = h_n[-1]                                                # [B, gru_hidden]
        internal_next = self.internal_out(h_last)

        # ── BOTTLENECK MIXING ─────────────────────────────────────────────────
        # Fuse temporal and spatial representations jointly, then reshape to coarse graph
        fused = torch.cat([h_last, coarse_flat], dim=-1)                # [B, gru_hidden + coarse_flat]
        x_dec = self.bottleneck_mixer(fused)                            # [B, coarse_flat]
        x_dec = x_dec.view(batch_size * self.coarse_nodes, self.coarse_channels)

        # ── DECODER ───────────────────────────────────────────────────────────
        for i, (conv, norm, drop) in enumerate(
            zip(self.dec_convs, self.dec_norms, self.dec_drops)
        ):
            enc_idx = self._n_enc_levels - 1 - i
            x_enc, p_fine, clusters_b = encoder_history[enc_idx]

            n_fine = self.node_counts[enc_idx]
            edges_b = self._batch_edges(
                getattr(self, f"edges_{enc_idx}"), batch_size, n_fine
            )

            # Unpool: broadcast each coarse node to its fine members
            x_dec = x_dec[clusters_b]                                   # [B*N_fine, C_dec]

            # Skip connection from the symmetric encoder level
            x_dec = torch.cat([x_dec, x_enc], dim=-1)

            # Distance-aware refinement using the saved fine-level deformed positions
            dist = self._edge_distances(edges_b, p_fine)
            x_dec = conv(x_dec, edges_b, dist)
            x_dec = drop(torch.tanh(norm(x_dec)))

        # ── FINAL PROJECTION ─────────────────────────────────────────────────
        out = self.final_projection(x_dec)                              # [B*N, dim]
        return out.view(batch_size, self.n_nodes * self.dim), internal_next
