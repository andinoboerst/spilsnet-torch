import torch
import torch.nn as nn

from spilsnet.utils import build_mlp


class SPILSNetCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_nodes = config["input_size"] // config["dimension"]
        self.dim = config["dimension"]
        self.drop_p = config.get("dropout_rate", 0.0)

        # --- 1. ENCODER (The Eye) ---
        self.encoder_stack = nn.ModuleList()
        current_in = self.dim

        # Note: We removed 'skip_processors' because this is an asymmetric model.
        # We don't need U-Net skips for the MLP decoder.
        for layer_cfg in config["encoder_structure"]:
            block = nn.Sequential(
                nn.Conv1d(current_in, layer_cfg["out"],
                          kernel_size=layer_cfg["k"], stride=layer_cfg["s"], padding=layer_cfg["p"],
                          padding_mode="replicate", dtype=torch.float64),
                nn.Tanh(),
                nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity()
            )
            self.encoder_stack.append(block)
            current_in = layer_cfg["out"]

        # --- DYNAMIC SIZE DETECTION (The Dummy Pass Trick) ---
        # 1. Create a fake input tensor matching your actual data shape: [Batch=1, Channels=Dim, Length=Nodes]
        dummy_input = torch.zeros(1, self.dim, self.n_nodes, dtype=torch.float64)

        # 2. Pass it through the encoder stack without tracking gradients
        with torch.no_grad():
            dummy_out = dummy_input
            for layer in self.encoder_stack:
                dummy_out = layer(dummy_out)

        # 3. Read the exact spatial dimension that survived!
        # dummy_out shape is [1, current_in, spatial_nodes_out]
        spatial_nodes_out = dummy_out.size(2)

        # Define how many "Macro-Regions" you want the skip connection to have
        self.skip_target_nodes = config.get("skip_target_nodes", 3)

        # The Learned Spatial Downsampler!
        # Maps whatever comes out of the Conv stack down to exactly 3 nodes.
        self.spatial_downsampler = nn.Sequential(
            nn.Linear(spatial_nodes_out, self.skip_target_nodes, dtype=torch.float64),
            nn.Tanh(),
            # nn.Dropout(self.drop_p) if self.drop_p > 0 else nn.Identity()
        )

        # The flat size for the Decoder MLP is now strictly guaranteed:
        skip_connection_size = current_in * self.skip_target_nodes

        # --- 2. BOTTLENECK (The Brain) ---
        self.pool_size = config["bottleneck_pool_size"]
        self.pooling_layer = nn.AdaptiveAvgPool1d(self.pool_size)

        # Fix: Ensure we get the channel count from the last config dict
        last_layer_out = config["encoder_structure"][-1]["out"]
        flat_size = last_layer_out * self.pool_size

        latent_dim = config["latent_dim"]
        self.gru_hidden = config["gru_hidden_size"]
        self.gru_layers = config.get("gru_layers", 1)

        # Map Spatial Features -> GRU Input
        self.latent_enc = build_mlp(flat_size, config["latent_encoder_mlp"], latent_dim, drop_p=0.0)

        # Removed 'self.latent_dec' (Dead Code):
        # We don't need to reconstruct the bottleneck because we use the concatenation skip.

        # --- 3. PHYSICS CORE (GRU) ---
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            dropout=0.0,
            dtype=torch.float64
        )

        # Internal State Handling
        self.total_hidden_params = self.gru_layers * self.gru_hidden
        self.internal_in = build_mlp(config["internal_state_size"], config["internal_input_mlp"], self.total_hidden_params, drop_p=0.0)
        self.internal_out = build_mlp(self.gru_hidden, config["internal_output_mlp"], config["internal_state_size"], drop_p=0.0)

        # --- 4. GLOBAL DECODER (The Projector) ---
        # Input: History (GRU) + Current Context (Pooled Features)
        global_in_size = self.gru_hidden + skip_connection_size

        # Output: FULL NODAL VECTOR
        self.latent_decoder = build_mlp(
            in_size=global_in_size,
            hidden_sizes=config.get("latent_decoder_structure", [512, 1024]),
            out_size=self.n_nodes * self.dim,
            drop_p=self.drop_p
        )

        # --- 5. SMOOTHING LAYER ---
        # A final convolution to clean up MLP noise.
        # k=3, padding="same" ensures dimensions don't change.
        self.smoothing_layer = nn.Conv1d(self.dim, self.dim, kernel_size=config.get("smoothing_kernel_size", 3), padding="same",
                                         padding_mode="replicate", dtype=torch.float64)

    def forward(self, x_in, internal_state):
        # 1. Reshape Input: [Batch, Nodes*Dim] -> [Batch, Dim, Nodes]
        # view(-1, Nodes, Dim) -> permute(0, 2, 1) ensures we group (x1,y1), (x2,y2)... correctly
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
        # We define the force based on History (h_last) AND Current Strain (pooled)
        global_input = torch.cat([h_last, skip_connection], dim=1)

        # MLP Output: [Batch, Nodes * Dim]
        raw_force = self.latent_decoder(global_input)

        # 1. View as [Batch, Nodes, Dim] -> Restores (x,y) pairs
        # 2. Permute -> [Batch, Dim, Nodes] -> Ready for Conv1d
        force_spatial = raw_force.view(batch_size, self.n_nodes, self.dim).permute(0, 2, 1)

        # 7. Smoothing Pass
        # Removes high-frequency jitter from the MLP prediction
        final_spatial = self.smoothing_layer(force_spatial)

        # 8. Flatten for Output
        out_flat = final_spatial.permute(0, 2, 1).reshape(x_in.shape)

        return out_flat, internal_next
