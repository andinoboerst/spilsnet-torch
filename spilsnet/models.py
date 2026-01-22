import torch
import torch.nn as nn


class SPILSNetCore(nn.Module):
    def __init__(
        self,
        dimension,
        input_size,
        internal_state_size,
        spatial_linear_layers: list,
        hidden_internal_size: int = 16,
        conv_layers_out_channels: list = [1],
        kernel_size: int = 3,
        internal_layers_in: list = [],
        n_gru_cells: int = 1,
        internal_layers_out: list = [],
        deconv_layers_out_channels: list = [],
        dropout_rate=0.2
    ):
        super(SPILSNetCore, self).__init__()

        if len(conv_layers_out_channels) < 1:
            raise ValueError("At least one convolutional layer must be specified")
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for 'same' padding")
        if kernel_size < 1:
            raise ValueError("Kernel size must be at least 1")
        if n_gru_cells < 1:
            raise ValueError("At least one GRU cell must be specified")
        if len(spatial_linear_layers) < 1:
            raise ValueError("At least one spatial linear layer must be specified")
        if hidden_internal_size < 1:
            raise ValueError("Hidden internal size must be at least 1")
        if input_size % dimension != 0:
            raise ValueError("Input size must be divisible by the dimension")

        self.dimension = dimension
        self.n_nodes = input_size // self.dimension

        if spatial_linear_layers[-1] % self.n_nodes != 0:
            raise ValueError("The last spatial linear layer size must be divisible by the number of nodes")

        self.spatial_linear_layers = spatial_linear_layers

        self.convolutional_stack = nn.ModuleList()
        in_channels = dimension
        for out_channels in conv_layers_out_channels:
            self.convolutional_stack.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                dtype=torch.float64
            ))
            self.convolutional_stack.append(nn.Tanh())
            self.convolutional_stack.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.internal_in_layers = nn.ModuleList()
        internal_layers_in.append(hidden_internal_size)
        internal_in_input_size = internal_state_size
        for layer_size in internal_layers_in:
            self.internal_in_layers.append(nn.Linear(internal_in_input_size, layer_size, dtype=torch.float64))
            self.internal_in_layers.append(nn.Tanh())
            self.internal_in_layers.append(nn.Dropout(dropout_rate))
            internal_in_input_size = layer_size

        self.gru_cells = nn.ModuleList()
        input_size_for_gru = self.n_nodes * conv_layers_out_channels[-1]
        for _ in range(n_gru_cells):
            self.gru_cells.append(nn.GRUCell(input_size_for_gru, hidden_internal_size, dtype=torch.float64))
            input_size_for_gru = hidden_internal_size

        self.internal_out_layers = nn.ModuleList()
        # internal_layers_out.append(internal_state_size)
        internal_out_input_size = hidden_internal_size
        for layer_size in internal_layers_out:
            self.internal_out_layers.append(nn.Linear(internal_out_input_size, layer_size, dtype=torch.float64))
            self.internal_out_layers.append(nn.Tanh())
            self.internal_out_layers.append(nn.Dropout(dropout_rate))
            internal_out_input_size = layer_size
        self.internal_out_layers.append(nn.Linear(internal_out_input_size, internal_state_size, dtype=torch.float64))

        self.spatial_linear_layers_stack = nn.ModuleList()
        spatial_input_size = hidden_internal_size + self.n_nodes * conv_layers_out_channels[-1]
        for layer_size in spatial_linear_layers:
            self.spatial_linear_layers_stack.append(nn.Linear(spatial_input_size, layer_size, dtype=torch.float64))
            self.spatial_linear_layers_stack.append(nn.Tanh())
            self.spatial_linear_layers_stack.append(nn.Dropout(dropout_rate))
            spatial_input_size = layer_size

        self.deconv_layers = nn.ModuleList()
        in_channels = spatial_linear_layers[-1] // self.n_nodes
        deconv_layers_out_channels.append(dimension)
        deconv_padding = kernel_size // 2
        for out_channels in deconv_layers_out_channels:
            self.deconv_layers.append(nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=deconv_padding,
                output_padding=0,
                dtype=torch.float64
            ))
            self.deconv_layers.append(nn.Tanh())
            self.deconv_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.spatial_out_final = nn.Linear(input_size, input_size, dtype=torch.float64)

    def forward(self, x_in, internal_state) -> tuple:

        disps = x_in[:, :self.n_nodes * self.dimension]
        x_disps = disps[:, ::self.dimension]
        y_disps = disps[:, 1::self.dimension]
        if self.dimension == 3:
            z_disps = disps[:, 2::self.dimension]
            z_conv = torch.stack([x_disps, y_disps, z_disps], dim=1)
        else:
            z_conv = torch.stack([x_disps, y_disps], dim=1)

        for layer in self.convolutional_stack:
            z_conv = layer(z_conv)

        z_conv = z_conv.flatten(1)

        z_i = internal_state.clone()
        for layer in self.internal_in_layers:
            z_i = layer(z_i)

        z_GRU_main = z_conv.clone()
        z_GRU_hidden = z_i.clone()
        for gru_cell in self.gru_cells:
            z_GRU_hidden = gru_cell(z_GRU_main, z_GRU_hidden)
            z_GRU_main = z_GRU_hidden

        i_next = z_GRU_hidden.clone()
        for layer in self.internal_out_layers:
            i_next = layer(i_next)

        z_F_linear = torch.cat([z_conv, z_GRU_hidden], dim=1)
        for layer in self.spatial_linear_layers_stack:
            z_F_linear = layer(z_F_linear)

        z_F_deconv = z_F_linear.view(-1, self.spatial_linear_layers[-1] // self.n_nodes, self.n_nodes)
        for layer in self.deconv_layers:
            z_F_deconv = layer(z_F_deconv)

        F_permuted = z_F_deconv.permute(0, 2, 1).flatten(start_dim=1)
        F = self.spatial_out_final(F_permuted)

        return F, i_next
