# spilsnet-torch

A PyTorch implementation of SPILSNet (Structure-Preserving Input-Output Learning System Network).

## Installation

You can install the package directly from the source:

```bash
pip install .
```

## Usage

```python
import torch
from spilsnet import SPILSNetCore

# Define model parameters
model = SPILSNetCore(
    dimension=2,
    input_size=10,
    internal_state_size=5,
    spatial_linear_layers=[5],
    hidden_internal_size=16,
    conv_layers_out_channels=[16, 1],
    kernel_size=3,
    internal_layers_in=[],
    n_gru_cells=1,
    internal_layers_out=[16, 16],
    deconv_layers_out_channels=[16],
    dropout_rate=0.2
)

# Create dummy input
batch_size = 32
x_in = torch.randn(batch_size, 10)
internal_state = torch.randn(batch_size, 5)

# Forward pass
output, next_internal_state = model(x_in, internal_state)

print("Output shape:", output.shape)
print("Next internal state shape:", next_internal_state.shape)
```