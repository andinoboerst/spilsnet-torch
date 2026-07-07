# SPILSNet-Torch

[![PyPI version](https://img.shields.io/pypi/v/spilsnet-torch.svg)](https://pypi.org/project/spilsnet-torch/)
[![Tests](https://github.com/andinoboerst/spilsnet-torch/actions/workflows/tests.yml/badge.svg)](https://github.com/andinoboerst/spilsnet-torch/actions)
[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.1234/zenodo.1234567-blue.svg)](https://doi.org/10.5281/zenodo.21236798)

A high-performance PyTorch implementation of **SPILSNet** (Spatiotemporal Physics-derived Internal Latent Space Network). 

SPILSNet is designed for modeling complex dynamical systems where preserving physical structure (like spatial relationships and temporal consistency) is critical. It combines convolutional encoders for spatial feature extraction with a Gated Recurrent Unit (GRU) core to capture temporal evolution, while maintaining a learned skip-connection architecture to preserve high-frequency details.

## Features

- **Unified Non-Pickle Serialization**: Industry-standard `safetensors` format for weights and metadata (config, scalers), ensuring cross-platform safety and performance.
- **Physics-derived Temporal Dynamics**: GRU-based core for robust state-space modeling.
- **Flexible Scaling**: Built-in support for Scikit-learn scalers and custom transformers (e.g., CubeRoot).
- **Professional Engineering**: Full type hinting, Google-style docstrings, and robust serialization.
- **Extensible Loss**: Custom `spils_loss` including Laplacian smoothness terms for spatial consistency.

## Installation

Install via pip:

```bash
pip install spilsnet-torch
```

For development:

```bash
git clone https://github.com/andinoboerst/spilsnet-torch.git
cd spilsnet-torch
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import torch
from spilsnet import SPILSNet
from sklearn.preprocessing import StandardScaler

# 1. Configure the architecture
model_config = {
    "dimension": 2,                # 2D coordinates (x, y)
    "input_size": 102,             # 51 nodes * 2 dimensions
    "internal_state_size": 16,     # Size of physical internal states
    "encoder_structure": [
        {"out": 32, "k": 3, "s": 1, "p": 1},
        {"out": 16, "k": 3, "s": 1, "p": 1},
    ],
    "bottleneck_pool_size": 4,
    "latent_dim": 32,
    "gru_hidden_size": 64,
    "latent_encoder_mlp": [64, 64],
    "internal_input_mlp": [32],
    "internal_output_mlp": [32],
    "dropout_rate": 0.1,
}

# 2. Initialize the wrapper
model = SPILSNet(
    model_config=model_config,
    input_scaler_class=StandardScaler(),
    internal_in_scaler_class=StandardScaler(),
    internal_out_scaler_class=StandardScaler(),
    output_scaler_class=StandardScaler()
)

# 3. Fit the model
# X: [Sims, Steps, Input_Size], Y: [Sims, Steps, Output_Size], I: [Sims, Steps, Internal_Size]
X, Y, I = np.random.randn(10, 50, 102), np.random.randn(10, 50, 102), np.random.randn(10, 50, 16)
model.fit(X, Y, I)

# 4. Sequential Inference
model.initialize_memory_variables()
current_x = np.random.randn(102)
next_y = model.predict(current_x)

print(f"Predicted next state shape: {next_y.shape}")
```

## Testing

Run the test suite using `pytest`:

```bash
pytest
```

To run with coverage:

```bash
pytest --cov=spilsnet
```

## License

This project is licensed under the AGPL 3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite the associated paper and this repository.

### Paper Citation
```bibtex
@article{boerst2026spilsnet,
  title={Accelerating Transient Structural Dynamics via SPILS-Net, a Physics-Derived Latent Space Subdomain Surrogate},
  author={Börst, Andino and Díez, Pedro and Zlotnik, Sergio and Cavaliere, Fabiola and Curtosi, Gabriel and Larráyoz, Xabier},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2026},
  doi={[DOI — to be added upon publication]}
}
```

### Software Citation
(this repository, [spilsnet-torch](https://github.com/andinoboerst/spilsnet-torch), available on [PyPI](https://pypi.org/project/spilsnet-torch/)):
```bibtex
@software{boerst_spilsnet_torch_2026,
  author={Börst, Andino},
  title={spilsnet-torch: PyTorch Implementation of SPILS-Net},
  year={2026},
  publisher={Zenodo},
  url={https://github.com/andinoboerst/spilsnet-torch},
  doi={10.5281/zenodo.21236781},
  version={1.0.1}
}
```

For machine-readable citation metadata, see [`CITATION.cff`](CITATION.cff).
