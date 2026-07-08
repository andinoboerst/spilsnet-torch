import glob
import pytest
import os

@pytest.fixture(autouse=True)
def cleanup_model_file():
    yield
    # Remove all temporary model files generated during tests
    for pattern in ["current_lstm_model*", "spilsnet_model*", "*.safetensors"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass
