import glob
import pytest
import os

@pytest.fixture(autouse=True)
def cleanup_model_file():
    yield
    # Remove all files starting with current_lstm_model
    for f in glob.glob("current_lstm_model*"):
        os.remove(f)
