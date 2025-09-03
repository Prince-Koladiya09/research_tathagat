from mylib.config import (
    DEFAULT_CONFIG,
    OPTIMIZER_MAP,
    LOG_DIR,
    get_model_path,
    def_callbacks
)
from keras import layers

# Create a copy so that local changes don't affect global config
CONFIG = DEFAULT_CONFIG["cnn"].copy()
OPTIMIZERS = OPTIMIZER_MAP

def get_custom_layers(num_classes : int) -> list :
    return [
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(DEFAULT_CONFIG["num_classes"], activation="softmax")
    ]