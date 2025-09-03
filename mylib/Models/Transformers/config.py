from mylib.config import (
    DEFAULT_CONFIG,
    OPTIMIZER_MAP,
    get_model_path,
    def_callbacks
)
# Copy global config
CONFIG = DEFAULT_CONFIG["transformers"].copy()
OPTIMIZERS = OPTIMIZER_MAP