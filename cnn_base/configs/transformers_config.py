from .base_config import Global_Config, Optimizer_Config

TRANSFORMER_OPTIMIZER_CONFIG = Optimizer_Config(
    name="adamw",
    learning_rate=5e-5,
    params={"weight_decay": 0.01}
)

DEFAULT_TRANSFORMER_CONFIG = Global_Config(
    optimizer=TRANSFORMER_OPTIMIZER_CONFIG
)