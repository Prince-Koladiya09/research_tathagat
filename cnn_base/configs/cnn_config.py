from .base_config import Global_Config, Optimizer_Config

CNN_OPTIMIZER_CONFIG = Optimizer_Config(
    name="adam",
    learning_rate=1e-3,
    scheduler_name="cosine_decay"
)

DEFAULT_CNN_CONFIG = Global_Config(
    optimizer=CNN_OPTIMIZER_CONFIG
)