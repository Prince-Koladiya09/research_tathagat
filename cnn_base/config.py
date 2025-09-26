import os
from datetime import datetime
from typing import List, Tuple, Any, Dict, Optional

from pydantic import BaseModel, Field
from keras import layers
from keras.optimizers import (
    Adam, AdamW, Nadam, Adagrad, Adamax, Adadelta,
    SGD, RMSprop, Lamb, Lion
)
from keras.optimizers.schedules import (
    ExponentialDecay, PiecewiseConstantDecay, InverseTimeDecay,
    PolynomialDecay, CosineDecay
)
from keras.metrics import (
    Accuracy, Precision, Recall, AUC
)
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

PROJECT_ROOT = os.getcwd()
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
CV_DIR = os.path.join(STORAGE_DIR, "cv_results")
MODEL_DIR = os.path.join(STORAGE_DIR, "models")
LOG_DIR = os.path.join(STORAGE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 10
    seed: int = 42
    loss: str = "sparse_categorical_crossentropy"
    metrics: List[Any] = [
        Accuracy(name="accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]

class ModelConfig(BaseModel):
    num_classes: int = 4
    img_size: Tuple[int, int] = (224, 224)
    N_layers_to_tune: int = 20
    N_layers_to_remove: int = 1

class OptimizerConfig(BaseModel):
    name: str = "adam"
    params: Dict[str, Any] = {}
    learning_rate: float = 1e-3
    scheduler_name: Optional[str] = None
    scheduler_params: Dict[str, Any] = {}

class GlobalConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

CONFIG = GlobalConfig()

OPTIMIZERS = {
    "adam": {"class": Adam, "params": {}},
    "adamw": {"class": AdamW, "params": {"weight_decay": 1e-5}},
    "sgd": {"class": SGD, "params": {"momentum": 0.9}},
    "rmsprop": {"class": RMSprop, "params": {"momentum": 0.9}},
    "nadam": {"class": Nadam, "params": {}},
    "adagrad": {"class": Adagrad, "params": {}},
    "adadelta": {"class": Adadelta, "params": {}},
    "adamax": {"class": Adamax, "params": {}},
    "lamb": {"class": Lamb, "params": {}},
    "lion": {"class": Lion, "params": {}},
}

LR_SCHEDULERS = {
    "exponential_decay": {
        "class": ExponentialDecay,
        "params": {"decay_steps": 10000, "decay_rate": 0.9, "staircase": True}
    },
    "piecewise_constant_decay": {
        "class": PiecewiseConstantDecay,
        "params": {"boundaries": [10000, 20000], "values": [1e-3, 0.5e-3, 0.1e-3]}
    },
    "polynomial_decay": {
        "class": PolynomialDecay,
        "params": {"decay_steps": 10000, "end_learning_rate": 1e-5, "power": 1.0}
    },
    "inverse_time_decay": {
        "class": InverseTimeDecay,
        "params": {"decay_steps": 1.0, "decay_rate": 0.5}
    },
    "cosine_decay": {
        "class": CosineDecay,
        "params": {"decay_steps": 10000}
    }
}

def get_model_path(model_name: str, extension: str = ".keras") -> str:
    """
    Returns a unique path to save a model, including timestamp.
    Ensures every saved model has a base model name (no defaults).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}{extension}"
    return os.path.join(MODEL_DIR, filename)

def def_callbacks(logger, model_name: str) -> list:
    """Change the default callbacks here"""
    try:
        model_path = get_model_path(f"best_{model_name}")
        model_checkpoint = ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            mode="min",
            verbose=1
        )
        callbacks = [model_checkpoint, early_stopping, reduce_lr]
        logger.info(f"Default callbacks defined successfully. Best model will be saved to {model_path}")
        return callbacks
    except Exception as e:
        logger.error(f"Error defining callbacks: {e}")
        return []

def get_custom_layers(num_classes: int, dropout_rate: float = 0.3) -> list:
    return [
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")
    ]