import os
from datetime import datetime
from copy import deepcopy

from keras import layers

from keras.optimizers import (
    Adam, AdamW, Nadam, Adagrad, Adamax, Adadelta,
    SGD, RMSprop,
    Lamb, Lion
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

from .Callbacks import (
    ProgressiveUnfreezer, DiscriminativeLRScheduler
)

# Root directory of the project
PROJECT_ROOT = os.getcwd()

# Storage directories
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
MODEL_DIR = os.path.join(STORAGE_DIR, "models")
LOG_DIR = os.path.join(STORAGE_DIR, "logs")

# Make sure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Global Training Configuration ---
CONFIG = {
    "batch_size": 32,
    "epochs": 10,
    "seed": 42,
    "num_classes": 4,
    "img_size": (224, 224),
    
    "optimizer": "adam",
    "optimizer_params": {}, # Overrides default optimizer params
    "learning_rate": 1e-3,

    "lr_scheduler": None, # e.g., "cosine_decay"
    "lr_scheduler_params": {}, # Overrides default scheduler params

    "loss": "sparse_categorical_crossentropy",
    "metrics": [
        Accuracy(name="accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ],

    # For custom model methods
    "N": 20,
    "remove_N": 1,
}

# --- Optimizer Definitions ---
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

# --- Learning Rate Scheduler Definitions ---
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

def get_model_path(model_name : str, extensions : tuple[str] = (".keras", "pkl")) -> str:
    """
    Returns a unique path to save a model, including timestamp.
    Ensures every saved model has a base model name (no defaults).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}")
    return tuple(os.path.join(name, extension) for extension in extensions)


def def_callbacks(logger) -> list:
    """Change the default callbacks here"""
    try:
        model_checkpoint = ModelCheckpoint(
            filepath=get_model_path("best_model"),
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
            min_lr=0.00001,
            mode="min",
            verbose=1
        )
        # Use custom callbacks here if you want
        # progressive_unfreezer = ProgressiveUnfreezer(logger = logger)
        # discriminative_lr_scheduler = DiscriminativeLRScheduler(logger = logger)
        callbacks = [model_checkpoint, early_stopping, reduce_lr]
        logger.info("Callbacks defined successfully")
        return callbacks
    except Exception as e:
        logger.error(f"Error defining callbacks: {e}")


def get_custom_layers(num_classes : int) -> list :
    """Change the layers here you want to add while fine tuning"""
    return [
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(CONFIG["num_classes"], activation="softmax")
    ]