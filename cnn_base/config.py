import os
from datetime import datetime
from keras import layers

from keras.optimizers import (
    Adam, AdamW, Nadam, Adagrad, Adamax, Adadelta,
    SGD, RMSprop,
    Lamb, Lion
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

# Default training hyperparameters
DEFAULT_CONFIG = {
    "batch_size": 32,
    "epochs": 10,

    "optimizer": "adam", #  pick from the optimzer map below
    "learning_rate" : 1e-3,
    "momentum" : 0,
    "weight_decay" : 1e-5,

    "loss": "sparse_categorical_crossentropy",
    "metrics": [
        Accuracy(name = "accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ],

    "seed" : 42,
    "num_classes" : 4,
    "img_size" : (224, 224),

    "N" : 20,
    "remove_N" : 1,
}

# default parameters for every optimizer
PARAMS = {
    "learning_rate" : DEFAULT_CONFIG["learning_rate"],
}

# remove None and add a dict of extra params you want to add to the corresponding optimizer
OPTIMIZER_MAP = { # [optimizer, parameters other than ones in params]
    "sgd":  [SGD, dict( # update here for more params
        momentum = DEFAULT_CONFIG["momentum"],
                )],
    "adam":  [Adam, None],
    "rmsprop":  [RMSprop, dict( # update here for more params
        momentum = DEFAULT_CONFIG["momentum"],
        )],

    "nadam":  [Nadam, None],
    "adamw":  [AdamW, dict( # update here for more params
        weight_decay = DEFAULT_CONFIG["weight_decay"],
        )],
    "adagrad":  [Adagrad, None],
    "adadelta":  [Adadelta, None],
    "adamax":  [Adamax, None],

    "lamb":  [Lamb, None],
    "lion":  [Lion, None],
}

CONFIG = DEFAULT_CONFIG.copy()


def get_model_path(model_name: str, extension: str = ".keras") -> str:
    """
    Returns a unique path to save a model, including timestamp.
    Ensures every saved model has a base model name (no defaults).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(MODEL_DIR, f"{model_name}_{timestamp}{extension}")


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
        layers.Dense(DEFAULT_CONFIG["num_classes"], activation="softmax")
    ]