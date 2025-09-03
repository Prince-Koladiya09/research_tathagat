import os
from datetime import datetime
from keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import Accuracy, Precision, Recall, AUC
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Root directory of the project
PROJECT_ROOT = os.getcwd()

# Storage directories
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
MODEL_DIR = os.path.join(STORAGE_DIR, "models")
LOG_DIR = os.path.join(STORAGE_DIR, "logs")

# Make sure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

OPTIMIZER_MAP = {
    "adam" : Adam,
    "sgd" : SGD,
    "rmsprop" : RMSprop,
}

# Default training hyperparameters
DEFAULT_CONFIG = {
    "cnn" : {
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": [Accuracy(name = "accuracy"),
                    Precision(name="precision"),
                    Recall(name="recall"),
                    AUC(name="auc")],
        "seed" : 42,
        "num_classes" : 4,
        "img_size" : (224, 224),
        "N" : 20,
        "remove_N" : 1,
        "learning_rate" : 1e-3,
    },
    "transformers" : {}
}




def get_model_path(model_name: str, extension: str = ".keras") -> str:
    """
    Returns a unique path to save a model, including timestamp.
    Ensures every saved model has a base model name (no defaults).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(MODEL_DIR, f"{model_name}_{timestamp}{extension}")


def def_callbacks(logger) -> list:
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
        callbacks = [model_checkpoint, early_stopping, reduce_lr]
        logger.info("Callbacks defined successfully")
        return callbacks
    except Exception as e:
        logger.error(f"Error defining callbacks: {e}")