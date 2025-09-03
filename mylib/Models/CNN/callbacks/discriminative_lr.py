from tensorflow import keras

class DiscriminativeLRScheduler(keras.callbacks.Callback):
    def __init__(self, logger, base_lr=1e-4, multiplier=2.0):
        super().__init__()
        self.base_lr = base_lr
        self.multiplier = multiplier
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        try :
            # progressively assign LR depending on layer depth
            for i, layer in enumerate(self.model.layers):
                lr = self.base_lr * (self.multiplier ** (i / len(self.model.layers)))
                if hasattr(layer, "kernel"):
                    keras.backend.set_value(self.model.optimizer.lr, lr)
            self.logger.info(f"[DiscriminativeLRScheduler] Epoch {epoch} - LR updated")
        except Exception as e :
            self.logger.error(f"[DiscriminativeLRScheduler] Error while updateing error during epoch {epoch}")