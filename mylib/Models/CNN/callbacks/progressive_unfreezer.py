from tensorflow import keras

class ProgressiveUnfreezer(keras.callbacks.Callback):
    def __init__(self, model, logger, block_size : int = 10, interval : int = 5):
        super().__init__()
        self.model = model
        self.block_size = block_size
        self.interval = interval
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        try :
            if epoch % self.interval == 0 and epoch > 0:
                # Unfreeze next block
                num_unfrozen = self.block_size * (epoch // self.interval)
                for layer in self.model.layers[-num_unfrozen:]:
                    layer.trainable = True
                self.logger.info(f"[ProgressiveUnfreezer] Unfroze last {num_unfrozen} layers at epoch {epoch}")
        except Exception as e :
            self.logger.error(f"[ProgressiveUnfreezer] Error Unfreezing last {num_unfrozen} layers at epoch {epoch}")