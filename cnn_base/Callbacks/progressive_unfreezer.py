from tensorflow import keras

class ProgressiveUnfreezer(keras.callbacks.Callback):
    def __init__(self, logger, block_size: int = 10, interval: int = 5):
        super().__init__()
        self.block_size = block_size
        self.interval = interval
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        try:
            if epoch % self.interval == 0 and epoch > 0:
                # Unfreeze next block
                num_unfrozen = min(
                    self.block_size * (epoch // self.interval),
                    len(self.model.layers)
                )
                for layer in self.model.layers[-num_unfrozen:]:
                    layer.trainable = True

                # Need to recompile to take effect
                self.model.compile(
                    optimizer=self.model.optimizer.__class__.from_config(
                        self.model.optimizer.get_config()
                    ),
                    loss=self.model.loss,
                    metrics=self.model.metrics,
                )

                self.logger.info(
                    f"[ProgressiveUnfreezer] Unfroze last {num_unfrozen} layers at epoch {epoch}"
                )
        except Exception as e:
            self.logger.error(
                f"[ProgressiveUnfreezer] Error unfreezing at epoch {epoch}: {e}"
            )