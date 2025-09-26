import keras
import re

class Progressive_Unfreezer(keras.callbacks.Callback):
    def __init__(self, logger, interval: int = 1):
        super().__init__()
        self.interval = interval
        self.unfreeze_block_count = 0
        self.logger = logger
        self.transformer_blocks = []

    def on_train_begin(self, logs=None):
        try:
            # Find all transformer blocks (layers) in the model, usually indexed
            # Assumes huggingface naming convention e.g., 'tf_bert_layer_._0', 'tf_bert_layer_._1', ...
            block_regex = re.compile(r"(layer|block|resblock)_\._\d+$", re.IGNORECASE)

            # Identify layers belonging to the encoder/transformer body
            transformer_body = None
            for layer in self.model.layers:
                if "main_layer" in layer.name and hasattr(layer, 'layers'):
                    transformer_body = layer
                    break

            if transformer_body:
                self.transformer_blocks = [l for l in transformer_body.layers if block_regex.search(l.name)]
                self.transformer_blocks.reverse() # Start unfreezing from the top

            if not self.transformer_blocks:
                self.logger.warning("[ProgressiveUnfreezer] Could not find transformer blocks to unfreeze.")
                return

            # Initially freeze all transformer blocks
            for block in self.transformer_blocks:
                block.trainable = False

            # Ensure head and embeddings are trainable
            for layer in self.model.layers:
                if 'pooler' in layer.name or 'classifier' in layer.name or 'embeddings' in layer.name:
                     layer.trainable = True

            self.logger.info(f"[ProgressiveUnfreezer] Initial state: All {len(self.transformer_blocks)} blocks frozen.")

            # A re-compile is necessary to apply trainable changes
            self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)

        except Exception as e:
            self.logger.error(f"[ProgressiveUnfreezer] Error on_train_begin: {e}")


    def on_epoch_begin(self, epoch, logs=None):
        if not self.transformer_blocks:
            return

        try:
            if epoch > 0 and epoch % self.interval == 0:
                self.unfreeze_block_count += 1
                unfreeze_upto = min(self.unfreeze_block_count, len(self.transformer_blocks))

                if self.transformer_blocks[unfreeze_upto - 1].trainable:
                    self.logger.info(f"[ProgressiveUnfreezer] Epoch {epoch+1}: Block {unfreeze_upto-1} already unfrozen.")
                    return

                block_to_unfreeze = self.transformer_blocks[unfreeze_upto - 1]
                block_to_unfreeze.trainable = True
                
                self.logger.info(
                    f"[ProgressiveUnfreezer] Epoch {epoch+1}: Unfreezing transformer block {len(self.transformer_blocks) - unfreeze_upto} ({block_to_unfreeze.name})"
                )

                # Re-compile to apply changes
                self.model.compile(
                    optimizer=self.model.optimizer,
                    loss=self.model.loss,
                    metrics=self.model.metrics,
                )
        except Exception as e:
            self.logger.error(
                f"[ProgressiveUnfreezer] Error during unfreezing at epoch {epoch+1}: {e}"
            )