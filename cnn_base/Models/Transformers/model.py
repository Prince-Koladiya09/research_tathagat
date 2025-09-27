import tensorflow as tf
import keras
from keras.optimizers import AdamW

from ..base_model import Base_Model
from .providers import get_model as get_transformer_layers
from ...configs.transformers_config import DEFAULT_TRANSFORMER_CONFIG

class Model(Base_Model):
    def __init__(self, name: str = "transformer_model", **kwargs):
        super().__init__(name=name, config=DEFAULT_TRANSFORMER_CONFIG, **kwargs)

    def get_base_model(self, name: str) -> 'Model':
        try:
            inputs, outputs = get_transformer_layers(name, img_size=self.config.model.img_size)
            self.base_model = keras.Model(inputs, outputs, name=name)
            
            x = keras.layers.LayerNormalization(epsilon=1e-6)(self.base_model.output)
            x = keras.layers.Dense(self.config.model.num_classes, activation="softmax", name="predictions")(x)
            self.outputs_layer = x

            self._rebuild_model()
            self.logger.info(f"Transformer base model '{name}' built successfully with a new prediction head.")
        except Exception as e:
            self.logger.error(f"Error building Transformer base model '{name}': {e}")
        return self

    def freeze_patch_embeddings(self) -> 'Model':
        try:
            for layer in self.base_model.layers:
                if 'patch_embeddings' in layer.name or 'embedding' in layer.name:
                    layer.trainable = False
            self.logger.info("Froze patch embedding layers.")
        except Exception as e:
            self.logger.error(f"Error freezing patch embeddings: {e}")
        return self
    
    def unfreeze_all(self) -> 'Model':
        for layer in self.base_model.layers:
            layer.trainable = True
        self.logger.info("Unfroze all transformer layers.")
        return self

    def unfreeze_last_n_blocks(self, n_blocks: int) -> 'Model':
        try:
            transformer_layers = [l for l in self.base_model.layers if 'transformer_block' in l.name or 'layer' in l.name]
            
            for layer in self.base_model.layers:
                layer.trainable = False 
            
            if n_blocks > len(transformer_layers):
                 self.logger.warning(f"Requested to unfreeze {n_blocks}, but found only {len(transformer_layers)}. Unfreezing all of them.")
                 n_blocks = len(transformer_layers)

            for layer in transformer_layers[-n_blocks:]:
                layer.trainable = True

            for layer in self.model.layers:
                if 'normalization' in layer.name or 'predictions' in layer.name:
                    layer.trainable = True

            self.logger.info(f"Unfroze the last {n_blocks} transformer blocks and the head.")
        except Exception as e:
            self.logger.error(f"Error unfreezing last {n_blocks} blocks: {e}")
        return self
    
    def compile_with_llrd(self, head_lr_multiplier: float = 2.0, decay_rate: float = 0.75) -> 'Model':
        self.logger.info(f"Setting up optimizer with Layer-wise Learning Rate Decay (LLRD).")
        
        opt_config = self.config.optimizer
        learning_rate = opt_config.learning_rate
        weight_decay = opt_config.params.get("weight_decay", 0.01)

        layer_lrs = {}
        
        encoder_layers = [l for l in self.base_model.layers if 'transformer_block' in l.name or 'layer' in l.name]
        num_encoder_layers = len(encoder_layers)
        
        for i, layer in enumerate(encoder_layers):
            lr_scale = decay_rate ** (num_encoder_layers - i - 1)
            layer_lrs[layer.name] = learning_rate * lr_scale
        
        embedding_layer_names = [l.name for l in self.base_model.layers if 'embedding' in l.name]
        for name in embedding_layer_names:
            layer_lrs[name] = learning_rate * (decay_rate ** num_encoder_layers)

        head_layer_names = [l.name for l in self.model.layers if l not in self.base_model.layers]
        for name in head_layer_names:
            layer_lrs[name] = learning_rate * head_lr_multiplier

        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        optimizer.learning_rate.assign(learning_rate)
        optimizer.add_variable_with_custom_gradient(
            'learning_rate_multipliers',
            initial_value=layer_lrs,
            dtype=tf.float32,
        )
        
        try:
            self.model.compile(
                optimizer=optimizer,
                loss=self.config.training.loss,
                metrics=self.config.training.metrics
            )
            self.logger.info("Model compiled successfully with LLRD optimizer.")
        except Exception as e:
            self.logger.error(f"Error compiling model with LLRD: {e}")
            
        return self