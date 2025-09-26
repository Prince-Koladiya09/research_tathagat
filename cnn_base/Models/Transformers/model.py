import tensorflow as tf
import keras
from keras import layers
from transformers import TFAutoModel, AdamWeightDecay
from ..base_model import Base_Model
import re

class Model(Base_Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("Transformer_Model initialized.")
        
    def get_base_model(self, name: str) -> 'Model':
        self.logger.info(f"Attempting to load Transformer model: {name}")
        hf_path = self._MODEL_DICT.get(name.lower())
        
        if not hf_path:
            self.logger.error(f"Transformer model '{name}' not found in registry. Searching on HuggingFace Hub.")
            hf_path = name
        
        try:
            hf_model = TFAutoModel.from_pretrained(hf_path, from_pt=True)
            input_tensor = layers.Input(shape=self.config.img_size + (3,), name="input_pixels")
            
            hf_output = hf_model(input_tensor, training=False).last_hidden_state
            self.outputs_layer = hf_output[:, 0, :]
            
            # Encapsulate the HuggingFace model in a Keras layer to make it serializable
            self.base_model = keras.Model(inputs=input_tensor, outputs=self.outputs_layer)
            self._rebuild_model()
            self.logger.info(f"Hugging Face transformer '{hf_path}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading transformer from Hugging Face '{hf_path}': {e}")
            
        return self
        
    def add_custom_layers(self, layers_list: list = None) -> 'Model':
        if not self.base_model:
            self.logger.error("Base model not loaded. Cannot add layers.")
            return self
            
        if layers_list is None:
            layers_list = [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(self.config.num_classes, activation="softmax", name="classifier")
            ]
        
        x = self.outputs_layer
        for l in layers_list:
            x = l(x)
        self.outputs_layer = x
        self._rebuild_model()
        self.logger.info("Added custom transformer classification head.")
        return self

    def freeze_all_but_head(self) -> 'Model':
        if not self.model: return self
        for layer in self.model.layers:
            if "classifier" not in layer.name:
                layer.trainable = False
        self.logger.info("Froze all layers except the classification head.")
        return self

    def freeze_all_but_biases(self) -> 'Model':
        if not self.model: return self
        for layer in self.model.layers:
            if hasattr(layer, 'trainable_weights'):
                for weight in layer.trainable_weights:
                    if 'bias' not in weight.name:
                        weight.trainable = False
        self.logger.info("Froze all weights except for bias terms (BitFit).")
        return self
    
    def unfreeze_all(self) -> 'Model':
        if not self.model: return self
        for layer in self.model.layers:
            layer.trainable = True
        self.logger.info("Unfroze all model layers.")
        return self
        
    def unfreeze_last_n_blocks(self, n_blocks: int) -> 'Model':
        if not self.model: return self
        
        block_regex = re.compile(r"(layer|block)_\._\d+$", re.IGNORECASE)
        transformer_layer = None
        for layer in self.model.layers:
             if 'main_layer' in layer.name and hasattr(layer, 'layers'):
                  transformer_layer = layer
                  break

        if not transformer_layer:
            self.logger.error("Could not find main transformer layer to unfreeze blocks.")
            return self
            
        self.logger.info("Freezing all transformer blocks initially.")
        transformer_layer.trainable = False

        blocks = [l for l in transformer_layer.layers if block_regex.search(l.name)]
        if len(blocks) < n_blocks:
            self.logger.warning(f"Requested to unfreeze {n_blocks}, but only {len(blocks)} blocks found. Unfreezing all.")
            n_blocks = len(blocks)

        for block in blocks[-n_blocks:]:
            block.trainable = True
            
        self.logger.info(f"Unfroze the last {n_blocks} transformer blocks.")
        return self
        
    def compile_with_llrd(self, head_lr_multiplier: float = 2.0, decay_rate: float = 0.75):
        self.logger.info("Compiling with Layer-wise Learning Rate Decay (LLRD).")
        if not self.model:
            self.logger.error("Model must be built before compiling.")
            return

        opt_config = self.config.optimizer
        if 'adamw' not in opt_config.name:
            self.logger.warning("LLRD is typically used with AdamW. Using the configured optimizer.")
            
        lr = self.config.learning_rate
        wd = opt_config.params.get("weight_decay", 0.01)

        block_regex = re.compile(r"(layer|block)_\._\d+$", re.IGNORECASE)
        
        learning_rate_multipliers = {}
        
        # Default for all variables
        for var in self.model.trainable_variables:
            learning_rate_multipliers[var.name] = 1.0

        transformer_layer = next((l for l in self.model.layers if 'main_layer' in l.name and hasattr(l, 'layers')), None)
        
        if transformer_layer:
            blocks = [l for l in transformer_layer.layers if block_regex.search(l.name)]
            num_blocks = len(blocks)
            
            # Apply decay from bottom to top
            for i, block in enumerate(blocks):
                multiplier = decay_rate ** (num_blocks - i)
                for var in block.trainable_variables:
                    learning_rate_multipliers[var.name] = multiplier
        
        # Apply head multiplier
        for layer in self.model.layers:
            if 'classifier' in layer.name:
                for var in layer.trainable_variables:
                    learning_rate_multipliers[var.name] = head_lr_multiplier

        optimizer = AdamWeightDecay(
            learning_rate=lr,
            weight_decay_rate=wd,
            learning_rate_multipliers=learning_rate_multipliers
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics
        )
        self.logger.info(f"Model compiled with LLRD. Base LR: {lr}, Decay Rate: {decay_rate}")
        return self