import tensorflow as tf
import keras
from keras import layers
import re
from ..base_model import Base_Model
from ...config import get_custom_layers


class Model(Base_Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info("CNN Model initialized.")
        
    def get_base_model(self, name: str) -> 'Model':
        self.logger.info(f"Attempting to load CNN model: {name}")
        cleaned_name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
        
        model_class = self._MODEL_DICT.get(cleaned_name)
        if model_class:
            try:
                self.base_model = model_class(
                    weights="imagenet",
                    include_top=False,
                    input_shape=self.config.img_size + (3,)
                )
                self.outputs_layer = self.base_model.output
                self._rebuild_model()
                self.logger.info(f"Keras CNN model '{name}' loaded successfully.")
            except Exception as e:
                self.logger.error(f"Error loading Keras CNN '{name}': {e}")
        else:
            self.logger.error(f"CNN model '{name}' not found in Keras Applications.")
        return self
    
    def _add_layers(self, output_tensor: tf.Tensor, layers_list: list) -> tf.Tensor:
        x = output_tensor
        for l in layers_list:
            x = l(x)
        return x
        
    def add_custom_layers(self, layers_list: list = None) -> 'Model':
        if not self.base_model:
            self.logger.error("Base model not loaded. Cannot add layers.")
            return self
            
        if layers_list is None:
            layers_list = get_custom_layers(self.config.num_classes)
        
        self.outputs_layer = self._add_layers(self.outputs_layer, layers_list)
        self._rebuild_model()
        self.logger.info("Added custom classification head.")
        return self

    def freeze_all(self) -> 'Model':
        if not self.model: return self
        for layer in self.base_model.layers:
            layer.trainable = False
        self._rebuild_model()
        self.logger.info("All base model layers frozen.")
        return self

    def unfreeze_all(self) -> 'Model':
        if not self.model: return self
        for layer in self.base_model.layers:
            layer.trainable = True
        self._rebuild_model()
        self.logger.info("All base model layers unfrozen.")
        return self

    def freeze_upto_layer(self, layer_name: str) -> 'Model':
        if not self.model: return self
        try:
            for layer in self.base_model.layers:
                layer.trainable = False
                if layer.name == layer_name:
                    break
            self._rebuild_model()
            self.logger.info(f"Froze layers up to {layer_name}.")
        except Exception as e:
            self.logger.error(f"Error freezing up to layer {layer_name}: {e}")
        return self

    def unfreeze_later_n(self, n: int = None) -> 'Model':
        if not self.model: return self
        if n is None: n = self.config.N
        try:
            self.freeze_all()
            for layer in self.base_model.layers[-n:]:
                layer.trainable = True
            self._rebuild_model()
            self.logger.info(f"Unfroze last {n} layers.")
        except Exception as e:
            self.logger.error(f"Error unfreezing last {n} layers: {e}")
        return self

    def cut_at_layer(self, layer_name: str) -> 'Model':
        if not self.model: return self
        try:
            self.outputs_layer = self.model.get_layer(layer_name).output
            self._rebuild_model()
            self.logger.info(f"Model output cut at layer {layer_name}.")
        except Exception as e:
            self.logger.error(f"Error cutting at layer {layer_name}: {e}")
        return self
        
    def remove_n_layers(self, n: int = None) -> 'Model':
        if not self.model: return self
        if n is None: n = self.config.remove_N
        try:
            target_layer_name = self.model.layers[-(n + 1)].name
            self.cut_at_layer(target_layer_name)
            self.logger.info(f"Removed last {n} layers.")
        except IndexError:
             self.logger.error(f"Cannot remove {n} layers; model is not deep enough.")
        return self

    def remove_layer_in_between(self, layer_name: str) -> 'Model':
        if not self.model: return self
        try:
            target_layer = self.model.get_layer(layer_name)
            index = self.model.layers.index(target_layer)
            prev_layer = self.model.layers[index - 1]
            
            x = prev_layer.output
            for layer in self.model.layers[index + 1:]:
                x = layer(x)
                
            self.outputs_layer = x
            self._rebuild_model()
            self.logger.info(f"Removed layer {layer_name} and reconnected the network.")
        except Exception as e:
            self.logger.error(f"Error removing layer {layer_name}: {e}")
        return self