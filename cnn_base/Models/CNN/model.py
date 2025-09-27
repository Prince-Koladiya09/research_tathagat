import tensorflow as tf
from typing import List
import keras

from ..base_model import Base_Model
from .providers import get_model as get_cnn_layers
from ...configs.base_config import get_custom_layers
from ...configs.cnn_config import DEFAULT_CNN_CONFIG

class Model(Base_Model):
    def __init__(self, name: str = "cnn_model", **kwargs):
        super().__init__(name=name, config=DEFAULT_CNN_CONFIG, **kwargs)

    def get_base_model(self, name: str) -> 'Model':
        try:
            inputs, outputs = get_cnn_layers(name, img_size=self.config.model.img_size)
            self.base_model = keras.Model(inputs, outputs, name=name)
            self.outputs_layer = self.base_model.output
            self._rebuild_model()
            self.logger.info(f"CNN base model '{name}' built successfully.")
        except Exception as e:
            self.logger.error(f"Error building CNN base model '{name}': {e}")
        return self

    def _set_trainable_status(self, layers_to_modify: List[keras.layers.Layer], trainable: bool):
        count = 0
        for layer in layers_to_modify:
            try:
                if layer.trainable != trainable:
                    layer.trainable = trainable
                    count += 1
            except Exception as e:
                self.logger.error(f"Could not set trainable={trainable} for layer {layer.name}: {e}")
        status = "unfrozen" if trainable else "frozen"
        self.logger.info(f"{count} layers have been {status}.")

    def freeze_all(self) -> 'Model':
        self._set_trainable_status(self.base_model.layers, trainable=False)
        return self

    def unfreeze_all(self) -> 'Model':
        self._set_trainable_status(self.base_model.layers, trainable=True)
        return self

    def unfreeze_later_n(self, n: int = None) -> 'Model':
        if n is None:
            n = self.config.model.n_layers_to_tune
        if n > len(self.base_model.layers):
             self.logger.warning(f"Requested to unfreeze {n} layers, but model only has {len(self.base_model.layers)}. Unfreezing all.")
             n = len(self.base_model.layers)
        self._set_trainable_status(self.base_model.layers[-n:], trainable=True)
        return self
    
    def freeze_upto_layer(self, layer_name: str) -> 'Model':
        try:
            layer_found = False
            for layer in self.base_model.layers:
                layer.trainable = False
                if layer.name == layer_name:
                    layer_found = True
                    break
            if layer_found:
                 self.logger.info(f"Froze layers up to and including '{layer_name}'.")
            else:
                 self.logger.error(f"Layer '{layer_name}' not found in the model. Froze all")
        except Exception as e:
            self.logger.error(f"Error during freeze_upto_layer: {e}")
        return self
    
    def add_custom_layers(self, layers_list: list = None) -> 'Model':
        try:
            if layers_list is None:
                layers_list = get_custom_layers(self.config.model.num_classes)
            
            x = self.outputs_layer
            for layer in layers_list:
                x = layer(x)
            self.outputs_layer = x
            
            self._rebuild_model()
            self.logger.info(f"Added {len(layers_list)} custom layers to the model head.")
        except Exception as e:
            self.logger.error(f"Error adding custom layers: {e}")
        return self

    def cut_at_layer_and_add_custom_layers(self, layer_name: str, layers_list: list = None) -> 'Model':
        try:
            if layers_list is None:
                layers_list = get_custom_layers(self.config.model.num_classes)
            
            target_output = self.model.get_layer(layer_name).output
            x = target_output
            for layer in layers_list:
                x = layer(x)
            self.outputs_layer = x
            
            self._rebuild_model()
            self.logger.info(f"Cut model at '{layer_name}' and added {len(layers_list)} custom layers.")
        except Exception as e:
            self.logger.error(f"Error cutting and adding layers at '{layer_name}': {e}")
        return self