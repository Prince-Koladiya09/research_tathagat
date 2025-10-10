import tensorflow as tf
from typing import List
import keras
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

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
            self.freeze_all()

            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dense(128, activation = "relu")(x)
            x = Dropout(0.3)(x)
            x = Dense(self.config.model.num_classes, activation = "softmax")(x)
            self.outputs_layer = x
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

    def freeze_early_n(self, n: int = None) -> 'Model':
        if n is None:
            n = self.config.model.n_layers_to_tune
        if n > len(self.base_model.layers):
             self.logger.warning(f"Requested to freeze {n} layers, but model only has {len(self.base_model.layers)}. Freezing all.")
             n = len(self.base_model.layers)
        self._set_trainable_status(self.base_model.layers[:n], trainable=False)
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
    
    def unfreeze_after_layer(self, layer_name: str) -> 'Model':
        try:
            layer_found = False
            for layer in self.base_model.layers:
                if layer.name == layer_name:
                    layer_found = True
                layer.trainable = layer_found
            if layer_found:
                 self.logger.info(f"Froze layers up to '{layer_name}' and unfroze all after and including it.")
            else:
                 self.logger.error(f"Layer '{layer_name}' not found in the model. Froze all")
        except Exception as e:
            self.logger.error(f"Error during unfreeze_after_layer : {e}")
        return self
    
    def add_custom_layers(self, layers_list: list = None) -> 'Model':
        """
        layers : list of keras.layers.Layer
                    If given, then adds these layer after layer_name
                    If not given, adds custom layers from config
                    after adding these, adds the remaining layers from the model
        """
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

    def add_layers_in_between(self, layer_name : str, layers : list[keras.layers.Layer]) -> 'Model' :
        """
        layers : list of keras.layers.Layer
                    If given, then adds these layer after layer_name
                    If not given, adds custom layers from config
                    after adding these, adds the remaining layers from the model
        """
        try :
            if layers_list is None:
                layers_list = get_custom_layers(self.config.model.num_classes)
            
            target_layer = self.model.get_layer(layer_name)
            index = self.model.layers.index(target_layer)

            x = target_layer.output

            for layer in layers_list :
                x = layer(x)

            for layer in self.model.layers[index + 1 :] :
                x = layer(x)

            self.outputs_layer = x

            self._rebuild_model()
            self.logger.info(f"Added custom layers after {layer_name}")
        except Exception as e :
            self.logger.error(e)
        return self

    def cut_at_layer_and_add_custom_layers(self, layer_name: str, layers_list: list = None) -> 'Model':
        try:
            if layers_list is None:
                layers_list = get_custom_layers(self.config.model.num_classes)
            
            x = self.model.get_layer(layer_name).output
            for layer in layers_list:
                x = layer(x)
            self.outputs_layer = x
            
            self._rebuild_model()
            self.logger.info(f"Cut model at '{layer_name}' and added {len(layers_list)} custom layers.")
        except Exception as e:
            self.logger.error(f"Error cutting and adding layers at '{layer_name}': {e}")
        return self

    def cut_at_layer(self, layer_name: str) -> 'Model':
        try:
            self.outputs_layer = self.model.get_layer(layer_name).output
            
            self._rebuild_model()
            self.logger.info(f"Cut model at '{layer_name}'.")
        except Exception as e:
            self.logger.error(f"Error cutting at '{layer_name}': {e}")
        return self

    def remove_last_layer(self) -> 'Model' :
        try :
            last_second_layer = self.model.layers[-2]
            self.outputs_layer = last_second_layer.output
            self._rebuild_model()
            self.logger.info("Removed last layer")
        except Exception as e :
            self.logger.error(e)
        return self

    def remove_N_layers(self, n : int = None) -> 'Model' :
        """
        If you do not provide N, it takes default from config
        """
        try :
            if n is None:
                n = self.config.model.n_layers_to_tune
            if n > len(self.base_model.layers):
                self.logger.warning(f"Requested to remove {n} layers, but model only has {len(self.base_model.layers)}. Removing none.")
                return self
            
            remove_after_layer = self.model.layers[-(n + 1)].name
            self.outputs_layer = self.model.get_layer(remove_after_layer).output
            self.logger.info(f"Removed last {n} layers")
        except Exception as e :
            self.logger.error(e)
        return self

    def remove_layer_in_between(self, layer_name : str) -> 'Model' :
        try :
            target_layer = self.model.get_layer(layer_name)
            index = self.model.layers.index(target_layer)
            prev_layer = self.model.layers[index - 1]

            x = prev_layer.output
            for layer in self.model.layers[index + 1 :]:
                x = layer(x)
            self.outputs_layer = x

            self._rebuild_model()
            self.logger.info(f"Removed {layer_name} layer and reconnected network")
        except Exception as e :
            self.logger.error(e)
        return self