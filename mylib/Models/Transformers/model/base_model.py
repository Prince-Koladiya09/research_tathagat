import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from loggers import Logger
# from models import get_model
from copy import deepcopy
from config import DEFAULT_CONFIG, OPTIMIZER_MAP, get_custom_layers, get_model_path, def_callbacks


class Model_Train(keras.Model):
    def __init__(self, name: str = "custom_model", update_config_kwargs : dict = None, **kwargs):
        super().__init__(name=name, **kwargs)
    #     self.base_model = None
    #     self.model = None
    #     self.outputs_layer = None
    #     self.config = deepcopy(DEFAULT_CONFIG)
    #     if update_config_kwargs is not None :
    #         self.config.update(update_config_kwargs)
    #     self.logger = Logger()
    #     self.callbacks = def_callbacks(self.logger)
    #     self.logger.info("Transformers model initialized")


    # # ---------------- Model Handling ----------------
    # def rebuild_model(self) -> None:
    #     if self.base_model is None or self.outputs_layer is None:
    #         self.logger.error(f"Cannot rebuild model: {"base_model" if self.base_model is None else "outputs_layer"} missing")
    #         return
    #     try:
    #         self.model = Model(inputs=self.base_model.input, outputs=self.outputs_layer)
    #         self.logger.info("Model rebuilt successfully")
    #     except Exception as e:
    #         self.logger.error(f"Error rebuilding model: {e}")

    # def get_base_model(self, name: str):
    #     try:
    #         self.base_model = get_model(name)(
    #             weights="imagenet", include_top=False, input_shape=self.config["img_size"] + (3,)
    #         )
    #         self.outputs_layer = self.base_model.output
    #         self.rebuild_model()
    #         self.logger.info(f"Base model {name} loaded")
    #     except Exception as e:
    #         self.logger.error(f"Error loading base model {name}: {e}")

    # def compile(self, lr: float = None) -> None:
    #     try:
    #         if lr is None :
    #             lr = self.config["learning_rate"]
    #         super().compile(
    #             optimizer=OPTIMIZER_MAP[self.config["optimizer"]](learning_rate=lr),
    #             loss=self.config["loss"],
    #             metrics=self.config["metrics"]
    #         )
    #         self.logger.info(f"Model compiled with learning rate {lr}")
    #     except Exception as e:
    #         self.logger.error(f"Error compiling model: {e}")

    # def summary(self) -> None:
    #     if self.model:
    #         self.model.summary()
    #         self.logger.info("Model summary printed")
    #     else:
    #         self.logger.error("No model to summarize")

    # def call(self, inputs, training=False):
    #     return self.model(inputs, training=training)

    # # ---------------- Saving & Fitting ----------------
    # def save(self, file_name: str = None):
    #     try:
    #         if file_name is None:
    #             file_name = get_model_path(self.base_model.name)
    #         super().save(file_name)
    #         self.logger.info(f"Model saved as {file_name}")
    #     except Exception as e:
    #         self.logger.error(f"Error saving model: {e}")

    # def fit(self, train, val, epochs: int = None, batch_size: int = None, callbacks : list = None):
    #     try:
    #         if callbacks is None :
    #             callbacks = self.callbacks
    #         if epochs is None :
    #             epochs = self.config["epochs"]
    #         if batch_size is None :
    #             batch_size = self.config["batch_size"]

    #         self.logger.info(f"Training started for {epochs} epochs")
    #         return super().fit(
    #             train,
    #             validation_data=val,
    #             epochs=epochs,
    #             batch_size=batch_size,
    #             callbacks=callbacks
    #         )
    #     except Exception as e:
    #         self.logger.error(f"Error during training: {e}")
    

    # def predict(self, data, batch_size: int = None) :
    #     if batch_size is None :
    #         batch_size = self.config["batch_size"]
    #     return super().predict(data, batch_size=batch_size)
    

    # def evaluate(self, data, batch_size: int = None) :
    #     if batch_size is None :
    #         batch_size = self.config["batch_size"]
    #     return super().evaluate(data, batch_size=batch_size)

    # # ----------------- Freeze / Unfreeze ----------------- #
    # def freeze_all(self):
    #     for layer in self.model.layers:
    #         layer.trainable = False
    #     self.logger.info("All layers frozen")

    # def freeze_early_N(self, N: int = None):
    #     if N is None :
    #         N = self.config["N"]
    #     for layer in self.model.layers[N:]:
    #         layer.trainable = False
    #     self.logger.info(f"Froze all layers after first {N}")

    # def freeze_upto_layer(self, layer_name: str = None):
    #     for layer in self.model.layers:
    #         layer.trainable = False
    #         if layer.name == layer_name:
    #             break
    #     self.logger.info(f"Froze layers up to {layer_name}")

    # def unfreeze_later_N(self, N: int = None):
    #     if N is None :
    #         N = self.config["N"]
    #     for layer in self.model.layers[-N:]:
    #         layer.trainable = True
    #     self.logger.info(f"Unfroze last {N} layers")

    # def unfreeze_after_layer(self, layer_name: str = None):
    #     freeze = True
    #     for layer in self.model.layers:
    #         layer.trainable = not freeze
    #         if layer.name == layer_name:
    #             freeze = False
    #     self.logger.info(f"Unfroze layers after {layer_name}")

    # def unfreeze_all(self):
    #     for layer in self.model.layers:
    #         layer.trainable = True
    #     self.logger.info("All layers unfrozen")

    # # ----------------- Add Layers ----------------- #
    # def add_custom_layers_after_a_layer(self, existing_layer, layers) -> keras.layers:
    #     x = existing_layer
    #     for l in layers:
    #         x = l(x)
    #     return x

    # def add_custom_layers(self, layers_list=None):
    #     if layers_list is None:
    #         layers_list = get_custom_layers(self.config["num_classes"])
    #     self.outputs_layer = self.add_custom_layers_after_a_layer(self.outputs_layer, layers_list)
    #     self.rebuild_model()
    #     self.logger.info("Added custom layers at the end")

    # def add_layers_in_between(self, layer_name: str, layers):
    #     target_layer = self.model.get_layer(layer_name)
    #     index = self.model.layers.index(target_layer)
    #     x = self.add_custom_layers_after_a_layer(target_layer.output, layers)
    #     x = self.add_custom_layers_after_a_layer(x.output, self.model.layers[index + 1:])
    #     self.outputs_layer = x
    #     self.rebuild_model()
    #     self.logger.info(f"Added custom layers after {layer_name}")

    # # ----------------- Cut Layers ----------------- #
    # def cut_at_layer(self, layer_name: str):
    #     self.outputs_layer = self.model.get_layer(layer_name).output
    #     self.rebuild_model()
    #     self.logger.info(f"Cut model at {layer_name}")

    # def cut_at_layer_and_add_custom_layers(self, layer_name: str, layers_list=None):
    #     if layers_list is None:
    #         layers_list = get_custom_layers(self.config["num_classes"])
    #     outputs = self.model.get_layer(layer_name).output
    #     self.outputs_layer = self.add_custom_layers_after_a_layer(outputs, layers_list)
    #     self.rebuild_model()
    #     self.logger.info(f"Cut model at {layer_name} and added custom layers")

    # # ----------------- Remove Layers ----------------- #
    # def remove_last_layer(self):
    #     last_second_layer = self.model.layers[-2]
    #     self.outputs_layer = last_second_layer.output
    #     self.rebuild_model()
    #     self.logger.info("Removed last layer")

    # def remove_N_layers(self, N: int = None):
    #     if N is None :
    #         N = self.config["remove_N"]
    #     self.cut_at_layer(self.model.layers[-(N + 1)].name)
    #     self.logger.info(f"Removed last {N} layers")

    # def remove_layer_in_between(self, layer_name: str):
    #     target_layer = self.model.get_layer(layer_name)
    #     index = self.model.layers.index(target_layer)
    #     prev_layer = self.model.layers[index - 1]
    #     self.outputs_layer = self.add_custom_layers_after_a_layer(prev_layer.output, self.model.layers[index + 1:])
    #     self.rebuild_model()
    #     self.logger.info(f"Removed {layer_name} layer and reconnected network")