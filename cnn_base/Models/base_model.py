import tensorflow as tf
import keras
from keras import layers
from keras.models import Model as Keras_Model
from cnn_base.loggers import Logger
from copy import deepcopy
from .get_model import get_model
from cnn_base.config import CONFIG, OPTIMIZERS, LR_SCHEDULERS, get_custom_layers, get_model_path, def_callbacks


class Model(keras.Model) :
    def __init__(self, name : str = "custom_model", update_config_kwargs : dict = None, **kwargs) :
        super().__init__(name=name, **kwargs)
        self.base_model = None
        self.model = None
        self.outputs_layer = None
        self.config = deepcopy(CONFIG)
        if update_config_kwargs is not None :
            self.config.update(update_config_kwargs)
        self.logger = Logger()
        self.callbacks = def_callbacks(self.logger)
        self.logger.info("CNN model initialized")

    @staticmethod
    def update_global_config(updates: dict) -> None :
        """
        Updates the global CONFIG dictionary.
        This affects all future Model instances created without custom overrides.
        """
        CONFIG.update(updates)
        print("Global config updated.")


    # ---------------- Model Handling ----------------
    def get_base_model(self, name : str) :
        """
        Used to get the base keras model, example resnet50
        Just give the name
        regex used to filter the name : '[^a-zA-Z0-9]'
        """
        try :
            self.base_model = get_model(self.logger, name)(
                weights="imagenet", include_top=False, input_shape=self.config["img_size"] + (3,)
            )
            self.outputs_layer = self.base_model.output
            self.rebuild_model()
            self.logger.info(f"Base model {name} loaded")
        except Exception as e :
            self.logger.error(f"Error loading base model {name} : {e}")

    def rebuild_model(self) -> None :
        if self.base_model is None or self.outputs_layer is None :
            self.logger.error(f"Cannot rebuild model : {"base_model" if self.base_model is None else "outputs_layer"} missing")
            return
        try :
            self.model = Keras_Model(inputs=self.base_model.input, outputs=self.outputs_layer)
            self.logger.info("Model rebuilt successfully")
        except Exception as e :
            self.logger.error(f"Error rebuilding model : {e}")

    def compile(self) -> None :
        """
        Compiles the model using the optimizer and learning rate (or scheduler)
        specified in the instance's configuration (if not provided).
        """
        try:
            learning_rate_or_schedule = self.config["learning_rate"]
            scheduler_name = self.config.get("lr_scheduler")

            if scheduler_name and scheduler_name in LR_SCHEDULERS:
                scheduler_info = deepcopy(LR_SCHEDULERS[scheduler_name])
                scheduler_class = scheduler_info["class"]
                scheduler_params = scheduler_info["params"]
                
                # Override default scheduler params with user-defined ones
                scheduler_params.update(self.config.get("lr_scheduler_params", {}))

                # Some schedulers require the initial learning rate
                if "initial_learning_rate" in scheduler_params or scheduler_name == "piecewise_constant_decay":
                    # For PiecewiseConstantDecay, values can be relative to the learning rate
                    if scheduler_name != "piecewise_constant_decay":
                        scheduler_params["initial_learning_rate"] = self.config["learning_rate"]

                learning_rate_or_schedule = scheduler_class(**scheduler_params)
                self.logger.info(f"Using LR Scheduler: {scheduler_name}")
            else:
                self.logger.info(f"Using constant learning rate: {learning_rate_or_schedule}")

            # 2. Instantiate the Optimizer
            optimizer_name = self.config["optimizer"]
            if optimizer_name not in OPTIMIZERS:
                raise ValueError(f"Optimizer '{optimizer_name}' not found in configuration.")

            optimizer_info = deepcopy(OPTIMIZERS[optimizer_name])
            optimizer_class = optimizer_info["class"]
            optimizer_params = optimizer_info["params"]

            # Add learning rate and any user overrides
            optimizer_params["learning_rate"] = learning_rate_or_schedule
            optimizer_params.update(self.config.get("optimizer_params", {}))

            optimizer_instance = optimizer_class(**optimizer_params)
            
            # 3. Compile the Keras Model
            if not self.model:
                self.logger.error("Cannot compile: self.model is not built yet.")
                return

            self.model.compile(
                optimizer=optimizer_instance,
                loss=self.config["loss"],
                metrics=self.config["metrics"]
            )
            self.logger.info(f"Model compiled with Optimizer: {optimizer_name}")
        except Exception as e :
            self.logger.error(f"Error compiling model : {e}")

    def summary(self) -> None :
        if self.model :
            self.model.summary()
            self.logger.info("Model summary printed")
        else :
            self.logger.error("No model to summarize")

    def call(self, inputs, training=False) :
        return self.model(inputs, training=training)

    # ---------------- Saving & Fitting ----------------
    def save(self, file_name : str = None) :
        """
        If you do not provide file name, it makes default one from config
        """
        try :
            if file_name is None :
                file_name = get_model_path(self.base_model.name)
            super().save(file_name)
            self.logger.info(f"Model saved as {file_name}")
        except Exception as e :
            self.logger.error(f"Error saving model : {e}")

    def fit(self, train, val, epochs : int = None, batch_size : int = None, callbacks : list = None) :
        """
        If you do not provide parameters, it takes default ones from config
        """
        try :
            if callbacks is None :
                callbacks = self.callbacks
            if epochs is None :
                epochs = self.config["epochs"]
            if batch_size is None :
                batch_size = self.config["batch_size"]

            self.logger.info(f"Training started for {epochs} epochs")
            return super().fit(
                train,
                validation_data=val,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        except Exception as e :
            self.logger.error(f"Error during training : {e}")
    

    def predict(self, data, batch_size : int = None) :
        """
        If you do not provide batch size, it takes default from config
        """
        if batch_size is None :
            batch_size = self.config["batch_size"]
        return super().predict(data, batch_size=batch_size)
    

    def evaluate(self, data, batch_size : int = None) :
        """
        If you do not provide batch size, it takes default from config
        """
        if batch_size is None :
            batch_size = self.config["batch_size"]
        return super().evaluate(data, batch_size=batch_size)

    # ----------------- Freeze / Unfreeze ----------------- #
    def freeze_all(self) :
        try :
            for i, layer in enumerate(self.model.layers) :
                try :
                    layer.trainable = False
                except Exception as e :
                    self.logger.error(f"Error while freezing layer {i + 1} ({layer.name}) : {e}")
            self.logger.info("All layers frozen")
        except Exception as e :
            self.logger.error(e)

    def freeze_early_N(self, N : int = None) :
        """
        If you do not provide N, it takes default from config
        """
        try :
            if N is None :
                N = self.config["N"]
            for i, layer in enumerate(self.model.layers[N :]) :
                try :
                    layer.trainable = False
                except Exception as e :
                    self.logger.error(f"Error while freezing layer {i + 1} ({layer.name}) : {e}")
            self.logger.info(f"Froze all layers after first {N}")
        except Exception as e :
            self.logger.error(e)

    def freeze_upto_layer(self, layer_name : str = None) :
        try :
            for i, layer in enumerate(self.model.layers) :
                try :
                    layer.trainable = False
                    if layer.name == layer_name :
                        break
                except Exception as e :
                    self.logger.error(f"Error while freezing layer {i + 1} ({layer.name}) : {e}")
            self.logger.info(f"Froze layers up to {layer_name}")
        except Exception as e :
            self.logger.error(e)

    def unfreeze_later_N(self, N : int = None) :
        """
        If you do not provide N, it takes default from config
        """
        try :
            if N is None :
                N = self.config["N"]
            for i, layer in enumerate(self.model.layers[-N :]) :
                try :
                    layer.trainable = True
                except Exception as e :
                    self.logger.error(f"Error while unfreezing layer {i + 1} ({layer.name}) : {e}")
            self.logger.info(f"Unfroze last {N} layers")
        except Exception as e :
            self.logger.error(e)

    def unfreeze_after_layer(self, layer_name : str = None) :
        try :
            freeze = True
            for i, layer in enumerate(self.model.layers) :
                try :
                    layer.trainable = not freeze
                    if layer.name == layer_name :
                        freeze = False
                except Exception as e :
                    self.logger.error(f"Error during layer {i + 1} ({layer.name}) (freeze = {freeze}) : {e}")
            self.logger.info(f"Unfroze layers after {layer_name}")
        except Exception as e :
            self.logger.error(e)

    def unfreeze_all(self) :
        try :
            for i, layer in enumerate(self.model.layers) :
                try :
                    layer.trainable = True
                except Exception as e :
                    self.logger.error(f"Error during layer {i + 1} ({layer.name}) : {e}")
            self.logger.info("All layers unfrozen")
        except Exception as e :
            self.logger.error(e)

    # ----------------- Add Layers ----------------- #
    def _add_custom_layers_after_a_layer(self, output : tf.Tensor, layers : list[keras.layers.Layer]) -> tf.Tensor :
        """
        Takes an output tensor from an existing layer and applies new layers on top.
        """
        x = output
        for l in layers :
            x = l(x)
        return x

    def add_custom_layers(self, layers_list : list[keras.layers.Layer] = None) :
        """
        layers_list : list of keras.layers.Layer
                    If given, then adds these layer after layer_name
                    If not given, adds custom layers from config
        """
        try :
            if layers_list is None :
                layers_list = get_custom_layers(self.config["num_classes"])

            self.outputs_layer = self._add_custom_layers_after_a_layer(self.outputs_layer, layers_list)

            self.rebuild_model()
            self.logger.info("Added custom layers at the end")
        except Exception as e :
            self.logger.error(e)

    def add_layers_in_between(self, layer_name : str, layers : list[keras.layers.Layer]) :
        """
        layers : list of keras.layers.Layer
                    If given, then adds these layer after layer_name
                    If not given, adds custom layers from config
                    after adding these, adds the remaining layers from the model
        """
        try :
            target_layer = self.model.get_layer(layer_name)
            index = self.model.layers.index(target_layer)

            x = self._add_custom_layers_after_a_layer(target_layer.output, layers)

            for layer in self.model.layers[index + 1 :] :
                x = layer(x)

            self.outputs_layer = x

            self.rebuild_model()
            self.logger.info(f"Added custom layers after {layer_name}")
        except Exception as e :
            self.logger.error(e)

    # ----------------- Cut Layers ----------------- #
    def cut_at_layer(self, layer_name : str) :
        try :
            self.outputs_layer = self.model.get_layer(layer_name).output

            self.rebuild_model()
            self.logger.info(f"Cut model at {layer_name}")
        except Exception as e :
            self.logger.error(e)

    def cut_at_layer_and_add_custom_layers(self, layer_name : str, layers_list : list[keras.layers.Layer] = None) :
        """
        layers_list : list of keras.layers.Layer
                    If given, then adds these layer after layer_name
                    If not given, adds custom layers from config
        """
        try :
            if layers_list is None :
                layers_list = get_custom_layers(self.config["num_classes"])

            outputs = self.model.get_layer(layer_name).output

            self.outputs_layer = self._add_custom_layers_after_a_layer(outputs, layers_list)

            self.rebuild_model()
            self.logger.info(f"Cut model at {layer_name} and added custom layers")
        except Exception as e :
            self.logger.error(e)

    # ----------------- Remove Layers ----------------- #
    def remove_last_layer(self) :
        try :
            last_second_layer = self.model.layers[-2]
            self.outputs_layer = last_second_layer.output
            self.rebuild_model()
            self.logger.info("Removed last layer")
        except Exception as e :
            self.logger.error(e)

    def remove_N_layers(self, N : int = None) :
        """
        If you do not provide N, it takes default from config
        """
        try :
            if N is None :
                N = self.config["remove_N"]
            self.cut_at_layer(self.model.layers[-(N + 1)].name)
            self.logger.info(f"Removed last {N} layers")
        except Exception as e :
            self.logger.error(e)

    def remove_layer_in_between(self, layer_name : str) :
        try :
            target_layer = self.model.get_layer(layer_name)
            index = self.model.layers.index(target_layer)
            prev_layer = self.model.layers[index - 1]
            self.outputs_layer = self._add_custom_layers_after_a_layer(prev_layer.output, self.model.layers[index + 1 :])
            self.rebuild_model()
            self.logger.info(f"Removed {layer_name} layer and reconnected network")
        except Exception as e :
            self.logger.error(e)