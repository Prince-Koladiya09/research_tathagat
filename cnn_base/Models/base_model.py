import tensorflow as tf
from keras import Model as Keras_Model
from joblib import dump, load
from abc import ABC, abstractmethod

from ..loggers import Logger
from ..config import (CONFIG, OPTIMIZERS, LR_SCHEDULERS,
                    get_model_path, def_callbacks)
from copy import deepcopy

class Base_Model(ABC):

    def __init__(self, update_config_kwargs: dict = None, name: str = "custom_model"):
        self.base_model = None
        self.model: Keras_Model = None
        self.outputs_layer: tf.Tensor = None
        self.config = CONFIG.deepcopy()
        self.logger = Logger()
        self.callbacks = def_callbacks(self.logger)
        self.name = name
        self.logger.info("Base_Model initialized.")

    @staticmethod
    def update_global_config(updates: dict, logger = None) -> None:
        CONFIG.update(updates)
        if logger :
            logger.info("Global config updated.")
        else :
            print("Global config updated.")

    @abstractmethod
    def get_base_model(self, name: str) -> 'Base_Model':
        pass

    @abstractmethod
    def add_custom_layers(self, layers_list: list = None) -> 'Base_Model':
        pass

    def _rebuild_model(self) -> 'Base_Model':
        if self.base_model and self.outputs_layer is not None:
            try:
                self.model = Keras_Model(inputs=self.base_model.input, outputs=self.outputs_layer)
                self.logger.info("Model rebuilt successfully.")
            except Exception as e:
                self.logger.error(f"Error rebuilding model: {e}")
        else:
            self.logger.error("Cannot rebuild model: Base model or output layer is missing.")
        return self

    def compile(self) -> 'Base_Model':
        try:
            lr_config = self.config.lr_scheduler
            lr_or_schedule = self.config.learning_rate
            if lr_config.name and lr_config.name in LR_SCHEDULERS:
                scheduler_info = deepcopy(LR_SCHEDULERS[lr_config.name])
                scheduler_params = scheduler_info["params"]
                scheduler_params.update(lr_config.params)
                if "initial_learning_rate" in scheduler_params or scheduler_info['class'].__name__ == "CosineDecay":
                    scheduler_params["initial_learning_rate"] = self.config.learning_rate
                lr_or_schedule = scheduler_info["class"](**scheduler_params)
                self.logger.info(f"Using LR Scheduler: {lr_config.name}")

            opt_config = self.config.optimizer
            if opt_config.name not in OPTIMIZERS:
                raise ValueError(f"Optimizer '{opt_config.name}' not recognized.")
            
            optimizer_info = deepcopy(OPTIMIZERS[opt_config.name])
            optimizer_params = optimizer_info["params"]
            optimizer_params.update(opt_config.params)
            
            optimizer_instance = optimizer_info["class"](learning_rate=lr_or_schedule, **optimizer_params)

            if not self.model:
                self.logger.error("Cannot compile: self.model is not built yet.")
                return self
                
            self.model.compile(
                optimizer=optimizer_instance,
                loss=self.config.loss,
                metrics=self.config.metrics
            )
            self.logger.info(f"Model compiled with Optimizer: {opt_config.name}")

        except Exception as e:
            self.logger.error(f"Error compiling model: {e}")
        return self

    def fit(self, train_data, validation_data, **kwargs):
        epochs = kwargs.get('epochs', self.config.epochs)
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        callbacks = kwargs.get('callbacks', self.callbacks)

        try:
            self.logger.info(f"Starting training for {epochs} epochs...")
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            self.logger.info("Training finished successfully.")
            return history
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return None

    def save(self, file_names: tuple[str] = None):
        if not self.model:
            self.logger.error("No model to save.")
            return

        try:
            if file_names is None:
                model_name = self.base_model.name if self.base_model else self.name
                file_names = get_model_path(model_name)

            self.model.save(file_names[0])
            with open(file_names[1], 'wb') as f:
                dump(self, f)
            self.logger.info(f"Model saved as {file_names[0]} and wrapper as {file_names[1]}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    @staticmethod
    def load(file_name: str):
        try:
            with open(file_name, 'rb') as f:
                model_wrapper = load(f)
            print(f"Model wrapper loaded from {file_name}")
            return model_wrapper
        except Exception as e:
            print(f"Error loading model wrapper: {e}")

    def summary(self) -> None:
        if self.model:
            self.model.summary()
        else:
            self.logger.error("No model to summarize.")

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def evaluate(self, data, **kwargs):
        return self.model.evaluate(data, **kwargs)