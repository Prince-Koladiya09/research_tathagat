import tensorflow as tf
import keras
from keras.models import Model as Keras_Model
from copy import deepcopy
from joblib import dump, load
from abc import ABC, abstractmethod

import os
from ..loggers import Logger
from ..configs.base_config import (
    Global_Config, OPTIMIZERS, LR_SCHEDULERS, get_model_path, def_callbacks
)
from ..configs.cnn_config import DEFAULT_CNN_CONFIG
import math
import numpy as np
from ..Tracking.tracker import Experiment_Tracker
from ..utils.lr_finder import find_learning_rate
from ..Data.augmentation import get_default_augmentations

class Base_Model(Keras_Model, ABC):
    def __init__(self, name: str = "base_model", config: Global_Config = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_model = None
        self.model: Keras_Model = None
        self.outputs_layer: tf.Tensor = None
        self.augmentation_layer = None
        self.config: Global_Config = deepcopy(config) if config else deepcopy(DEFAULT_CNN_CONFIG)
        self.logger: Logger = Logger(name=self.name)
        self.callbacks: list = def_callbacks(self.logger, self.name)
        self.tracker = Experiment_Tracker()
        self.logger.info(f"Model '{self.name}' initialized.")
        self.logger.debug(f"Initial config: {self.config.model_dump_json(indent=2)}")

    @staticmethod
    def update_global_config(updates: dict) -> None:
        updated_dict = DEFAULT_CNN_CONFIG.model_dump()
        updated_dict.update(updates)
        new_config = Global_Config.model_validate(updated_dict)
        Logger().info(f"Global config updated to: {new_config.model_dump_json(indent=2)}")

    @abstractmethod
    def get_base_model(self, name: str) -> 'Base_Model':
        raise NotImplementedError

    def _rebuild_model(self) -> 'Base_Model':
        if self.base_model is None or self.outputs_layer is None:
            self.logger.error("Cannot rebuild model: base_model or outputs_layer is missing.")
            return self
        try:
            self.model = Keras_Model(inputs=self.base_model.input, outputs=self.outputs_layer)
            self.logger.info("Model rebuilt successfully.")
        except Exception as e:
            self.logger.error(f"Error rebuilding model: {e}")
        return self

    def compile(self, optimizer_config: dict = None) -> 'Base_Model':
        try:
            opt_config = self.config.optimizer
            if optimizer_config:
                opt_config = opt_config.model_copy(update=optimizer_config)

            learning_rate = opt_config.learning_rate
            
            if opt_config.scheduler_name and opt_config.scheduler_name in LR_SCHEDULERS:
                scheduler_info = deepcopy(LR_SCHEDULERS[opt_config.scheduler_name])
                scheduler_params = scheduler_info["params"]
                scheduler_params.update(opt_config.scheduler_params)
                if "initial_learning_rate" in scheduler_params:
                     scheduler_params["initial_learning_rate"] = learning_rate
                learning_rate = scheduler_info["class"](**scheduler_params)
                self.logger.info(f"Using LR Scheduler: {opt_config.model_dump_json(indent=2)}")
            
            optimizer_info = deepcopy(OPTIMIZERS[opt_config.name])
            optimizer_params = optimizer_info["params"]
            optimizer_params.update(opt_config.params)
            
            if "learning_rate" not in optimizer_info["class"].__init__.__code__.co_varnames:
                optimizer_instance = optimizer_info["class"](**optimizer_params)
                keras.backend.set_value(optimizer_instance.learning_rate, learning_rate)
            else:
                optimizer_params["learning_rate"] = learning_rate
                optimizer_instance = optimizer_info["class"](**optimizer_params)
            
            self.model.compile(
                optimizer=optimizer_instance,
                loss=self.config.training.loss,
                metrics=self.config.training.metrics
            )
            self.logger.info(f"Model compiled with Optimizer: {opt_config.name} and Loss: {self.config.training.loss}")
        except Exception as e:
            self.logger.error(f"Error compiling model: {e}")
        return self

    def summary(self) -> None:
        if self.model:
            self.model.summary(print_fn=lambda x: self.logger.info(x))
        else:
            self.logger.error("No model to summarize.")

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def save(self, file_path: str = None):
        saved_path = None
        try:
            if file_path is None:
                file_path_keras, file_path_wrapper = get_model_path(self.name, [".keras", ".pkl"])
            else:
                os.makedirs(file_path, exist_ok = True)
                file_path_keras = os.path.join(file_path, self.name + ".keras")
                file_path_wrapper = os.path.join(file_path, self.name + ".pkl")

            self.model.save(file_path_keras)
            
            model_to_save = self.model
            self.model = None
            dump(self, file_path_wrapper)
            self.model = model_to_save
            saved_path = file_path_keras

            self.logger.info(f"Model saved to {file_path_keras} and wrapper to {file_path_wrapper}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
        return saved_path

    @staticmethod
    def load(wrapper_path: str) -> 'Base_Model':
        try:
            model_wrapper = load(wrapper_path)
            keras_path = wrapper_path.replace(".pkl", ".keras")
            model_wrapper.model = keras.models.load_model(keras_path, compile=False)
            model_wrapper.logger.info(f"Model wrapper loaded from {wrapper_path}")
            model_wrapper.logger.info(f"Keras model loaded from {keras_path}. Re-compile the model before training.")
            return model_wrapper
        except Exception as e:
            Logger().error(f"Error loading model from {wrapper_path}: {e}")
            return None

    def fit(self, train_data, validation_data = None, validation_split = 0.2, **kwargs):
        fit_config = self.config.training.model_dump()
        fit_config.update(kwargs)
        self.logger.info(f"Starting training with tracking for {fit_config['epochs']} epochs with batch size {fit_config['batch_size']}.")
        
        try:
            fit_params = dict(
                train_data,
                epochs=fit_config['epochs'],
                batch_size=fit_config['batch_size'],
                callbacks=kwargs.get('callbacks', self.callbacks),
                verbose=kwargs.get('verbose', 1)
            )
            if validation_data :
                fit_params["validation_data"] = validation_data
            else :
                fit_params["validation_split"] = validation_split
            
            history = self.model.fit(**fit_params)
            
            self.logger.info("Training finished successfully.")

            best_model_path = next((cb.filepath for cb in self.callbacks if isinstance(cb, keras.callbacks.ModelCheckpoint)), None)
            if best_model_path and history:
                self.tracker.log_experiment(self, history, best_model_path)
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return None
        return history

    def predict(self, data, **kwargs):
        pred_config = self.config.training.model_dump()
        pred_config.update(kwargs)
        return self.model.predict(data, batch_size=pred_config['batch_size'], verbose=kwargs.get('verbose', 1))

    def evaluate(self, data, **kwargs):
        eval_config = self.config.training.model_dump()
        eval_config.update(kwargs)
        return self.model.evaluate(data, batch_size=eval_config['batch_size'], verbose=kwargs.get('verbose', 1))
    
    def build_with_augmentations(self, augmentation_layers=None):
        if augmentation_layers is None:
            self.logger.info("Using default augmentation layers.")
            self.augmentation_layer = get_default_augmentations()
        else:
            self.augmentation_layer = augmentation_layers

        inputs = self.base_model.input
        x = self.augmentation_layer(inputs)
        x = self.base_model(x, training=True) # Pass training=True for augmentations
        outputs = self.outputs_layer(x) # Assuming outputs_layer is the head

        self.model = keras.Model(inputs, outputs)
        self.logger.info("Model rebuilt with augmentation layers.")
        return self

    def find_lr(self, train_data, min_lr=1e-7, max_lr=1e-1):
        self.logger.info(f"Starting learning rate finder from {min_lr} to {max_lr}.")
        if not isinstance(train_data, tf.data.Dataset):
            train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(self.config.training.batch_size)
        
        find_learning_rate(self.model, train_data, min_lr, max_lr)