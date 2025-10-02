import keras
from keras.models import Model as Keras_Model
from copy import deepcopy
from abc import ABC

from ..loggers import Logger
from ..configs.base_config import Global_Config, OPTIMIZERS, LR_SCHEDULERS, get_model_path, def_callbacks

@keras.saving.register_keras_serializable()
class Base_Model(Keras_Model, ABC):
    # We remove the complex __init__ from here. The child classes will handle it.
    # The child classes (CNN_Model, Transformer_Model) will call the super Keras_Model init directly.

    def setup_custom_attributes(self, config: Global_Config, base_model_name : str):
        """A helper to initialize our custom, non-Keras attributes."""
        self.base_model_name = base_model_name
        self.config = config
        self.logger = Logger(name=self.name)
        self.callbacks = def_callbacks(self.logger, self.name)
        self.logger.info(f"Model '{self.name}' initialized with config.")
        self.logger.debug(f"Initial config: {self.config.model_dump_json(indent=2)}")

    def get_config(self):
        base_config = super().get_config()
        config = {"config": self.config.model_dump()}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        pydantic_config_dict = config.pop("config")
        pydantic_config = Global_Config.model_validate(pydantic_config_dict)
        
        # This is how Keras rebuilds the model from its config
        model = super(Base_Model, cls).from_config(config, custom_objects)
        
        # After Keras rebuilds it, we re-attach our custom attributes
        model.setup_custom_attributes(pydantic_config)
        return model

    def compile(self, **kwargs) -> 'Base_Model':
        # This logic remains mostly the same, just simplified
        opt_config = self.config.optimizer
        learning_rate = opt_config.learning_rate
        
        if opt_config.scheduler_name:
             scheduler_info = deepcopy(LR_SCHEDULERS[opt_config.scheduler_name])
             scheduler_params = scheduler_info["params"]
             learning_rate = scheduler_info["class"](initial_learning_rate=learning_rate, **scheduler_params)

        optimizer_info = deepcopy(OPTIMIZERS[opt_config.name])
        optimizer_params = optimizer_info["params"]
        optimizer_params.update(opt_config.params)
        optimizer_params["learning_rate"] = learning_rate
        optimizer_instance = optimizer_info["class"](**optimizer_params)
        
        super().compile(optimizer=optimizer_instance, loss=self.config.training.loss, metrics=self.config.training.metrics, **kwargs)
        self.logger.info(f"Model compiled with Optimizer: {opt_config.name}")
        return self

    def save_model(self, file_path: str = None):
        if file_path is None:
            file_path = get_model_path(self.name, "keras")
        try:
            self.save(file_path)
            self.logger.info(f"Model and custom config saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    @staticmethod
    def load_model(file_path: str):
        try:
            # We provide the custom_objects dictionary so Keras knows how to find our classes
            custom_objects = {
                "CNN_Model": "cnn_base.Models.CNN",
                "Transformer_Model": "cnn_base.Models.Transformers",
            }
            with keras.saving.custom_object_scope(custom_objects):
                model_wrapper = keras.models.load_model(file_path)
            
            model_wrapper.logger.info(f"Model wrapper loaded from {file_path}.")
            return model_wrapper
        except Exception as e:
            Logger().error(f"Error loading model from {file_path}: {e}")
            return None