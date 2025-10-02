from ..loggers import Logger
from .cnn.model import Model as CNN_Model
from .transformers.model import Model as Transformer_Model
from .cnn.providers import _KERAS_MODEL_DICT as cnn_keras_models, _HUB_URLS as cnn_hub_models
from .transformers.providers import _HUB_URLS as transformer_hub_models

logger = Logger(name="Model_Factory")

def get_model(model_name: str, **kwargs):
    clean_name = model_name.lower().replace('-', '_').replace(' ', '_')

    cnn_models = list(cnn_keras_models.keys()) + list(cnn_hub_models.keys())
    transformer_models = list(transformer_hub_models.keys())

    if clean_name in cnn_models:
        logger.info(f"'{model_name}' identified as a CNN. Instantiating CNN_Model.")
        # Directly instantiate the class, passing the model name to the constructor
        return CNN_Model(base_model_name=clean_name, **kwargs)
    
    elif clean_name in transformer_models:
        logger.info(f"'{model_name}' identified as a Vision Transformer. Instantiating Transformer_Model.")
        # Directly instantiate the class
        return Transformer_Model(base_model_name=clean_name, **kwargs)
        
    else:
        logger.error(f"Model '{model_name}' not found in any model provider.")
        raise ValueError(f"Model '{model_name}' is not supported.")