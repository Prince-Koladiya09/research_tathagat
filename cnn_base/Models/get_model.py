from ..loggers import Logger
from .CNN import Model as CNN_Model
from .Transformers import Model as Transformer_Model
from .CNN.providers import _KERAS_MODEL_DICT as cnn_keras_models, _HUB_URLS as cnn_hub_models
from .Transformers.providers import _HUB_URLS as transformer_hub_models

logger = Logger("Model_Factory", "get_model_info.log", "get_model_error.log")

def get_model(model_name: str, **kwargs):
    clean_name = model_name.lower().replace('-', '_').replace(' ', '_')

    cnn_models = list(cnn_keras_models.keys()) + list(cnn_hub_models.keys())
    transformer_models = list(transformer_hub_models.keys())

    if clean_name in cnn_models:
        logger.info(f"'{model_name}' identified as a CNN. Instantiating CNN_Model.")
        model_instance = CNN_Model(name=clean_name, **kwargs)
        model_instance.get_base_model(clean_name)
        return model_instance
    
    elif clean_name in transformer_models:
        logger.info(f"'{model_name}' identified as a Vision Transformer. Instantiating Transformer_Model.")
        model_instance = Transformer_Model(name=clean_name, **kwargs)
        model_instance.get_base_model(clean_name)
        return model_instance
        
    else:
        logger.error(f"Model '{model_name}' not found in any model provider.")
        raise ValueError(f"Model '{model_name}' is not supported.")