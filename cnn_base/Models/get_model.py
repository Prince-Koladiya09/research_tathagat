from cnn_base.loggers import Logger
from cnn_base.Models.CNN.model import Model as CNN_Model
from cnn_base.Models.Transformers.model import Model as Transformer_Model
from cnn_base.Models.CNN.providers import _KERAS_MODEL_DICT as cnn_keras_models, _HUB_URLS as cnn_hub_models
from cnn_base.Models.Transformers.providers import _HUB_URLS as transformer_hub_models
# --- New PyTorch Imports ---
from cnn_base.Models.PyTorch.model import PyTorch_Model
import timm

logger = Logger("Get_Model_Logger")

def get_all_models() -> tuple[list[str], list[str], list[str]]:
    """Returns a tuple of (cnn_models, transformer_models, pytorch_models)"""
    cnn_tf = list(cnn_keras_models.keys())
    # transformer_tf = list(transformer_hub_models.keys())
    # --- Get available timm models ---
    pytorch_models = timm.list_models(pretrained=True)
    return cnn_tf, pytorch_models
    # return cnn_tf, transformer_tf, pytorch_models

def get_model(model_name: str, **kwargs):
    clean_name = model_name.lower().replace('-', '_').replace(' ', '_')

    cnn_tf_models, pytorch_models = get_all_models()
    # cnn_tf_models, transformer_tf_models, pytorch_models = get_all_models()

    # --- UPDATED LOGIC ---
    if clean_name in cnn_tf_models:
        logger.info(f"'{model_name}' identified as a Keras CNN. Instantiating CNN_Model.")
        model_instance = CNN_Model(name=clean_name, **kwargs)
        model_instance.get_base_model(clean_name)
        return model_instance
        
    elif model_name in pytorch_models:
        logger.info(f"'{model_name}' not found in Keras providers. Found in PyTorch/timm. Instantiating PyTorch_Model.")
        # We pass the original model name to timm, not the cleaned one
        return PyTorch_Model(name=model_name, **kwargs)
    
    # elif clean_name in transformer_tf_models:
    #     logger.info(f"'{model_name}' identified as a Keras Transformer. Instantiating Transformer_Model.")
    #     model_instance = Transformer_Model(name=clean_name, **kwargs)
    #     model_instance.get_base_model(clean_name)
    #     return model_instance
        
    else:
        logger.error(f"Model '{model_name}' not found in any available provider (Keras, or PyTorch/timm).")
        # logger.error(f"Model '{model_name}' not found in any available provider (Keras, TF Hub, or PyTorch/timm).")
        raise ValueError(f"Model '{model_name}' is not supported.")