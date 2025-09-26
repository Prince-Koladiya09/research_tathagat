from .CNN.model import Model as CNN_Model
from .Transformers.model import Model as Transformer_Model

def get_model(model_name: str, update_config_kwargs: dict = None):
    
    cnn_models = list(CNN_Model._MODEL_DICT.keys())
    transformer_models = list(Transformer_Model._MODEL_DICT.keys())

    cleaned_name = model_name.lower().replace("_", "")

    if cleaned_name in cnn_models:
        return CNN_Model(name=model_name, update_config_kwargs=update_config_kwargs)
    elif cleaned_name in transformer_models:
        return Transformer_Model(name=model_name, update_config_kwargs=update_config_kwargs)
    else:
        print(f"Warning: Model '{model_name}' not in predefined lists. Attempting to load as a Hugging Face Transformer.")
        print("For predefined CNNs, please use names like: 'resnet50', 'vgg16', 'efficientnetb0', etc.")
        print("For predefined Transformers, use: 'vit-base', 'swin-transformer', 'beit', etc.")
        return None