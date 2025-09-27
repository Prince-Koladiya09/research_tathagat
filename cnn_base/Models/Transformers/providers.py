import keras
import tensorflow_hub as hub

_HUB_URLS = {
    "vit-base": "google/vit-base-patch16-224-in21k",
    "vit-large": "google/vit-large-patch16-224-in21k",
    "swin-transformer": "microsoft/swin-base-patch4-window7-224-in22k",
    "swin-v2": "microsoft/swinv2-base-patch4-window12-192-22k",
    "deit-small": "facebook/deit-small-distilled-patch16-224",
    "deit-base": "facebook/deit-base-distilled-patch16-224",
    "beit": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "convnext": "facebook/convnext-base-224-22k",
    "mobilevit": "apple/mobilevit-small",

    # Not implemented yet
    # "pvt" (Pyramid Vision Transformer)
    # "t2t-vit"
    # "cvt" (Convolutional Vision Transformer)
    # "vitaev2"
    # "efficientformer-l1"
    # "convmixer"
    # "poolformer"
    # "twins-svt"
    # "hrnet"
    # "bit"
    # "noisy-student"

}

def _get_hub_model(model_name: str, **kwargs):
    handle = _HUB_URLS[model_name]
    input_shape = kwargs["img_size"] + (3,)
    
    inputs = keras.Input(shape=input_shape)
    hub_layer = hub.KerasLayer(handle, trainable=kwargs.get("trainable", True))
    outputs = hub_layer(inputs)
    
    if isinstance(outputs, dict):
        output_key = "default" if "default" in outputs else list(outputs.keys())[0]
        final_output = outputs[output_key]
    else:
        final_output = outputs
        
    return inputs, final_output

def get_model(name: str, **kwargs):
    if name in _HUB_URLS:
        return _get_hub_model(name, **kwargs)
    else:
        raise ValueError(f"Transformer model '{name}' is not implemented or the name is incorrect.")