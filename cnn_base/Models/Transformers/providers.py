import keras
import tensorflow_hub as hub

_HUB_URLS = {
    "vit_base": "https://tfhub.dev/google/vit_base_patch16_224/1",
    "vit_large": "https://tfhub.dev/google/vit_large_patch16_224/1",
    "swin_transformer": "https://tfhub.dev/google/swin_tiny_patch4_window7_224/1",
    "deit_base": "https://tfhub.dev/google/deit/base_distilled_patch16_224/1",
    "beit": "https://tfhub.dev/google/beit/base/patch16/224/1",
    "mobilevit": "https://tfhub.dev/google/mobilevit/xs/1",
    "poolformer": "https://www.kaggle.com/models/sayannath/poolformer/frameworks/TensorFlow2/variations/poolformer-s12-fe/versions/1",
    "pvt": "https://www.kaggle.com/models/sayannath/pvt/frameworks/TensorFlow2/variations/pvt-tiny-fe/versions/1",
    "twins-svt": "https://www.kaggle.com/models/sayannath/twins-svt/frameworks/TensorFlow2/variations/twins-svt-s-fe/versions/1",
    "t2t-vit": "https://www.kaggle.com/models/sayannath/t2t-vit/frameworks/TensorFlow2/variations/t2t-vit-14-fe/versions/1",
    "efficientformer-l1": "https://www.kaggle.com/models/google/efficientformer/frameworks/TensorFlow2/variations/l1-fe/versions/1",
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