import keras
import tensorflow_hub as hub
from keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, DenseNet121, DenseNet201, InceptionV3,
    InceptionResNetV2, Xception, EfficientNetB0, EfficientNetB7, MobileNetV2,
    NASNetMobile
)

_KERAS_MODEL_DICT = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "densenet121": DenseNet121,
    "densenet201": DenseNet201,
    "inceptionv3": InceptionV3,
    "inceptionresnetv2": InceptionResNetV2,
    "xception": Xception,
    "efficientnetb0": EfficientNetB0,
    "efficientnetb7": EfficientNetB7,
    "mobilenetv2": MobileNetV2,
    "nasnetmobile": NASNetMobile,
}

_HUB_URLS = {
    "se_resnet50": "https://tfhub.dev/google/imagenet/se_resnet_50/feature_vector/1",
    "resnext101": "https://tfhub.dev/google/imagenet/resnext_101/feature_vector/4",
    "regnety_800mf": "https://tfhub.dev/google/regnety_008/imagenet/feature_vector/1",
    "bit_r50x1": "https://tfhub.dev/google/bit/m-r50x1/1",
    "noisy_student_efficientnet_l2": "https://tfhub.dev/google/efficientnet/noisy-student/l2/feature-vector/1",
    "convnext_tiny": "https://tfhub.dev/google/convnext_tiny_1k_224/feature_vector/2",
}

def _get_keras_app_model(name: str, **kwargs):
    model_fn = _KERAS_MODEL_DICT[name]
    input_shape = kwargs["img_size"] + (3,)
    
    base_model = model_fn(
        weights=kwargs.get("weights", "imagenet"),
        include_top=kwargs.get("include_top", False),
        input_shape=input_shape
    )
    return base_model.input, base_model.output

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
    if name in _KERAS_MODEL_DICT:
        return _get_keras_app_model(name, **kwargs)
    elif name in _HUB_URLS:
        return _get_hub_model(name, **kwargs)
    else:
        raise ValueError(f"CNN model '{name}' is not implemented or the name is incorrect.")