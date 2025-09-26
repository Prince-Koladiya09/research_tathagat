import keras
import tensorflow_hub as hub
import re
from keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, DenseNet121, DenseNet201, InceptionV3,
    InceptionResNetV2, Xception, EfficientNetB0, EfficientNetB7, MobileNetV2,
    NASNetMobile
)

_KERAS_MODEL_DICT = {
    # VGG
    "vgg16": VGG16,
    "vgg19": VGG19,
    # ResNet
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    # DenseNet
    "densenet121": DenseNet121,
    "densenet201": DenseNet201,
    # Inception
    "inceptionv3": InceptionV3,
    "inceptionresnetv2": InceptionResNetV2,
    # Others
    "xception": Xception,
    "efficientnetb0": EfficientNetB0,
    "efficientnetb7": EfficientNetB7,
    "mobilenetv2": MobileNetV2,
    "nasnetmobile": NASNetMobile,
}

_HUB_URLS = {
    "se_resnet50": "https://www.kaggle.com/models/google/se-resnet-50/frameworks/TensorFlow2/variations/classification/versions/1",
    "resnext101": "https://www.kaggle.com/models/google/resnext/frameworks/TensorFlow2/variations/101-26x4d/versions/1",
    "regnety_800mf": "https://www.kaggle.com/models/keras/regnet/frameworks/TensorFlow2/variations/regnety008/versions/1",
    "bit_r50x1": "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1",
    "noisy_student_efficientnet_l2": "https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet1k-l-21k-ft1k/versions/2",
    "convnext_tiny": "https://www.kaggle.com/models/keras/convnext/frameworks/TensorFlow2/variations/convnext_tiny_imagenet_1k_224/versions/1",
}

def _get_keras_app_model(name: str, **kwargs) -> keras.Model :
    return _KERAS_MODEL_DICT[name](
                weights=kwargs.get("weights", "imagenet"),
                include_top=kwargs.get("include_top", False),
                input_shape=kwargs["img_size"] + (3,)
                )

def _get_hub_model(model_name : str, **kwargs) -> keras.Model :
    handle = _HUB_URLS[model_name]
    hub_layer = hub.KerasLayer(handle, trainable=kwargs.get("trainable", False))
    inputs = keras.Input(shape=kwargs["shape"])
    outputs = hub_layer(inputs)
    return keras.Model(inputs, outputs)

def get_model(logger, name : str, **kwargs) -> keras.Model :
    try :
        pattern = re.compile("[^a-zA-Z0-9]")
        name = re.sub(pattern, "", name).lower()

        if name in _KERAS_MODEL_DICT :
            return _get_keras_app_model(name, kwargs)
        elif name in _HUB_URLS :
            return _get_hub_model(name, kwargs)
        else :
            raise ValueError("Either model is not implemented or the names is not correct")

    except Exception as e :
        logger.error(e)