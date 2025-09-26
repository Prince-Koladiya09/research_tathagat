import keras
import tensorflow_hub as hub
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

def _get_keras_app_model(model_name: str) -> keras.Model :
    if model_name in _KERAS_MODEL_DICT :
        return None
    return _KERAS_MODEL_DICT[model_name]

def _load_hub_model(model_name : str, input_shape, trainable=False) -> keras.Model :
    if model_name in _HUB_URLS :
        return None
    handle = _HUB_URLS[model_name]
    hub_layer = hub.KerasLayer(handle, trainable=trainable)
    inputs = keras.Input(shape=input_shape)
    outputs = hub_layer(inputs)
    return keras.Model(inputs, outputs)
