import keras
import re
from keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, ResNet152,
    InceptionV3, Xception, DenseNet121, DenseNet169,
    DenseNet201, EfficientNetB0, EfficientNetB1, EfficientNetB7,
    MobileNetV2, NASNetMobile
)

_MODEL_DICT = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "inceptionv3": InceptionV3,
    "xception": Xception,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "efficientnetb0": EfficientNetB0,
    "efficientnetb1": EfficientNetB1,
    "efficientnetb7": EfficientNetB7,
    "mobilenetv2": MobileNetV2,
    "nasnetmobile": NASNetMobile,
}

def get_model(logger, name : str) -> keras.Model :
    try :
        pattern = re.compile("[^a-zA-Z0-9]")
        name = re.sub(pattern, "", name).lower()

        return _MODEL_DICT[name]
    except Exception as e :
        logger.error(e)