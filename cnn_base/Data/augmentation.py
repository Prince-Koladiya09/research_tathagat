import keras
from keras import layers

def get_default_augmentations():
    """
    Returns a keras.Sequential model with a default set of data augmentations.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="default_augmentation_pipeline")