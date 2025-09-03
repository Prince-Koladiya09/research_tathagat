from .model import Model_Train
from . import callbacks
import re

__all__ = ["Model_Train", "callbacks"]

_MODEL_DICT = {}

def get_model(name : str) :
    pattern = re.compile("[^a-zA-Z0-9]")
    name = re.sub(pattern, "", name).lower()

    return _MODEL_DICT[name]