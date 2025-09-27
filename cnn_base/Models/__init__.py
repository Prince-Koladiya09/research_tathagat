from .CNN.model import Model as CNN
from .Transformers.model import Model as Transformer
from .get_model import get_model

__all__ = ["CNN", "Transformer", "get_model"]