
from .swin import *
from .videoswin import *
from .mlp import *
from .encoders_joey import *

model_dict = {
    'swin': SwinTransformer,
    'videoswin': SwinTransformer3D,
    'mlp': MLP,
    'transformerencoder': TransformerEncoder
}

__all__ = ['model_dict']