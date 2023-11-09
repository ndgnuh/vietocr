import warnings
from traceback import print_exc

try:
    from .models_torch import (OCRModel, get_available_backbones,
                              get_available_heads)
except ImportError:
    msg = "Can't import required modules for torch models, some functions won't be available."
    warnings.warn(msg, UserWarning)
    print_exc()
    del msg



del warnings, print_exc
