# 2D Neural Operators
from .uno import UNO2D
from .fno import FNO2D
from .gnot import GNOT2D
from .deeponet import DeepONet2D

# 1D Neural Operators
from .uno import UNO1D
from .fno import FNO1D
from .gnot import GNOT1D
from .deeponet import DeepONet1D

__all__ = [
    # 2D
    "UNO2D",
    "FNO2D",
    "GNOT2D",
    "DeepONet2D",
    # 1D
    "UNO1D",
    "FNO1D",
    "GNOT1D",
    "DeepONet1D",
]