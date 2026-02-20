from __future__ import annotations

from . import dml, independent, naive, studygroup
from .common import mask_activation_ratio, weighted_mean

METHODS = {
    "independent": independent.directional_weights,
    "naive": naive.directional_weights,
    "dml": dml.directional_weights,
    "studygroup": studygroup.directional_weights,
}


def get_directional_weight_builder(method_name: str):
    key = method_name.lower()
    if key not in METHODS:
        raise ValueError(f"Unsupported method '{method_name}'. Use one of: {', '.join(METHODS.keys())}")
    return METHODS[key]


__all__ = [
    "get_directional_weight_builder",
    "weighted_mean",
    "mask_activation_ratio",
]
