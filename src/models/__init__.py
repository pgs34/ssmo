"""Task-specific model builders with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_classification_model",
    "build_operator_model",
    "build_segmentation_model",
    "build_time_series_model",
]


def __getattr__(name: str):
    if name == "build_classification_model":
        return import_module(".classification", __name__).build_classification_model
    if name == "build_operator_model":
        return import_module(".operator", __name__).build_operator_model
    if name == "build_segmentation_model":
        return import_module(".segmentation", __name__).build_segmentation_model
    if name == "build_time_series_model":
        return import_module(".time_series", __name__).build_time_series_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
