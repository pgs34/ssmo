"""Task-level dataset and runner utilities with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ClassificationDataConfig",
    "build_classification_dataloaders",
    "OperatorDataConfig",
    "build_operator_dataloaders",
    "SegmentationDataConfig",
    "build_segmentation_dataloaders",
    "TimeSeriesDataConfig",
    "build_time_series_dataloaders",
]


def __getattr__(name: str):
    if name in {"ClassificationDataConfig", "build_classification_dataloaders"}:
        module = import_module(".classification", __name__)
        return getattr(module, name)
    if name in {"OperatorDataConfig", "build_operator_dataloaders"}:
        module = import_module(".operator", __name__)
        return getattr(module, name)
    if name in {"SegmentationDataConfig", "build_segmentation_dataloaders"}:
        module = import_module(".segmentation", __name__)
        return getattr(module, name)
    if name in {"TimeSeriesDataConfig", "build_time_series_dataloaders"}:
        module = import_module(".time_series", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
