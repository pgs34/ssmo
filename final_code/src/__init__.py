"""Experiment package."""

from __future__ import annotations

from collections import OrderedDict
import typing

# Compatibility with older Python versions where typing.OrderedDict may be absent.
if not hasattr(typing, "OrderedDict"):
    typing.OrderedDict = OrderedDict
