"""Custom-mode validation plugins for composite and novel modulations."""

from rmv.plugins.base import CustomModeResult, CustomModeValidator
from rmv.plugins.registry import get, list_plugins, register

__all__ = [
    "CustomModeResult",
    "CustomModeValidator",
    "get",
    "list_plugins",
    "register",
]
