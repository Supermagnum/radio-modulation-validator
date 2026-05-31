"""Base types and interface for custom-mode IQ validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


@dataclass
class CustomModeResult:
    """Result of a plugin-based custom mode validation."""

    mode_id: str
    pass_overall: bool
    confidence: float
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CustomModeValidator(ABC):
    """Validate composite or novel modulations not handled by the family/order CNN."""

    mode_id: str
    description: str

    @abstractmethod
    def validate(
        self,
        samples: np.ndarray,
        sample_rate_hz: float,
        metadata: dict[str, Any],
    ) -> CustomModeResult:
        """
        Validate IQ chunks for this custom mode.

        Parameters
        ----------
        samples:
            Shape (N, 2, 1024) float32 interleaved I/Q chunks.
        sample_rate_hz:
            Sample rate from sidecar metadata.
        metadata:
            Full sidecar JSON as a dict.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return what this validator measures, for documentation and CLI."""
