"""Shared dataclasses and type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dataclasses_json import dataclass_json

SCHEMA_VERSION = "1.0"


@dataclass
class IQDataset:
    """Standard dataset container returned by all loaders."""

    samples: np.ndarray  # (N, 2, 1024) float32
    labels: np.ndarray  # (N,) int32
    snr_db: np.ndarray  # (N,) float32
    class_names: list[str]
    source: str  # radioml2016 | hisarmod | cspb | synthetic


@dataclass
class ClassifierResult:
    """Per-chunk or aggregated classification output."""

    family: str
    family_confidence: float
    order: str
    order_confidence: float
    family_logits: np.ndarray = field(repr=False)
    order_logits: np.ndarray = field(repr=False)
    pass_threshold: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "family": self.family,
            "family_confidence": self.family_confidence,
            "order": self.order,
            "order_confidence": self.order_confidence,
            "pass_threshold": self.pass_threshold,
        }


@dataclass_json
@dataclass
class ValidationResult:
    """Result of validating one IQ file against its sidecar metadata."""

    iq_file: str
    block_name: str
    source_repo: str
    expected_family: str
    expected_order: str
    predicted_family: str
    predicted_order: str
    family_confidence: float
    order_confidence: float
    family_pass: bool
    order_pass: bool
    snr_db: float | None
    timestamp: str
    notes: str
    schema_version: str = SCHEMA_VERSION
    hard_fail: bool = False
    hard_fail_reason: str | None = None
    custom_mode: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "iq_file": self.iq_file,
            "block_name": self.block_name,
            "source_repo": self.source_repo,
            "expected_family": self.expected_family,
            "expected_order": self.expected_order,
            "predicted_family": self.predicted_family,
            "predicted_order": self.predicted_order,
            "family_confidence": self.family_confidence,
            "order_confidence": self.order_confidence,
            "family_pass": self.family_pass,
            "order_pass": self.order_pass,
            "snr_db": self.snr_db,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }
        if self.hard_fail:
            d["hard_fail"] = True
            d["hard_fail_reason"] = self.hard_fail_reason
        if self.custom_mode is not None:
            d["custom_mode"] = self.custom_mode
        return d


@dataclass
class IQSidecar:
    """Metadata sidecar for contributed IQ samples."""

    source: str
    block_name: str
    expected_family: str
    expected_order: str
    sample_rate_hz: int
    center_freq_hz: int = 0
    snr_db: float | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IQSidecar:
        required = ("source", "block_name", "expected_family", "expected_order", "sample_rate_hz")
        missing = [k for k in required if k not in data]
        if missing:
            msg = f"Sidecar missing required fields: {', '.join(missing)}"
            raise ValueError(msg)
        return cls(
            source=str(data["source"]),
            block_name=str(data["block_name"]),
            expected_family=str(data["expected_family"]),
            expected_order=str(data["expected_order"]),
            sample_rate_hz=int(data["sample_rate_hz"]),
            center_freq_hz=int(data.get("center_freq_hz", 0)),
            snr_db=data.get("snr_db"),
            notes=str(data.get("notes", "")),
        )


def sidecar_path_for_iq(iq_file: Path) -> Path:
    """Return expected JSON sidecar path for an .iq file."""
    return iq_file.with_suffix(".json")
