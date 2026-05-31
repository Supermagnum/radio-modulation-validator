"""Public Python API for radio-modulation-validator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rmv.classifier import ClassifierResult, ModulationClassifier
from rmv.types import SCHEMA_VERSION, ValidationResult
from rmv.validate import load_iq_chunks, load_sidecar, run_validate_file

logger = logging.getLogger(__name__)


class RadioModulationValidator:
    """High-level API for classifying and validating IQ samples."""

    def __init__(
        self,
        models_dir: Path = Path("models"),
        confidence_threshold: float = 0.70,
        *,
        verify_checksums: bool = True,
    ) -> None:
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        self._classifier = ModulationClassifier(
            models_dir,
            confidence_threshold=confidence_threshold,
            verify_checksums=verify_checksums,
        )

    def validate_file(self, iq_file: Path) -> ValidationResult:
        """Validate a single .iq file using its sidecar .json."""
        return run_validate_file(
            iq_file,
            self._classifier,
            threshold=self.confidence_threshold,
            output_dir=None,
        )

    def validate_directory(self, directory: Path) -> list[ValidationResult]:
        """Validate all .iq files in a directory (recursive)."""
        results: list[ValidationResult] = []
        for iq_file in sorted(directory.glob("**/*.iq")):
            try:
                load_sidecar(iq_file)
            except FileNotFoundError:
                logger.warning("Skipping %s: no sidecar", iq_file)
                continue
            results.append(self.validate_file(iq_file))
        return results

    def classify(self, samples: np.ndarray) -> list[ClassifierResult]:
        """Classify raw IQ chunks directly, no sidecar needed."""
        return self._classifier.classify(samples)

    def classify_file(self, iq_file: Path, chunk_size: int = 1024) -> ClassifierResult:
        """Classify all chunks in an IQ file and aggregate."""
        chunks = load_iq_chunks(iq_file, chunk_samples=chunk_size)
        return self._classifier.classify_aggregate(chunks)

    def summary_report(
        self,
        results: list[ValidationResult],
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Aggregate results into a summary dict, optionally write JSON."""
        total = len(results)
        passed = sum(1 for r in results if r.family_pass and r.order_pass)
        hard_fails = sum(1 for r in results if r.hard_fail)
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "hard_fails": hard_fails,
            "pass_rate": passed / total if total else 0.0,
            "results": [r.to_dict() for r in results],
        }
        if output_path is not None:
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def format_summary_markdown(results: list[ValidationResult]) -> str:
    """Format summary report as markdown table."""
    lines = [
        "| Block | Repo | Family | Order | F-conf | O-conf | Pass |",
        "|-------|------|--------|-------|--------|--------|------|",
    ]
    for r in results:
        fam = f"{r.predicted_family}{'✓' if r.family_pass else '✗'}"
        ord_ = f"{r.predicted_order}{'✓' if r.order_pass else '✗'}"
        ok = "✓" if r.family_pass and r.order_pass else "✗"
        lines.append(
            f"| {r.block_name} | {r.source_repo} | {fam} | {ord_} | "
            f"{r.family_confidence:.2f} | {r.order_confidence:.2f} | {ok} |"
        )
    return "\n".join(lines) + "\n"
