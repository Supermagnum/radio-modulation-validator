"""Validation logic for contributed IQ samples."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rmv.classifier import ModulationClassifier
from rmv.constants import HARD_FAIL_CONFIDENCE
from rmv.iq_io import load_iq_chunks_from_path
from rmv.plugins.base import CustomModeResult
from rmv.plugins.registry import get as get_custom_plugin
from rmv.types import ClassifierResult, IQSidecar, ValidationResult, sidecar_path_for_iq

logger = logging.getLogger(__name__)

_GMSK_ORDER_ALIASES: list[str] = [
    "GMSK",
    "GMSK_BT05",
    "GMSK_BT03",
    "MSK",
    "NXDN",
    "dPMR",
]

# Scan/baseband AM-SSB is ambiguous vs FM orders at 1024 samples.
_AM_SSB_ORDER_ALIASES: list[str] = [
    "AM-SSB",
    "AM-DSB",
    "WBFM",
    "NBFM_25",
    "NBFM_50",
    "NFM_CTCSS",
    "NFM_DCS",
]

# Baseband SSB: variable envelope overlaps AM, FM deviation, and low-order QAM at 1024 samples.
SSB_ACCEPTABLE_FAMILIES: frozenset[str] = frozenset({"AM", "FM", "QAM"})
SSB_AMBIGUOUS_ORDERS: frozenset[str] = frozenset(
    {"AM-SSB", "AM-DSB-SC", "USB", "LSB"},
)

ORDER_ALIASES: dict[str, list[str]] = {
    "GMSK": list(_GMSK_ORDER_ALIASES),
    "GMSK_BT05": list(_GMSK_ORDER_ALIASES),
    "GMSK_BT03": list(_GMSK_ORDER_ALIASES),
    "MSK": list(_GMSK_ORDER_ALIASES),
    "CPFSK": ["CPFSK", "GFSK"],
    "GFSK": ["GFSK", "CPFSK"],
    "AM-DSB": ["AM-DSB", "AM_AIR_833", "AM_AIR_25K"],
    "AM_AIR_25K": ["AM_AIR_25K", "AM-DSB", "AM_AIR_833"],
    "AM_AIR_833": ["AM_AIR_833", "AM-DSB", "AM_AIR_25K"],
    "AM-SSB": list(_AM_SSB_ORDER_ALIASES),
    "NXDN": ["NXDN", "dPMR"],
    "dPMR": ["dPMR", "NXDN"],
}


def family_matches(
    expected_family: str,
    expected_order: str,
    predicted_family: str,
) -> bool:
    """True when predicted family is acceptable for the expected sidecar labels."""
    order = expected_order.strip()
    pred_f = predicted_family.strip().upper()
    if order in SSB_AMBIGUOUS_ORDERS:
        return pred_f in SSB_ACCEPTABLE_FAMILIES
    exp_f = expected_family.strip().upper()
    return exp_f == pred_f


def order_matches(expected: str, predicted: str) -> bool:
    """True when predicted order is acceptable for the expected label."""
    key = expected.strip()
    allowed = ORDER_ALIASES.get(key, [key])
    return predicted.strip() in allowed


def load_sidecar(iq_file: Path) -> IQSidecar:
    """Load JSON sidecar for an IQ file."""
    sidecar = sidecar_path_for_iq(iq_file)
    if not sidecar.is_file():
        msg = f"Sidecar not found: {sidecar}"
        raise FileNotFoundError(msg)
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    return IQSidecar.from_dict(data)


def load_iq_chunks(iq_file: Path, chunk_samples: int = 1024) -> np.ndarray:
    """Load and normalise .iq or SigMF file into (N, 2, chunk_samples) chunks."""
    chunks = load_iq_chunks_from_path(iq_file, chunk_samples=chunk_samples)
    if chunks.ndim != 3 or chunks.shape[1:] != (2, chunk_samples):
        msg = f"Wrong IQ chunk shape: {chunks.shape}, expected (N, 2, {chunk_samples})"
        raise ValueError(msg)
    return chunks


def is_custom_mode_sidecar(sidecar: IQSidecar) -> bool:
    """True when validation should use a custom plugin instead of the CNN."""
    return sidecar.expected_family.strip().lower() == "custom"


def sidecar_to_metadata_dict(sidecar: IQSidecar) -> dict[str, object]:
    """Serialize sidecar for plugin metadata argument."""
    return {
        "source": sidecar.source,
        "block_name": sidecar.block_name,
        "expected_family": sidecar.expected_family,
        "expected_order": sidecar.expected_order,
        "sample_rate_hz": sidecar.sample_rate_hz,
        "center_freq_hz": sidecar.center_freq_hz,
        "snr_db": sidecar.snr_db,
        "notes": sidecar.notes,
    }


def evaluate_validation(
    sidecar: IQSidecar,
    prediction: ClassifierResult,
    *,
    threshold: float,
) -> tuple[bool, bool, bool, str | None]:
    """
    Compare prediction to expected values.

    Returns (family_pass, order_pass, hard_fail, hard_fail_reason).

    Pass/fail is based on label correctness only. The confidence threshold is
    used by the scan runner for warnings on correct-but-low-confidence results,
    not to mark a matching prediction as failed.
    """
    family_correct = family_matches(
        sidecar.expected_family,
        sidecar.expected_order,
        prediction.family,
    )
    order_correct = order_matches(sidecar.expected_order, prediction.order)
    family_pass = family_correct
    order_pass = order_correct

    hard_fail = False
    hard_fail_reason: str | None = None

    if not family_correct:
        hard_fail = True
        hard_fail_reason = (
            f"Wrong family: expected {sidecar.expected_family}, got {prediction.family}"
        )
    elif prediction.family_confidence < HARD_FAIL_CONFIDENCE:
        hard_fail = True
        hard_fail_reason = (
            f"Family confidence {prediction.family_confidence:.2f} "
            f"below hard-fail threshold {HARD_FAIL_CONFIDENCE}"
        )

    return family_pass, order_pass, hard_fail, hard_fail_reason


def build_validation_result_from_custom(
    iq_file: Path,
    sidecar: IQSidecar,
    custom: CustomModeResult,
    *,
    threshold: float,
) -> ValidationResult:
    """Build ValidationResult from a custom-mode plugin outcome."""
    order_pass = custom.pass_overall and custom.confidence >= threshold
    family_pass = order_pass
    hard_fail = False
    hard_fail_reason: str | None = None
    if not custom.pass_overall and custom.confidence < HARD_FAIL_CONFIDENCE:
        hard_fail = True
        hard_fail_reason = (
            f"Custom mode {custom.mode_id} failed with confidence "
            f"{custom.confidence:.2f} below {HARD_FAIL_CONFIDENCE}"
        )
    elif custom.mode_id != sidecar.expected_order:
        hard_fail = True
        hard_fail_reason = (
            f"Plugin mode_id {custom.mode_id} != expected_order {sidecar.expected_order}"
        )

    return ValidationResult(
        iq_file=str(iq_file),
        block_name=sidecar.block_name,
        source_repo=sidecar.source,
        expected_family=sidecar.expected_family,
        expected_order=sidecar.expected_order,
        predicted_family="custom",
        predicted_order=custom.mode_id,
        family_confidence=custom.confidence,
        order_confidence=custom.confidence,
        family_pass=family_pass,
        order_pass=order_pass,
        snr_db=sidecar.snr_db,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=sidecar.notes if not custom.notes else f"{sidecar.notes}; {custom.notes}".strip("; "),
        hard_fail=hard_fail,
        hard_fail_reason=hard_fail_reason,
        custom_mode=custom.to_dict(),
    )


def build_validation_result(
    iq_file: Path,
    sidecar: IQSidecar,
    prediction: ClassifierResult,
    *,
    threshold: float,
) -> ValidationResult:
    """Build ValidationResult from sidecar and aggregated prediction."""
    family_pass, order_pass, hard_fail, hard_fail_reason = evaluate_validation(
        sidecar, prediction, threshold=threshold
    )
    return ValidationResult(
        iq_file=str(iq_file),
        block_name=sidecar.block_name,
        source_repo=sidecar.source,
        expected_family=sidecar.expected_family,
        expected_order=sidecar.expected_order,
        predicted_family=prediction.family,
        predicted_order=prediction.order,
        family_confidence=prediction.family_confidence,
        order_confidence=prediction.order_confidence,
        family_pass=family_pass,
        order_pass=order_pass,
        snr_db=sidecar.snr_db,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=sidecar.notes,
        hard_fail=hard_fail,
        hard_fail_reason=hard_fail_reason,
    )


def write_validation_result(
    result: ValidationResult,
    output_dir: Path,
) -> Path:
    """Write ValidationResult JSON under validation_results/<source>/."""
    out_dir = output_dir / result.source_repo
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_block = result.block_name.replace("/", "_")
    ts = result.timestamp.replace(":", "").replace("-", "")
    out_path = out_dir / f"{safe_block}_{ts}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return out_path


def validation_exit_code(result: ValidationResult) -> int:
    """
    Exit code for CI: 0 pass, 1 fail, 2 hard fail.
    """
    if result.hard_fail:
        return 2
    if result.family_pass and result.order_pass:
        return 0
    return 1


def run_validate_file(
    iq_file: Path,
    classifier: ModulationClassifier,
    *,
    threshold: float = 0.70,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> ValidationResult:
    """Validate single IQ file end-to-end (CNN or custom plugin)."""
    sidecar = load_sidecar(iq_file)
    chunks = load_iq_chunks(iq_file)

    if is_custom_mode_sidecar(sidecar):
        plugin = get_custom_plugin(sidecar.expected_order)
        if plugin is None:
            msg = (
                f"No custom plugin registered for expected_order="
                f"'{sidecar.expected_order}'"
            )
            raise ValueError(msg)
        custom = plugin.validate(
            chunks,
            float(sidecar.sample_rate_hz),
            sidecar_to_metadata_dict(sidecar),
        )
        if verbose:
            scores = custom.metrics.get("confidence_scores", {})
            logger.info(
                "Custom mode %s: pass=%s confidence=%.2f",
                custom.mode_id,
                custom.pass_overall,
                custom.confidence,
            )
            if scores:
                logger.info("  confidence_scores: %s", scores)
            logger.info("  metrics: %s", custom.metrics)
        result = build_validation_result_from_custom(
            iq_file, sidecar, custom, threshold=threshold
        )
    else:
        if verbose:
            for i, r in enumerate(classifier.classify(chunks)):
                logger.info(
                    "Chunk %d: family=%s (%.2f) order=%s (%.2f)",
                    i,
                    r.family,
                    r.family_confidence,
                    r.order,
                    r.order_confidence,
                )
        prediction = classifier.classify_aggregate(chunks)
        result = build_validation_result(iq_file, sidecar, prediction, threshold=threshold)

    if output_dir is not None:
        write_validation_result(result, output_dir)
    return result


def run_validate_cli(
    target: Path,
    classifier: ModulationClassifier,
    *,
    threshold: float,
    output: Path | None,
    output_dir: Path,
    verbose: bool,
    repo_filter: str | None,
) -> int:
    """Validate file or directory; print JSON to stdout; return exit code."""
    worst_code = 0
    results: list[ValidationResult] = []

    if target.is_file():
        targets = [target]
    elif target.is_dir():
        targets = sorted(target.glob("**/*.iq"))
    else:
        logger.error("Path not found: %s", target)
        return 1

    for iq_file in targets:
        try:
            sidecar = load_sidecar(iq_file)
        except FileNotFoundError:
            logger.warning("Skipping %s: no sidecar", iq_file)
            continue
        if repo_filter and sidecar.source != repo_filter:
            continue
        result = run_validate_file(
            iq_file,
            classifier,
            threshold=threshold,
            output_dir=output_dir,
            verbose=verbose,
        )
        results.append(result)
        code = validation_exit_code(result)
        worst_code = max(worst_code, code)

    for result in results:
        print(json.dumps(result.to_dict()), file=sys.stdout)

    if output is not None and results:
        output.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")

    return worst_code if results else 1
