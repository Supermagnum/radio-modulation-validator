"""Classifier vocabulary for rmv scan (labels must match trained models)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rmv.constants import (
    CSPB_TO_FAMILY,
    FAMILY_CLASSES,
    HISARMOD_TO_FAMILY,
    ORDER_CLASSES,
    RADIOML_TO_FAMILY,
    SYNTHETIC_TO_FAMILY,
)
from rmv.scan.mode_table import ModeSpec, all_mode_specs

logger = logging.getLogger(__name__)
console = Console(stderr=True)


@dataclass(frozen=True)
class ClassifierVocab:
    family_names: tuple[str, ...]
    order_names: tuple[str, ...]
    models_dir: Path


def _meta_path_for_model(models_dir: Path, stem: str) -> Path:
    onnx = models_dir / f"{stem}.onnx"
    meta = onnx.with_suffix(".meta.json")
    if meta.is_file():
        return meta
    base = stem.replace("_classifier", "")
    for name in (f"best_{base}_meta.json", f"best_{stem}_meta.json"):
        checkpoint_meta = models_dir.parent / "checkpoints" / name
        if checkpoint_meta.is_file():
            return checkpoint_meta
    return meta


def _load_names_from_meta(meta_path: Path, default: list[str]) -> list[str]:
    if not meta_path.is_file():
        return list(default)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    names = data.get("class_names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return list(default)


def load_classifier_vocab(models_dir: Path) -> ClassifierVocab:
    """Load family/order class names from ONNX sidecar or checkpoint meta JSON."""
    family = _load_names_from_meta(
        _meta_path_for_model(models_dir, "family_classifier"),
        FAMILY_CLASSES,
    )
    order = _load_names_from_meta(
        _meta_path_for_model(models_dir, "order_classifier"),
        ORDER_CLASSES,
    )
    return ClassifierVocab(
        family_names=tuple(family),
        order_names=tuple(order),
        models_dir=models_dir,
    )


def order_to_family(order: str) -> str | None:
    """Map an order label to its training family (best-effort)."""
    for mapping in (
        SYNTHETIC_TO_FAMILY,
        RADIOML_TO_FAMILY,
        HISARMOD_TO_FAMILY,
        CSPB_TO_FAMILY,
    ):
        if order in mapping:
            return mapping[order]
    if order == "FM":
        return "FM"
    if order == "NBFM":
        return "FM"
    return None


def resolve_classifier_labels(
    spec: ModeSpec,
    vocab: ClassifierVocab,
) -> tuple[str, str] | None:
    """
    Return (expected_family, expected_order) using exact classifier strings.

    None when labels are missing from the loaded vocabulary.
    """
    if spec.generation_method == "skip" or spec.expected_order in ("--", ""):
        return None

    order = spec.expected_order
    if order not in vocab.order_names:
        return None

    family = spec.expected_family
    inferred = order_to_family(order)
    if family in ("--", "") and inferred:
        family = inferred
    if family not in vocab.family_names:
        if inferred and inferred in vocab.family_names:
            family = inferred
        else:
            return None
    return family, order


@dataclass(frozen=True)
class ModeLabelMismatch:
    mode_name: str
    expected_family: str
    expected_order: str
    reason: str


def find_mode_label_mismatches(vocab: ClassifierVocab) -> list[ModeLabelMismatch]:
    """Compare scan mode table entries to classifier vocabulary."""
    mismatches: list[ModeLabelMismatch] = []
    for spec in all_mode_specs():
        if spec.generation_method in ("skip", "plugin"):
            continue
        resolved = resolve_classifier_labels(spec, vocab)
        if resolved is not None:
            continue
        reason_parts: list[str] = []
        if spec.expected_order not in vocab.order_names:
            reason_parts.append(f"order '{spec.expected_order}' not in classifier")
        fam = spec.expected_family
        if fam not in vocab.family_names:
            inf = order_to_family(spec.expected_order)
            if inf is None or inf not in vocab.family_names:
                reason_parts.append(f"family '{fam}' not in classifier")
        mismatches.append(
            ModeLabelMismatch(
                mode_name=spec.mode_name,
                expected_family=spec.expected_family,
                expected_order=spec.expected_order,
                reason="; ".join(reason_parts) or "label mismatch",
            )
        )
    return mismatches


def print_mode_label_warnings(vocab: ClassifierVocab) -> list[ModeLabelMismatch]:
    """Print a warning table for mode table entries not in classifier vocab."""
    mismatches = find_mode_label_mismatches(vocab)
    if not mismatches:
        return mismatches

    table = Table(title="Scan mode label mismatches (modes will be skipped)")
    table.add_column("README mode")
    table.add_column("Expected family")
    table.add_column("Expected order")
    table.add_column("Reason")
    for row in mismatches:
        table.add_row(
            row.mode_name,
            row.expected_family,
            row.expected_order,
            row.reason,
        )
    console.print("[yellow]Classifier vocabulary does not cover some scan modes:[/]")
    console.print(table)
    return mismatches
