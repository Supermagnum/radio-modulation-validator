"""Pre-export verification for trained family classifier checkpoints."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rmv.constants import RADIOML_SKIP_FOR_FAMILY
from rmv.dataset.preprocess import normalise_unit_power, upsample_iq_128_to_1024
from rmv.dataset.synthetic import generate_variant_chunks
from rmv.export import _load_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FamilyVerifyCase:
    """One family-classifier check."""

    label: str
    expected_family: str
    source: str  # "radioml" | "synthetic"


@dataclass
class FamilyVerifyResult:
    case: FamilyVerifyCase
    predicted: str
    confidence: float
    passed: bool
    note: str = ""


# RadioML 128-sample windows: valid for QAM/AM/FSK/PAM only (not WBFM/BPSK/QPSK).
RADIOML_VERIFY: tuple[tuple[str, int, str], ...] = (
    ("QAM16", 10, "QAM"),
    ("AM-DSB", 10, "AM"),
    ("CPFSK", 10, "FSK"),
    ("GFSK", 10, "FSK"),
    ("PAM4", 10, "PAM"),
)

SYNTHETIC_VERIFY: tuple[tuple[str, str], ...] = (
    ("WBFM", "FM"),
    ("BPSK", "PSK"),
    ("QPSK", "PSK"),
    ("NBFM_25", "FM"),
)


def _load_radioml_pickle(path: Path) -> dict[tuple[str, str], np.ndarray]:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def _predict_family(
    model: torch.nn.Module,
    chunks: np.ndarray,
    class_names: list[str],
) -> tuple[str, float]:
    model.eval()
    x = torch.from_numpy(np.asarray(chunks, dtype=np.float32))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        idx = int(probs[0].argmax().item())
        conf = float(probs[0].max().item())
    return class_names[idx], conf


def verify_family_checkpoint(
    checkpoint_dir: Path,
    *,
    radioml_pkl: Path | None = None,
    confidence_threshold: float = 0.60,
    n_chunks: int = 4,
    use_gnuradio: bool = True,
) -> list[FamilyVerifyResult]:
    """
    Verify family checkpoint before ONNX export.

    Uses RadioML only for modulations still in family training. WBFM/BPSK/QPSK
    are checked via synthetic IQ (same source used during training).
    """
    ckpt = checkpoint_dir / "best_family_classifier.pt"
    meta_path = checkpoint_dir / "best_family_meta.json"
    if not ckpt.is_file():
        msg = f"Family checkpoint not found: {ckpt}"
        raise FileNotFoundError(msg)

    model, ckpt_names, _ = _load_checkpoint(ckpt)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        class_names = [str(n) for n in meta["class_names"]]
    else:
        class_names = list(ckpt_names)

    results: list[FamilyVerifyResult] = []

    if radioml_pkl is not None and radioml_pkl.is_file():
        data = _load_radioml_pickle(radioml_pkl)
        for mod, snr, expected in RADIOML_VERIFY:
            case = FamilyVerifyCase(
                label=f"radioml:{mod}@{snr}dB",
                expected_family=expected,
                source="radioml",
            )
            key = (mod, snr)
            if key not in data:
                results.append(
                    FamilyVerifyResult(
                        case,
                        "",
                        0.0,
                        False,
                        note=f"missing pickle key {key}",
                    )
                )
                continue
            chunks = normalise_unit_power(upsample_iq_128_to_1024(data[key][:n_chunks]))
            pred, conf = _predict_family(model, chunks, class_names)
            passed = pred == expected and conf >= confidence_threshold
            results.append(FamilyVerifyResult(case, pred, conf, passed))

        for mod in sorted(RADIOML_SKIP_FOR_FAMILY):
            case = FamilyVerifyCase(
                label=f"radioml:{mod} (excluded)",
                expected_family="n/a",
                source="radioml_skip",
            )
            results.append(
                FamilyVerifyResult(
                    case,
                    "skipped",
                    0.0,
                    True,
                    note="not tested on RadioML; use synthetic row below",
                )
            )
    else:
        logger.warning("No RadioML pickle; skipping RadioML verification cases")

    for class_name, expected in SYNTHETIC_VERIFY:
        case = FamilyVerifyCase(
            label=f"synthetic:{class_name}",
            expected_family=expected,
            source="synthetic",
        )
        try:
            chunks = generate_variant_chunks(
                class_name,
                n_chunks,
                snr_db=10.0,
                use_gnuradio=use_gnuradio,
                apply_channel=True,
            )
        except ImportError as exc:
            chunks = generate_variant_chunks(
                class_name,
                n_chunks,
                snr_db=10.0,
                use_gnuradio=False,
                apply_channel=True,
            )
            note = f"numpy fallback ({exc})"
        else:
            note = ""
        pred, conf = _predict_family(model, chunks, class_names)
        passed = pred == expected and conf >= confidence_threshold
        results.append(FamilyVerifyResult(case, pred, conf, passed, note=note))

    return results


def print_verify_report(
    results: list[FamilyVerifyResult],
    *,
    console: object | None = None,
) -> bool:
    """Print table; return True if all checks passed."""
    from rich.console import Console

    out = console if console is not None else Console(stderr=True)
    all_pass = True
    for row in results:
        symbol = "OK" if row.passed else "FAIL"
        style = "green" if row.passed else "red"
        conf_str = f"{row.confidence:.2f}" if row.confidence else "-"
        pred_str = row.predicted or "-"
        out.print(
            f"  [{style}]{symbol}[/{style}] {row.case.label:28} "
            f"-> {pred_str:6} ({conf_str}) expected {row.case.expected_family}"
            + (f"  [{row.note}]" if row.note else "")
        )
        if not row.passed:
            all_pass = False

    if all_pass:
        out.print("[bold green]All checks passed — safe to export ONNX[/]")
    else:
        out.print("[bold red]FAILURES — do not export until fixed[/]")
        out.print(
            "[yellow]Note: Testing RadioML WBFM/BPSK on upsampled 128-sample "
            "windows will fail by design. Use [bold]rmv verify-family[/] instead.[/]"
        )
    return all_pass
