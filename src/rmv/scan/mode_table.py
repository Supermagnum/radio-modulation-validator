"""Known README mode names mapped to classifier labels and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GenerationMethod = Literal[
    "gr3_builtin",
    "gr4_builtin",
    "numpy",
    "synthetic",
    "plugin",
    "skip",
]


@dataclass(frozen=True)
class ModeSpec:
    mode_name: str
    expected_family: str
    expected_order: str
    generation_method: GenerationMethod
    note: str = ""
    protocol_only: bool = False


MODE_TABLE: dict[str, ModeSpec] = {
    "NBFM": ModeSpec("NBFM", "FM", "NBFM_25", "gr3_builtin"),
    "WBFM": ModeSpec("WBFM", "FM", "WBFM", "gr3_builtin"),
    "AM": ModeSpec("AM", "AM", "AM-DSB", "gr3_builtin"),
    "SSB": ModeSpec("SSB", "AM", "AM-SSB", "numpy"),
    "BPSK": ModeSpec("BPSK", "PSK", "BPSK", "gr3_builtin"),
    "QPSK": ModeSpec("QPSK", "PSK", "QPSK", "gr3_builtin"),
    "8PSK": ModeSpec("8PSK", "PSK", "8PSK", "gr3_builtin"),
    "GMSK": ModeSpec("GMSK", "FSK", "GMSK", "gr3_builtin"),
    "2FSK": ModeSpec(
        "2FSK",
        "FSK",
        "CPFSK",
        "numpy",
        note="Generic M-FSK reference; order label CPFSK matches RadioML training.",
    ),
    "4FSK": ModeSpec(
        "4FSK",
        "FSK",
        "CPFSK",
        "numpy",
        note="Generic M-FSK reference; order label CPFSK matches RadioML training.",
    ),
    "8FSK": ModeSpec(
        "8FSK",
        "FSK",
        "CPFSK",
        "numpy",
        note="Generic M-FSK reference; order label CPFSK matches RadioML training.",
    ),
    "DMR": ModeSpec(
        "DMR",
        "FSK",
        "DMR",
        "synthetic",
        protocol_only=True,
        note="Protocol-accurate 4FSK (ETSI TS 102 361-1); framing not verified.",
    ),
    "M17": ModeSpec(
        "M17",
        "FSK",
        "M17",
        "synthetic",
        protocol_only=True,
        note="Protocol-accurate 4FSK (M17 spec v1.0); framing not verified.",
    ),
    "YSF": ModeSpec(
        "YSF",
        "FSK",
        "YSF",
        "synthetic",
        protocol_only=True,
        note="Protocol-accurate C4FM; framing not verified.",
    ),
    "D-Star": ModeSpec("D-Star", "FSK", "GMSK", "gr3_builtin", protocol_only=True, note="GMSK layer only."),
    "DSTAR": ModeSpec("D-Star", "FSK", "GMSK", "gr3_builtin", protocol_only=True),
    "NXDN": ModeSpec(
        "NXDN",
        "FSK",
        "NXDN",
        "synthetic",
        protocol_only=True,
        note="Protocol-accurate 4FSK (2400 baud); framing not verified.",
    ),
    "dPMR": ModeSpec(
        "dPMR",
        "FSK",
        "dPMR",
        "synthetic",
        protocol_only=True,
        note="Protocol-accurate 4FSK (ETSI TS 102 490); framing not verified.",
    ),
    "P25": ModeSpec(
        "P25",
        "FSK",
        "CPFSK",
        "numpy",
        protocol_only=True,
        note="C4FM-style reference validated as CPFSK order (RadioML label).",
    ),
    "SOQPSK": ModeSpec("SOQPSK", "PSK", "QPSK", "numpy", note="Offset QPSK approximation."),
    "FreeDV": ModeSpec(
        "FreeDV",
        "FSK",
        "GMSK",
        "numpy",
        note="Approximation only; not a full FreeDV stack.",
    ),
    "IL2P": ModeSpec("IL2P", "--", "--", "skip", note="Protocol framing — no built-in equivalent."),
    "FX.25": ModeSpec("FX.25", "--", "--", "skip", note="Protocol framing — no built-in equivalent."),
    "AX.25": ModeSpec("AX.25", "--", "--", "skip", note="Protocol framing — no built-in equivalent."),
    "LDPC": ModeSpec("LDPC", "--", "--", "skip", note="FEC codec — not a modulator order."),
    "DSSS": ModeSpec(
        "DSSS",
        "--",
        "--",
        "skip",
        note="Spread spectrum; designed to appear as noise — classifier output not meaningful.",
    ),
    "GDSS": ModeSpec(
        "GDSS",
        "--",
        "--",
        "skip",
        note="Spread spectrum (GDSS); designed to appear as noise — classifier not meaningful.",
    ),
    "sleipnir_8qpsk": ModeSpec(
        "sleipnir_8qpsk",
        "custom",
        "sleipnir_8qpsk",
        "plugin",
        note="Validated with sleipnir_8qpsk custom plugin.",
    ),
    "8xQPSK": ModeSpec("sleipnir_8qpsk", "custom", "sleipnir_8qpsk", "plugin"),
}


def lookup_mode(mode_name: str) -> ModeSpec | None:
    if mode_name in MODE_TABLE:
        return MODE_TABLE[mode_name]
    key = mode_name.upper()
    for name, spec in MODE_TABLE.items():
        if name.upper() == key:
            return spec
    return None


def all_mode_specs() -> list[ModeSpec]:
    seen: set[str] = set()
    out: list[ModeSpec] = []
    for spec in MODE_TABLE.values():
        if spec.mode_name not in seen:
            seen.add(spec.mode_name)
            out.append(spec)
    return out
