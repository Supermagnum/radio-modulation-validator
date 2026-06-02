"""Parse OOT project README files for modes and metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

KNOWN_MODE_NAMES = [
    "NBFM",
    "WBFM",
    "GMSK",
    "DMR",
    "M17",
    "YSF",
    "D-Star",
    "DSTAR",
    "P25",
    "NXDN",
    "dPMR",
    "BPSK",
    "QPSK",
    "8PSK",
    "4FSK",
    "2FSK",
    "8FSK",
    "AM",
    "SSB",
    "FreeDV",
    "SOQPSK",
    "DSSS",
    "GDSS",
    "LDPC",
    "AX.25",
    "APRS",
    "IL2P",
    "FX.25",
    "BELL202",
    "G3RUH",
    "NFM_CTCSS",
    "NFM_DCS",
    "VARA FM",
    "sleipnir_8qpsk",
    "8xQPSK",
]


@dataclass
class ReadmeSummary:
    modulation_modes: list[str] = field(default_factory=list)
    gr_version_mentioned: str | None = None
    has_iq_generation_example: bool = False
    build_instructions: str = ""
    dependencies: list[str] = field(default_factory=list)
    is_ai_generated: bool = False
    ai_notice: str = ""


def parse_readme(readme_path: Path) -> ReadmeSummary:
    """Extract modulation modes and metadata from a project README."""
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    summary = ReadmeSummary()

    upper = text.upper()
    for mode in KNOWN_MODE_NAMES:
        pattern = re.escape(mode).replace(r"\-", r"[- ]?")
        if re.search(rf"\b{pattern}\b", text, re.IGNORECASE):
            canonical = mode
            if mode.upper() == "DSTAR":
                canonical = "D-Star"
            summary.modulation_modes.append(canonical)

    if re.search(r"sleipnir|8\s*[-x]?\s*carrier|8x\s*qpsk", text, re.IGNORECASE):
        if "sleipnir_8qpsk" not in summary.modulation_modes:
            summary.modulation_modes.append("sleipnir_8qpsk")

    summary.modulation_modes = sorted(set(summary.modulation_modes), key=str.lower)

    if re.search(r"gnu\s*radio\s*4|gr\s*4|gnuradio4", text, re.IGNORECASE):
        if re.search(r"gnu\s*radio\s*3|gr\s*3|3\.10", text, re.IGNORECASE):
            summary.gr_version_mentioned = "both"
        else:
            summary.gr_version_mentioned = "4"
    elif re.search(r"gnu\s*radio\s*3|gr\s*3|3\.10", text, re.IGNORECASE):
        summary.gr_version_mentioned = "3"

    summary.has_iq_generation_example = bool(
        re.search(r"\.iq|iq_sample|file_sink|vector_sink", text, re.IGNORECASE)
    )

    build_match = re.search(
        r"(#{1,3}\s*(build|install|compilation|getting started).*?)(?=\n#{1,3}\s|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if build_match:
        summary.build_instructions = build_match.group(1).strip()[:2000]

    dep_section = re.search(
        r"(#{1,3}\s*dependenc(?:y|ies).*?)(?=\n#{1,3}\s|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if dep_section:
        for line in dep_section.group(1).splitlines():
            line = line.strip().lstrip("-*").strip()
            if line and not line.startswith("#"):
                summary.dependencies.append(line[:200])

    ai_idx = -1
    for marker in ("AI-generated", "IMPORTANT NOTICE", "AI generated"):
        idx = upper.find(marker.upper())
        if idx >= 0:
            ai_idx = idx
            break
    if ai_idx >= 0:
        window = text[max(0, ai_idx - 200) : ai_idx + 400]
        if re.search(r"\bAI\b", window, re.IGNORECASE):
            summary.is_ai_generated = True
            notice_match = re.search(
                r"(IMPORTANT NOTICE.*?)(?=\n#{1,3}|\n\n\n|\Z)",
                text[ai_idx:],
                re.IGNORECASE | re.DOTALL,
            )
            if notice_match:
                summary.ai_notice = notice_match.group(1).strip()[:1500]

    return summary
