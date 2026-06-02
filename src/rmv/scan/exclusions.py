"""Built-in exclusions for rmv scan (non-validatable projects and modes)."""

from __future__ import annotations

from rmv.scan.mode_table import ModeSpec

# GNU Radio framework source trees (not OOT modules); never descend into these.
FRAMEWORK_DIR_NAMES = frozenset({"gnuradio", "gnuradio4", "gnuradio3"})

# Default OOT modulator projects to scan when --filter and --all are not set.
DEFAULT_INCLUDE_PROJECTS: frozenset[str] = frozenset({
    "gr-qradiolink",
    "gr-packet-protocols",
    "gr-sleipnir",
})

# Spread spectrum: designed to look like noise; CNN family/order labels are not meaningful.
NON_VALIDATABLE_MODE_NAMES = frozenset({"DSSS", "GDSS"})

# Framework trees (also block descent in discover.py).
DEFAULT_EXCLUDE_PROJECTS: frozenset[str] = (
    FRAMEWORK_DIR_NAMES
    | {
        # GNU Radio framework source (not OOT)
        # Cryptography: no RF modulation to validate
        "gr-linux-crypto",
        "gr-linux-crypto-backup",
        "gr-nacl",
        "gr-openssl",
        # Audio codecs: no RF modulation produced
        "gr-opus",
        # Spread spectrum reference / test (GDSS): not a modulator OOT
        "GR-K-GDSS",
        # Non-modulator tooling
        "app_rpt",
        # Utility / non-modulator OOT (not in default include list)
        "gr-ident",
        "gr-rake",
    }
)

EXCLUSION_REASONS: dict[str, str] = {
    "gnuradio": "Framework source tree, not an OOT module",
    "gnuradio4": "Framework source tree, not an OOT module",
    "gnuradio3": "Framework source tree, not an OOT module",
    "gr-linux-crypto": "Cryptography module; no modulation to validate",
    "gr-linux-crypto-backup": "Cryptography module; no modulation to validate",
    "gr-nacl": "Cryptography module; no modulation to validate",
    "gr-openssl": "Cryptography module; no modulation to validate",
    "gr-opus": "Audio codec; no RF modulation produced",
    "GR-K-GDSS": "Spread spectrum (GDSS); designed to appear as noise — classifier not applicable",
    "app_rpt": "Repeater/application tooling; not a modulator OOT",
    "gr-ident": "Identification utility; not a modulator OOT",
    "gr-rake": "Rake receiver / utility; not a modulator OOT",
}

MODE_EXCLUSION_REASONS: dict[str, str] = {
    "DSSS": "Spread spectrum; designed to appear as noise — classifier output not meaningful",
    "GDSS": "Spread spectrum (GDSS); designed to appear as noise — classifier output not meaningful",
    "LDPC": "FEC codec only; not a modulator order",
}


def merged_exclude_projects(config_excludes: tuple[str, ...] | frozenset[str]) -> frozenset[str]:
    """Built-in exclusions plus optional .rmv_config.toml entries."""
    extra = frozenset(config_excludes) if config_excludes else frozenset()
    return DEFAULT_EXCLUDE_PROJECTS | extra


def resolve_include_names(
    *,
    cli_filter: frozenset[str] | None,
    config_includes: tuple[str, ...] | frozenset[str],
    scan_all: bool,
) -> frozenset[str] | None:
    """
    Return project allowlist for discovery.

    None means no allowlist (all discovered minus excludes). Default is the three
    modulator OOT targets unless --filter or include_projects in config overrides.
    """
    if scan_all:
        return None
    if cli_filter is not None:
        return cli_filter
    if config_includes:
        return frozenset(config_includes)
    return DEFAULT_INCLUDE_PROJECTS


def project_exclusion_reason(project_name: str) -> str | None:
    """Return exclusion reason if this project must not be validated."""
    if project_name in EXCLUSION_REASONS:
        return EXCLUSION_REASONS[project_name]
    if project_name in DEFAULT_EXCLUDE_PROJECTS:
        return "Excluded by rmv scan policy (see scan/exclusions.py)"
    return None


def mode_exclusion_reason(mode_name: str) -> str | None:
    """Return reason if this README mode must not be validated."""
    if mode_name.upper() in {m.upper() for m in NON_VALIDATABLE_MODE_NAMES}:
        key = mode_name.upper()
        if key == "DSSS":
            return MODE_EXCLUSION_REASONS["DSSS"]
        if key == "GDSS":
            return MODE_EXCLUSION_REASONS["GDSS"]
    if mode_name in MODE_EXCLUSION_REASONS:
        return MODE_EXCLUSION_REASONS[mode_name]
    return None


def is_non_validatable_mode(spec: ModeSpec) -> bool:
    """True when IQ generation and CNN validation should be skipped for this mode."""
    if spec.generation_method == "skip":
        return True
    return spec.mode_name.upper() in {m.upper() for m in NON_VALIDATABLE_MODE_NAMES}
