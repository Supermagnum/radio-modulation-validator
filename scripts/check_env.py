#!/usr/bin/env python3
"""Verify Python environment for radio-modulation-validator."""

from __future__ import annotations

import sys


def _check(name: str, fn: object) -> bool:
    try:
        fn()  # type: ignore[operator]
        print(f"OK   {name}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        if name.startswith("GNU Radio") and "multiarray" in str(exc):
            print(
                "      Hint: run `uv sync` in the project root (rmv pins numpy<2 for "
                "system GNU Radio when the venv uses --system-site-packages)."
            )
        return False


def main() -> int:
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print()

    required_ok = True
    required_ok &= _check(
        "onnxruntime (inference)",
        lambda: __import__("onnxruntime"),
    )
    required_ok &= _check(
        "rmv package",
        lambda: __import__("rmv"),
    )
    required_ok &= _check(
        "RadioModulationValidator",
        lambda: __import__("rmv", fromlist=["RadioModulationValidator"]).RadioModulationValidator,
    )

    print()
    print("Optional:")
    def _torch() -> None:
        torch = __import__("torch")
        print(f"      CUDA available: {torch.cuda.is_available()}")

    _check("PyTorch (rmv train only; install: uv sync --extra train)", _torch)
    _check(
        "GNU Radio 3 (capture IQ in GRC; not required for rmv validate)",
        lambda: __import__("gnuradio.analog", fromlist=["analog"]),
    )

    print()
    if "radio-modulation-validator" not in sys.executable and "rmv" not in str(
        __import__("sys").prefix
    ):
        if not any(p.endswith(".venv") for p in sys.path):
            print(
                "Hint: use the project venv, not system python3:\n"
                "  cd /path/to/radio-modulation-validator\n"
                "  uv sync --extra dev --extra train\n"
                "  .venv/bin/python scripts/check_env.py\n"
                "  # or: uv run python scripts/check_env.py"
            )

    if not required_ok:
        print("\nRequired checks failed. Install the package in this environment:")
        print("  uv sync --extra dev")
        print("  uv pip install -e '.[train]'   # if you need rmv train")
        return 1

    print("\nRequired components ready for rmv validate / classify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
