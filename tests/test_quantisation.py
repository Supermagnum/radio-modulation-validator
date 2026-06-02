"""Tests for INT8 quantisation and NPU export (no SpacemiT SDK required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from rmv.cli import app
from rmv.export_npu import convert_to_npu, find_npu_convert, run_export_npu
from rmv.export_quantised import (
    load_calibration_data,
    quantise_model,
    verify_quantised_accuracy,
)
from rmv.export_quantised import run_export_quantised
from rmv.models_paths import (
    resolve_family_onnx_model,
    resolve_onnx_model,
    resolve_order_onnx_model,
)
from rmv.scan.backend import detect_cpu_classifier

runner = CliRunner()


@pytest.fixture
def calibration_npz(tmp_path: Path) -> Path:
    n = 800
    samples = np.random.randn(n, 2, 1024).astype(np.float32)
    snr_db = np.concatenate(
        [
            np.full(400, -10.0, dtype=np.float32),
            np.full(400, 10.0, dtype=np.float32),
        ]
    )
    labels = np.zeros(n, dtype=np.int32)
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(
        path,
        samples=samples,
        labels=labels,
        snr_db=snr_db,
        class_names=np.array(["A"], dtype=object),
        source="test",
    )
    return path


def test_load_calibration_data_shape(calibration_npz: Path) -> None:
    data = load_calibration_data(calibration_npz, n_chunks=64, snr_db_min=0.0, seed=0)
    assert data.shape == (64, 2, 1024)
    assert data.dtype == np.float32


def test_load_calibration_data_high_snr_filter(calibration_npz: Path) -> None:
    npz = np.load(calibration_npz)
    data = load_calibration_data(calibration_npz, n_chunks=100, snr_db_min=0.0, seed=1)
    assert len(data) == 100
    high_mask = npz["snr_db"] >= 0.0
    assert high_mask.sum() >= 100


def test_load_calibration_data_uses_all_when_few_high_snr(tmp_path: Path) -> None:
    samples = np.random.randn(10, 2, 1024).astype(np.float32)
    snr_db = np.full(10, -15.0, dtype=np.float32)
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(
        path,
        samples=samples,
        labels=np.zeros(10, dtype=np.int32),
        snr_db=snr_db,
        class_names=np.array(["A"], dtype=object),
        source="test",
    )
    data = load_calibration_data(path, n_chunks=512, snr_db_min=0.0)
    assert data.shape == (10, 2, 1024)  # all samples when none meet snr_min


@pytest.fixture
def tiny_fp32_onnx(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 8, kernel_size=5)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(8, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.conv(x)
            y = self.pool(y).squeeze(-1)
            return self.fc(y)

    model = Tiny()
    model.eval()
    out = tmp_path / "tiny.onnx"
    dummy = torch.randn(1, 2, 1024, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["iq_samples"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    return out


def test_quantise_model_produces_int8_onnx(
    tiny_fp32_onnx: Path,
    calibration_npz: Path,
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnxruntime.quantization")
    import onnx

    cal = load_calibration_data(calibration_npz, n_chunks=32, snr_db_min=0.0, seed=2)
    int8_path = tmp_path / "tiny_int8.onnx"
    quantise_model(tiny_fp32_onnx, int8_path, cal)
    assert int8_path.is_file()
    onnx.load(str(int8_path))


def test_verify_accuracy_passes_within_tolerance(
    tiny_fp32_onnx: Path,
    calibration_npz: Path,
    mocker: pytest.MockFixture,
) -> None:
    cal = load_calibration_data(calibration_npz, n_chunks=16, snr_db_min=0.0, seed=3)
    int8_path = tiny_fp32_onnx

    logits = np.zeros((16, 3), dtype=np.float32)
    logits[:, 1] = 5.0

    mock_sess = MagicMock()
    inp = MagicMock()
    inp.name = "iq_samples"
    mock_sess.get_inputs.return_value = [inp]
    mock_sess.run.return_value = [logits]

    mocker.patch(
        "onnxruntime.InferenceSession",
        return_value=mock_sess,
    )
    assert verify_quantised_accuracy(
        tiny_fp32_onnx,
        int8_path,
        cal,
        ["a", "b", "c"],
        tolerance_pct=3.0,
    )


def test_verify_accuracy_fails_on_large_drop(
    tiny_fp32_onnx: Path,
    calibration_npz: Path,
    mocker: pytest.MockFixture,
) -> None:
    cal = load_calibration_data(calibration_npz, n_chunks=20, snr_db_min=0.0, seed=4)

    def make_session(agree: bool) -> MagicMock:
        sess = MagicMock()
        inp = MagicMock()
        inp.name = "iq_samples"
        sess.get_inputs.return_value = [inp]

        def run(_outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            batch = list(feed.values())[0].shape[0]
            logits = np.zeros((batch, 3), dtype=np.float32)
            if agree:
                logits[:, 0] = 10.0
            else:
                for i in range(batch):
                    logits[i, i % 3] = 10.0
            return [logits]

        sess.run.side_effect = run
        return sess

    sessions = [make_session(True), make_session(False)]
    mocker.patch(
        "onnxruntime.InferenceSession",
        side_effect=sessions,
    )
    assert not verify_quantised_accuracy(
        tiny_fp32_onnx,
        tiny_fp32_onnx,
        cal,
        ["a", "b", "c"],
        tolerance_pct=3.0,
    )


def test_find_npu_convert_not_found(mocker: pytest.MockFixture) -> None:
    mocker.patch("rmv.export_npu.shutil.which", return_value=None)
    with patch.object(Path, "is_file", return_value=False):
        assert find_npu_convert() is None


def test_convert_to_npu_tool_missing(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mocker.patch("rmv.export_npu.find_npu_convert", return_value=None)
    int8 = tmp_path / "family_classifier_int8.onnx"
    int8.write_bytes(b"x")
    out = tmp_path / "family_classifier.nb"
    assert convert_to_npu(int8, out) is False


def test_convert_to_npu_success(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    converter = tmp_path / "spacemit-npu-convert"
    converter.write_text("#!/bin/sh\ntouch \"$4\"\n", encoding="utf-8")
    converter.chmod(0o755)
    mocker.patch("rmv.export_npu.find_npu_convert", return_value=converter)

    int8 = tmp_path / "model_int8.onnx"
    int8.write_bytes(b"onnx")
    out = tmp_path / "model.nb"

    mock_run = mocker.patch(
        "rmv.export_npu.subprocess.run",
        return_value=mocker.Mock(returncode=0, stderr=""),
    )
    assert convert_to_npu(int8, out, calibration_data_path=tmp_path / "cal.npz")
    assert mock_run.called


def test_run_export_npu_skips_missing_int8(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mocker.patch("rmv.export_npu.convert_to_npu", return_value=True)
    paths = run_export_npu(tmp_path)
    assert paths == []


def test_resolve_onnx_model_prefers_int8(tmp_path: Path) -> None:
    fp32 = tmp_path / "family_classifier.onnx"
    int8 = tmp_path / "family_classifier_int8.onnx"
    fp32.write_bytes(b"fp32")
    int8.write_bytes(b"int8")
    assert resolve_onnx_model(tmp_path, "family_classifier") == int8
    assert resolve_family_onnx_model(tmp_path) == int8


def test_resolve_order_ignores_int8(tmp_path: Path) -> None:
    fp32 = tmp_path / "order_classifier.onnx"
    int8 = tmp_path / "order_classifier_int8.onnx"
    fp32.write_bytes(b"fp32")
    int8.write_bytes(b"int8")
    assert resolve_order_onnx_model(tmp_path) == fp32


def test_detect_cpu_classifier_int8(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    family_int8 = tmp_path / "family_classifier_int8.onnx"
    order_fp32 = tmp_path / "order_classifier.onnx"
    family_int8.write_bytes(b"fake")
    order_fp32.write_bytes(b"fp32")
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="iq_samples")]
    session.run.return_value = [np.zeros((1, 3), dtype=np.float32)]
    mocker.patch("onnxruntime.InferenceSession", return_value=session)
    assert detect_cpu_classifier(tmp_path) is True


def test_run_export_quantised_order_failure_keeps_family(
    tmp_path: Path,
    calibration_npz: Path,
    mocker: pytest.MockFixture,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "family_classifier.onnx").write_bytes(b"fp32")
    (models_dir / "order_classifier.onnx").write_bytes(b"fp32")
    (models_dir / "family_classifier.meta.json").write_text('{"class_names": ["FM"]}')
    (models_dir / "order_classifier.meta.json").write_text('{"class_names": ["BPSK"]}')

    family_int8 = models_dir / "family_classifier_int8.onnx"
    order_int8 = models_dir / "order_classifier_int8.onnx"

    def side_effect(stem: str, *_args: object, **_kwargs: object) -> bool:
        if stem == "family_classifier":
            family_int8.write_bytes(b"ok")
            return True
        order_int8.write_bytes(b"bad")
        order_int8.unlink()
        return False

    mocker.patch(
        "rmv.export_quantised._quantise_and_verify",
        side_effect=side_effect,
    )
    mocker.patch(
        "rmv.export_quantised.load_calibration_data",
        return_value=np.zeros((8, 2, 1024), dtype=np.float32),
    )

    written = run_export_quantised(calibration_npz, models_dir)
    assert written == [family_int8]
    assert not order_int8.is_file()


def test_run_export_quantised_family_only(
    tmp_path: Path,
    calibration_npz: Path,
    mocker: pytest.MockFixture,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "family_classifier.onnx").write_bytes(b"fp32")
    (models_dir / "order_classifier.onnx").write_bytes(b"fp32")

    mocker.patch(
        "rmv.export_quantised._quantise_and_verify",
        return_value=True,
    )
    mocker.patch(
        "rmv.export_quantised.load_calibration_data",
        return_value=np.zeros((8, 2, 1024), dtype=np.float32),
    )
    written = run_export_quantised(
        calibration_npz,
        models_dir,
        family_only=True,
        skip_verify=True,
    )
    assert len(written) == 1
    assert written[0].name == "family_classifier_int8.onnx"


def test_cli_export_quantised(
    tmp_path: Path,
    mocker: pytest.MockFixture,
) -> None:
    written = [
        tmp_path / "models" / "family_classifier_int8.onnx",
        tmp_path / "models" / "order_classifier_int8.onnx",
    ]
    for p in written:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"int8")

    mocker.patch(
        "rmv.export_quantised.run_export_quantised",
        return_value=written,
    )
    result = runner.invoke(
        app,
        [
            "export-quantised",
            "--synthetic",
            str(tmp_path / "synthetic.npz"),
            "--models-dir",
            str(tmp_path / "models"),
            "--skip-verify",
        ],
    )
    assert result.exit_code == 0
    assert "family_classifier_int8.onnx" in result.stdout


def test_cli_export_npu_no_tool(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mocker.patch("rmv.export_npu.run_export_npu", return_value=[])
    result = runner.invoke(
        app,
        ["export-npu", "--int8-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
