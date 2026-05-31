"""Tests for dataset download, status, verification, and edge cases."""

from __future__ import annotations

import json
import pickle
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from rmv.cli import app
from rmv.dataset.checksums import DEFAULT_DOWNLOAD_TIMEOUT_SEC, sha256_file
from rmv.dataset.cspb_detect import CSPBVariant, analyze_cspb_directory, is_cspb_r2_download_link
from rmv.dataset.download import (
    DownloadError,
    convert_hisarmod_mat_to_h5,
    extract_radioml_tar,
    is_download_complete,
    is_matlab_v73,
    set_download_timeout,
    stream_download,
)
from rmv.dataset.download import get_download_timeout
from rmv.dataset.manifest import load_manifest, manifest_path, update_radioml_manifest
from rmv.dataset.manage import collect_status, ensure_datasets_for_training
from rmv.dataset.paths import (
    cspb_has_tim_files,
    detect_cspb,
    detect_cspb_present,
    detect_radioml,
    radioml_pkl_path,
)
from rmv.dataset.radioml_resolve import RadioMLPickleNotFoundError, extract_radioml_tar_strict
from tests.fixtures.synthetic_iq import radioml_mock_pickle


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_dataset_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code == 0
    assert "download" in result.stdout
    assert "info" in result.stdout


def test_resume_partial_download(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    """Partial file must resume with Range header, not skip as complete."""
    dest = tmp_path / "large.bin"
    partial = b"x" * 900
    dest.write_bytes(partial)
    remaining = b"y" * 200
    total_size = 1100

    class FakeResponse:
        status_code = 206
        headers = {
            "Content-Length": str(len(remaining)),
            "Content-Range": f"bytes 900-{total_size - 1}/{total_size}",
        }

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self, chunk_size: int = 0) -> object:
            del chunk_size
            yield remaining

    class FakeStream:
        def __init__(self) -> None:
            self.request_headers: dict[str, str] = {}

        def __enter__(self) -> FakeResponse:
            return FakeResponse()

        def __exit__(self, *args: object) -> None:
            pass

    fake_stream = FakeStream()

    mock_client = MagicMock()

    def stream(method: str, url: str, headers: dict[str, str] | None = None) -> FakeStream:
        del method, url
        if headers:
            fake_stream.request_headers = headers
        return fake_stream

    mock_client.stream = stream
    mock_client.head.return_value = MagicMock(
        status_code=200,
        headers={"Content-Length": str(total_size)},
    )
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mocker.patch("rmv.dataset.download.httpx.Client", return_value=mock_client)

    assert not is_download_complete(dest, None, total_size)
    stream_download("http://example.com/large.bin", dest, resume=True)
    assert dest.stat().st_size == total_size
    assert fake_stream.request_headers.get("Range") == "bytes=900-"


def test_checksum_mismatch_deletes_file(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"wrong content")
    expected = "a" * 64

    content = b"full file payload"

    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": str(len(content))}

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self, chunk_size: int = 0) -> object:
            del chunk_size
            yield content

    class FakeStream:
        def __enter__(self) -> FakeResponse:
            return FakeResponse()

        def __exit__(self, *args: object) -> None:
            pass

    mock_client = MagicMock()
    mock_client.stream.return_value = FakeStream()
    mock_client.head.return_value = MagicMock(status_code=200, headers={"Content-Length": str(len(content))})
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mocker.patch("rmv.dataset.download.httpx.Client", return_value=mock_client)

    with pytest.raises(DownloadError, match="Checksum mismatch"):
        stream_download("http://example.com/f", dest, expected_sha256=expected, resume=False)
    assert not dest.is_file()


def test_cspb_r2_warning_on_original(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cdir = tmp_path / "cspb"
    cdir.mkdir()
    (cdir / "CSPB.ML.2018_Batch1.zip").write_bytes(b"fake")
    (cdir / "signal_record.txt").write_text("1 bpsk 1 0 0 0 0 1 0\n")

    assert analyze_cspb_directory(cdir) == CSPBVariant.ORIGINAL
    assert not is_cspb_r2_download_link("https://example.com/cspb.ml.2018_batch1.zip")


def test_cspb_scraper_rejects_original_links() -> None:
    assert not is_cspb_r2_download_link(
        "https://cyclostationary.blog/cspb.ml.2018_batch1.zip",
        "CSPB.ML.2018 Batch 1",
    )
    assert is_cspb_r2_download_link(
        "https://cyclostationary.blog/cspb.ml.2018r2_batch1.zip",
        "CSPB.ML.2018R2 Batch 1",
    )


def test_hisarmod_mat_v73_uses_h5py(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    import h5py

    mat_path = tmp_path / "HisarMod.mat"
    out_path = tmp_path / "out.h5"
    with h5py.File(mat_path, "w") as f:
        f.create_dataset("test", data=np.zeros((2, 2)))

    assert is_matlab_v73(mat_path)
    mocker.patch("scipy.io.loadmat", side_effect=NotImplementedError("v7.3"))
    convert_hisarmod_mat_to_h5(mat_path, out_path)
    assert out_path.is_file()


def test_hisarmod_scipy_loadmat_failure_message(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mat_path = tmp_path / "old.mat"
    mat_path.write_bytes(b"MATLAB NOT HDF5")
    mocker.patch("rmv.dataset.download.is_matlab_v73", return_value=False)

    import scipy.io

    mocker.patch("scipy.io.loadmat", side_effect=NotImplementedError("v7.3"))
    with pytest.raises(DownloadError, match="scipy.io.loadmat cannot read"):
        convert_hisarmod_mat_to_h5(mat_path, tmp_path / "out.h5")


def test_manifest_written_after_download(tmp_path: Path) -> None:
    update_radioml_manifest(
        tmp_path,
        tar_sha256="a" * 64,
        pkl_sha256="b" * 64,
        status="verified",
    )
    mpath = manifest_path(tmp_path)
    assert mpath.is_file()
    data = load_manifest(tmp_path)
    assert data["schema_version"] == "1.0"
    assert data["datasets"]["radioml"]["status"] == "verified"
    assert data["datasets"]["radioml"]["version"] == "2016.10a"


def test_status_reads_manifest_not_rehash(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    update_radioml_manifest(tmp_path, pkl_sha256="c" * 64, status="verified")
    pkl = radioml_pkl_path(tmp_path)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(b"data")
    mocker.patch("rmv.dataset.manage.sha256_file", side_effect=AssertionError("should not hash"))
    rows = collect_status(tmp_path)
    assert rows[0].checksum == ("c" * 16)


def test_radioml_extraction_finds_correct_pkl(tmp_path: Path) -> None:
    wrong = tmp_path / "other.pkl"
    wrong.write_bytes(b"not radioml")
    inner = tmp_path / "nested"
    inner.mkdir()
    good = inner / "RML2016.10a_dict.pkl"
    pkl_data = {("BPSK", "0"): np.random.randn(1, 2, 128).astype(np.float32)}
    with good.open("wb") as f:
        pickle.dump(pkl_data, f)

    tar_path = tmp_path / "test.tar.bz2"
    with tarfile.open(tar_path, "w:bz2") as tar:
        tar.add(good, arcname="nested/RML2016.10a_dict.pkl")
        tar.add(wrong, arcname="other.pkl")

    out = extract_radioml_tar(tar_path, tmp_path / "radioml")
    assert out.name == "RML2016.10a_dict.pkl"


def test_radioml_zenodo_optimized_alias(tmp_path: Path) -> None:
    from rmv.dataset.radioml_resolve import ensure_canonical_pickle

    rdir = tmp_path / "radioml"
    rdir.mkdir()
    opt = rdir / "RML2016.10a_dict_optimized.pkl"
    opt.write_bytes(b"not a real pickle but enough for copy test")
    canonical = ensure_canonical_pickle(rdir)
    assert canonical.name == "RML2016.10a_dict.pkl"
    assert canonical.is_file()


def test_radioml_extraction_rejects_wrong_pkl_only(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong_only.pkl"
    with wrong.open("wb") as f:
        pickle.dump({"a": 1}, f)
    tar_path = tmp_path / "bad.tar.bz2"
    with tarfile.open(tar_path, "w:bz2") as tar:
        tar.add(wrong, arcname="wrong_only.pkl")
    with pytest.raises(RadioMLPickleNotFoundError):
        extract_radioml_tar_strict(tar_path, tmp_path / "out")


def test_train_prompts_on_missing_dataset(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mocker.patch("typer.confirm", return_value=False)
    mocker.patch("rmv.dataset.manage.download_datasets")
    r, h, c = ensure_datasets_for_training(
        tmp_path, None, None, None, interactive=True, auto_download=False
    )
    assert r is None and h is None and c is None


def test_train_auto_download_skips_prompt(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mock_dl = mocker.patch("rmv.dataset.manage.download_datasets", return_value=True)
    mocker.patch("typer.confirm", side_effect=AssertionError("should not prompt"))
    ensure_datasets_for_training(tmp_path, None, None, None, interactive=True, auto_download=True)
    assert mock_dl.called


def test_download_error_no_traceback(runner: CliRunner, tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mocker.patch(
        "rmv.dataset.cli.download_datasets",
        side_effect=DownloadError("network failed"),
    )
    result = runner.invoke(app, ["dataset", "download", "--dest", str(tmp_path), "--radioml"])
    assert result.exit_code == 1
    assert "network failed" in result.stderr or "network failed" in result.output
    assert "Traceback" not in (result.stderr or "")


def test_verify_exit_codes(runner: CliRunner, tmp_path: Path) -> None:
    missing = runner.invoke(app, ["dataset", "verify", "--dest", str(tmp_path)])
    assert missing.exit_code == 1

    pkl = radioml_pkl_path(tmp_path)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    radioml_mock_pickle(pkl)
    digest = sha256_file(pkl)
    update_radioml_manifest(tmp_path, pkl_sha256=digest, status="verified")
    partial = runner.invoke(app, ["dataset", "verify", "--dest", str(tmp_path)])
    out = partial.stderr or partial.output
    assert partial.exit_code == 1
    assert "HISARMOD 2019.1" in out
    assert "optional" in out.lower()
    assert "CSPB.ML.2018R2" in out


def test_verify_reports_cspb_when_hisarmod_missing(
    runner: CliRunner, tmp_path: Path
) -> None:
    pkl = radioml_pkl_path(tmp_path)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    radioml_mock_pickle(pkl)
    update_radioml_manifest(tmp_path, pkl_sha256=sha256_file(pkl), status="verified")

    cdir = tmp_path / "cspb"
    batch = cdir / "Batch_Dir_1"
    batch.mkdir(parents=True)
    (batch / "signal_1.tim").write_bytes(b"\x00" * 32)
    (cdir / "CSPB.ML_.2018R2_1.zip").write_bytes(b"zip")

    result = runner.invoke(app, ["dataset", "verify", "--dest", str(tmp_path)])
    out = result.stderr or result.output
    assert "CSPB.ML.2018R2" in out
    assert "truth file missing" in out.lower() or "signals present" in out.lower()


def test_download_cspb_truth_file(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    from rmv.dataset.download import download_cspb_truth_file

    cdir = tmp_path / "cspb"
    cdir.mkdir()
    content = b"1 bpsk 2 0 0 0 0 0 0 0.0\n"

    def fake_stream(url: str, dest: Path, **kwargs: object) -> None:
        del url, kwargs
        dest.write_bytes(content)

    mocker.patch("rmv.dataset.download.stream_download", side_effect=fake_stream)
    path = download_cspb_truth_file(tmp_path, force=True)
    assert path is not None
    assert path.name == "signal_record_C_2023.txt"
    assert (cdir / "signal_record.txt").is_file()


def test_cspb_batch_dir_detected_without_truth(tmp_path: Path) -> None:
    cdir = tmp_path / "cspb"
    batch = cdir / "Batch_Dir_1"
    batch.mkdir(parents=True)
    (batch / "signal_1.tim").write_bytes(b"\x00" * 32)
    (cdir / "CSPB.ML_.2018R2_1.zip").write_bytes(b"zip")

    assert detect_cspb_present(tmp_path) == cdir
    assert detect_cspb(tmp_path) is None
    assert cspb_has_tim_files(cdir)


def test_cspb_r2_truth_filename_detected(tmp_path: Path) -> None:
    from rmv.dataset.paths import find_cspb_truth_file

    cdir = tmp_path / "cspb"
    cdir.mkdir()
    truth = cdir / "signal_record_C_2023.txt"
    truth.write_text("1 bpsk 2 0 0 0 0 0 0 0.0\n", encoding="utf-8")
    assert find_cspb_truth_file(cdir) == truth


def test_find_cspb_truth_file_without_false_positive(tmp_path: Path) -> None:
    from rmv.dataset.paths import find_cspb_truth_file

    cdir = tmp_path / "cspb"
    cdir.mkdir()
    (cdir / "signal_1.tim").write_bytes(b"\x00" * 8)
    assert find_cspb_truth_file(cdir) is None


def test_timeout_configurable() -> None:
    set_download_timeout(600.0)
    t = get_download_timeout()
    assert t.read == 600.0
    set_download_timeout(DEFAULT_DOWNLOAD_TIMEOUT_SEC)


def test_checksum_update_writes_manifest(tmp_path: Path) -> None:
    pkl = radioml_pkl_path(tmp_path)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(b"manifest test")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["dataset", "checksum-update", "--dataset", "radioml", "--dest", str(tmp_path)],
    )
    assert result.exit_code == 0
    data = load_manifest(tmp_path)
    assert "radioml" in data["datasets"]


def test_empty_cspb_not_detected(tmp_path: Path) -> None:
    cdir = tmp_path / "cspb"
    cdir.mkdir()
    assert detect_cspb(tmp_path) is None
    from rmv.dataset.manage import collect_status

    rows = collect_status(tmp_path)
    cspb_row = [r for r in rows if "CSPB" in r.name][0]
    assert cspb_row.status in ("missing", "incomplete")
    assert cspb_row.status != "present"


def test_radioml_urls_use_deepsig_io() -> None:
    from rmv.dataset.checksums import RADIOML_DOWNLOAD_URLS

    assert any("deepsig.io" in u for u in RADIOML_DOWNLOAD_URLS)
    assert not any("deepsig.ai" in u for u in RADIOML_DOWNLOAD_URLS)


def test_detect_radioml(tmp_path: Path) -> None:
    pkl = radioml_pkl_path(tmp_path)
    pkl.parent.mkdir(parents=True)
    radioml_mock_pickle(pkl)
    found = detect_radioml(tmp_path)
    assert found is not None
    assert found.name == "RML2016.10a_dict.pkl"
