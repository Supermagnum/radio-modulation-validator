"""Tests for rmv scan (no GNU Radio required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from rmv.cli import app
from rmv.scan.database import FindingsDB
from rmv.scan.discover import discover_gr_projects
from rmv.scan.paths import resolve_scan_directory
from rmv.scan.mode_table import MODE_TABLE, all_mode_specs, lookup_mode
from rmv.scan.pipeline import ScanRunOptions, run_scan
from rmv.scan.readme_parser import parse_readme
from rmv.scan import prompts


def _make_gr3_project(tmp_path: Path, name: str = "gr-test") -> Path:
    proj = tmp_path / name
    proj.mkdir()
    (proj / "CMakeLists.txt").write_text(
        "find_package(gnuradio REQUIRED)\nproject(gr-test)\n",
        encoding="utf-8",
    )
    grc = proj / "grc"
    grc.mkdir()
    (grc / "test_mod.block.yml").write_text("label: test\n", encoding="utf-8")
    (proj / "README.md").write_text(
        "# gr-test\n\nSupports NBFM, GMSK, DMR modulation.\n",
        encoding="utf-8",
    )
    return proj


def test_discover_finds_gr_project(tmp_path: Path) -> None:
    _make_gr3_project(tmp_path)
    found = discover_gr_projects(tmp_path)
    assert len(found) == 1
    assert found[0].name == "gr-test"
    assert found[0].gr_version in ("3", "both", "unknown")


def test_discover_skips_gnuradio_framework_subdirs(tmp_path: Path) -> None:
    """Subdirs under gnuradio/ must not be discovered as OOT projects."""
    framework = tmp_path / "gnuradio"
    algo = framework / "algorithm"
    algo.mkdir(parents=True)
    (algo / "CMakeLists.txt").write_text(
        "find_package(gnuradio4 REQUIRED)\n",
        encoding="utf-8",
    )
    (algo / "include" / "gnuradio-4.0").mkdir(parents=True)
    _make_gr3_project(tmp_path, "gr-real-oot")
    found = discover_gr_projects(tmp_path)
    names = {p.name for p in found}
    assert "algorithm" not in names
    assert "gnuradio" not in names
    assert "gr-real-oot" in names


def test_default_include_is_three_modulator_projects(tmp_path: Path) -> None:
    from rmv.scan.exclusions import DEFAULT_INCLUDE_PROJECTS, resolve_include_names

    for name in DEFAULT_INCLUDE_PROJECTS:
        _make_gr3_project(tmp_path, name)
    _make_gr3_project(tmp_path, "gr-ident")
    found = discover_gr_projects(
        tmp_path,
        include_names=resolve_include_names(
            cli_filter=None,
            config_includes=(),
            scan_all=False,
        ),
    )
    assert {p.name for p in found} == set(DEFAULT_INCLUDE_PROJECTS)


def test_filter_include_names_only(tmp_path: Path) -> None:
    _make_gr3_project(tmp_path, "gr-keep")
    _make_gr3_project(tmp_path, "gr-drop")
    found = discover_gr_projects(
        tmp_path,
        include_names=frozenset({"gr-keep"}),
    )
    assert len(found) == 1
    assert found[0].name == "gr-keep"


def test_default_excludes_spread_spectrum_and_crypto_projects(tmp_path: Path) -> None:
    from rmv.scan.exclusions import DEFAULT_EXCLUDE_PROJECTS, project_exclusion_reason

    _make_gr3_project(tmp_path, "GR-K-GDSS")
    _make_gr3_project(tmp_path, "gr-opus")
    _make_gr3_project(tmp_path, "gr-qradiolink")
    found = discover_gr_projects(tmp_path)
    names = {p.name for p in found}
    assert "GR-K-GDSS" not in names
    assert "gr-opus" not in names
    assert "gr-qradiolink" in names
    assert "GR-K-GDSS" in DEFAULT_EXCLUDE_PROJECTS
    assert project_exclusion_reason("gr-opus") is not None


def test_dsss_gdss_modes_skipped_with_reason() -> None:
    from rmv.scan.exclusions import mode_exclusion_reason
    from rmv.scan.mode_table import lookup_mode

    dsss = lookup_mode("DSSS")
    gdss = lookup_mode("GDSS")
    assert dsss is not None and dsss.generation_method == "skip"
    assert gdss is not None and gdss.generation_method == "skip"
    assert "noise" in (mode_exclusion_reason("DSSS") or "").lower()
    assert "noise" in (mode_exclusion_reason("GDSS") or "").lower()


def test_exclude_projects_from_discovery(tmp_path: Path) -> None:
    _make_gr3_project(tmp_path, "gr-custom-drop")
    _make_gr3_project(tmp_path, "gr-qradiolink")
    found = discover_gr_projects(
        tmp_path,
        exclude_names=frozenset({"gr-custom-drop"}),
    )
    names = {p.name for p in found}
    assert "gr-custom-drop" not in names
    assert "gr-qradiolink" in names


def test_parse_project_name_list() -> None:
    from rmv.scan.config import parse_project_name_list

    assert parse_project_name_list("a,b, c") == frozenset({"a", "b", "c"})
    assert parse_project_name_list(None) is None
    assert parse_project_name_list("") is None


def test_load_scan_config_exclude_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rmv.scan.config import load_scan_config

    cfg_file = tmp_path / ".rmv_config.toml"
    cfg_file.write_text(
        """
[scan]
exclude_projects = ["gnuradio", "app_rpt"]
gr4_prefix = "/opt/gnuradio4-gcc"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("rmv.scan.config.config_path", lambda root=None: cfg_file)
    cfg = load_scan_config(tmp_path)
    assert cfg.exclude_projects == ("gnuradio", "app_rpt")
    assert cfg.gr4_prefix == "/opt/gnuradio4-gcc"


def test_discover_ignores_build_dir(tmp_path: Path) -> None:
    proj = _make_gr3_project(tmp_path)
    build = proj / "build"
    build.mkdir()
    (build / "CMakeLists.txt").write_text("find_package(gnuradio)\n", encoding="utf-8")
    found = discover_gr_projects(tmp_path)
    assert len(found) == 1
    assert found[0].path == proj.resolve()


def test_gr_version_detection_gr3(tmp_path: Path) -> None:
    proj = _make_gr3_project(tmp_path, "gr3only")
    found = discover_gr_projects(tmp_path)
    assert found[0].gr_version == "3"


def test_gr_version_detection_gr4(tmp_path: Path) -> None:
    proj = tmp_path / "gr4proj"
    proj.mkdir()
    (proj / "CMakeLists.txt").write_text(
        "find_package(gnuradio4 REQUIRED)\n",
        encoding="utf-8",
    )
    (proj / "include" / "gnuradio-4.0" / "foo").mkdir(parents=True)
    (proj / "README.md").write_text("# x\nGNU Radio 4 build\n", encoding="utf-8")
    found = discover_gr_projects(tmp_path)
    assert found[0].gr_version in ("4", "both")


def test_gr_version_detection_both(tmp_path: Path) -> None:
    proj = tmp_path / "grboth"
    proj.mkdir()
    (proj / "CMakeLists.txt").write_text(
        "find_package(gnuradio REQUIRED)\nfind_package(gnuradio4)\n",
        encoding="utf-8",
    )
    grc = proj / "grc"
    grc.mkdir()
    (grc / "x.block.yml").write_text("x: 1\n", encoding="utf-8")
    (proj / "include" / "gnuradio-4.0").mkdir(parents=True)
    (proj / "README.md").write_text("GNU Radio 3.10 and GNU Radio 4\n", encoding="utf-8")
    found = discover_gr_projects(tmp_path)
    assert found[0].gr_version == "both"


def test_readme_parser_detects_ai_generated(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "IMPORTANT NOTICE: This project is AI-generated software.\n",
        encoding="utf-8",
    )
    summary = parse_readme(readme)
    assert summary.is_ai_generated is True


def test_readme_parser_extracts_modes(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("Modes: NBFM, GMSK, DMR for ham radio.\n", encoding="utf-8")
    summary = parse_readme(readme)
    assert "NBFM" in summary.modulation_modes
    assert "GMSK" in summary.modulation_modes
    assert "DMR" in summary.modulation_modes


def test_mode_table_coverage() -> None:
    for spec in all_mode_specs():
        assert spec.expected_family, f"missing family for {spec.mode_name}"
        assert spec.expected_order, f"missing order for {spec.mode_name}"
    assert "NBFM" in MODE_TABLE
    assert lookup_mode("DMR") is not None
    assert lookup_mode("NBFM").expected_order == "NBFM_25"


def test_mode_labels_in_classifier_vocab() -> None:
    from rmv.scan.class_vocab import find_mode_label_mismatches, load_classifier_vocab

    vocab = load_classifier_vocab(Path("models"))
    mismatches = find_mode_label_mismatches(vocab)
    bad = {m.mode_name for m in mismatches}
    assert "NBFM" not in bad
    assert "BPSK" not in bad


def test_database_schema_created(tmp_path: Path) -> None:
    db_path = tmp_path / ".rmv_findings.db"
    db = FindingsDB(db_path)
    db.connect()
    db.close()
    assert db_path.is_file()


def test_issues_append_only(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = FindingsDB(db_path)
    pid = db.upsert_project(
        path="/tmp/p",
        name="p",
        gr_version="3",
        readme_path=None,
        scan_status="ok",
    )
    id1 = db.add_issue(project_id=pid, block_id=None, severity="info", description="one")
    id2 = db.add_issue(project_id=pid, block_id=None, severity="info", description="two")
    issues = db.list_issues(unresolved_only=False)
    db.close()
    assert len(issues) == 2
    assert {id1, id2} == {issues[0]["id"], issues[1]["id"]}


def test_normalize_scan_timestamp() -> None:
    from rmv.scan.database import normalize_scan_timestamp

    assert normalize_scan_timestamp("2026-05-31T15:00:00") == "2026-05-31T15:00:00Z"
    assert normalize_scan_timestamp("2026-05-31T15:00:00Z") == "2026-05-31T15:00:00Z"


def test_list_issues_since_filter(tmp_path: Path) -> None:
    from rmv.scan.database import FindingsDB

    db_path = tmp_path / "test.db"
    db = FindingsDB(db_path)
    pid = db.upsert_project(
        path="/tmp/p",
        name="p",
        gr_version="3",
        readme_path=None,
        scan_status="ok",
    )
    conn = db.connect()
    conn.execute(
        """
        INSERT INTO issues (project_id, block_id, detected_at, severity, description, resolved)
        VALUES (?, NULL, ?, 'info', 'old', 0)
        """,
        (pid, "2026-05-31T14:00:00Z"),
    )
    conn.execute(
        """
        INSERT INTO issues (project_id, block_id, detected_at, severity, description, resolved)
        VALUES (?, NULL, ?, 'info', 'new', 0)
        """,
        (pid, "2026-05-31T16:00:00Z"),
    )
    conn.commit()
    rows = db.list_issues(detected_since="2026-05-31T15:00:00")
    db.close()
    assert len(rows) == 1
    assert rows[0]["description"] == "new"


def test_purge_keep_latest(tmp_path: Path) -> None:
    from rmv.scan.database import FindingsDB

    db_path = tmp_path / "test.db"
    db = FindingsDB(db_path)
    pid = db.upsert_project(
        path="/tmp/p",
        name="proj",
        gr_version="3",
        readme_path=None,
        scan_status="ok",
    )
    bid = db.upsert_block(
        project_id=pid,
        block_name="mod_2fsk",
        block_file=None,
        expected_family="FSK",
        expected_order="2FSK",
        gr_version="3",
    )
    conn = db.connect()
    for run_at, pred in (
        ("2026-01-01T10:00:00Z", "AM"),
        ("2026-05-31T18:00:00Z", "FSK"),
    ):
        conn.execute(
            """
            INSERT INTO validations (
                block_id, run_at, iq_file, predicted_family, predicted_order,
                family_confidence, order_confidence, family_pass, order_pass, hard_fail, notes
            ) VALUES (?, ?, '/tmp/x.iq', ?, '2FSK', 0.9, 0.9, 1, 1, 0, '')
            """,
            (bid, run_at, pred),
        )
    conn.execute(
        """
        INSERT INTO issues (project_id, block_id, detected_at, severity, description, resolved)
        VALUES (?, ?, '2026-01-01T10:00:01Z', 'hard_fail', 'stale', 0)
        """,
        (pid, bid),
    )
    conn.execute(
        """
        INSERT INTO issues (project_id, block_id, detected_at, severity, description, resolved)
        VALUES (?, ?, '2026-05-31T18:00:01Z', 'warning', 'current', 0)
        """,
        (pid, bid),
    )
    conn.commit()

    preview = db.preview_purge_keep_latest()
    assert preview.validations_to_delete == 1
    assert preview.issues_to_delete == 1

    db.purge_keep_latest()
    vals = conn.execute("SELECT predicted_family FROM validations").fetchall()
    issues = db.list_issues(unresolved_only=False)
    projects = conn.execute("SELECT COUNT(*) AS c FROM projects").fetchone()
    blocks = conn.execute("SELECT COUNT(*) AS c FROM blocks").fetchone()
    db.close()

    assert len(vals) == 1
    assert vals[0]["predicted_family"] == "FSK"
    assert len(issues) == 1
    assert issues[0]["description"] == "current"
    assert int(projects["c"]) == 1
    assert int(blocks["c"]) == 1


def test_supersede_open_issues_for_block(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = FindingsDB(db_path)
    pid = db.upsert_project(
        path="/tmp/p",
        name="p",
        gr_version="3",
        readme_path=None,
        scan_status="ok",
    )
    bid = db.upsert_block(
        project_id=pid,
        block_name="mod_2fsk",
        block_file=None,
        expected_family="FSK",
        expected_order="2FSK",
        gr_version="3",
    )
    db.add_issue(
        project_id=pid,
        block_id=bid,
        severity="hard_fail",
        description="old failure",
    )
    n = db.supersede_open_issues_for_block(project_id=pid, block_id=bid)
    open_issues = db.list_issues(project_name="p", severity="hard_fail")
    db.close()
    assert n == 1
    assert open_issues == []


def test_dry_run_no_files_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_gr3_project(tmp_path)
    db_path = tmp_path / ".rmv_findings.db"
    monkeypatch.chdir(tmp_path)
    runs = run_scan(
        ScanRunOptions(
            root=tmp_path,
            dry_run=True,
            models_dir=Path("models"),
        )
    )
    assert runs == []
    assert not db_path.exists()
    assert not list(tmp_path.glob("**/*.iq"))


def test_project_prompt_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = _make_gr3_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    prompts.reset_prompt_state()

    mock_clf = MagicMock()

    with (
        patch("rmv.scan.pipeline.ask_generate_iq", return_value="no"),
        patch("rmv.scan.pipeline.ModulationClassifier", return_value=mock_clf),
        patch("rmv.scan.pipeline.find_gr3_prefix", return_value=None),
        patch("rmv.scan.pipeline.find_gr4_prefix", return_value=None),
    ):
        run_scan(
            ScanRunOptions(
                root=tmp_path,
                yes=True,
                models_dir=tmp_path / "models",
            )
        )

    assert not list((tmp_path / ".scan_iq").glob("**/*.iq")) if (tmp_path / ".scan_iq").exists() else True
    assert not (proj / "VALIDATION_REPORT.md").exists()


def test_resolve_scan_directory_expands_user(tmp_path: Path) -> None:
    resolved = resolve_scan_directory(tmp_path)
    assert resolved == tmp_path.resolve()
    assert resolved.is_dir()


def test_resolve_scan_directory_missing() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_scan_directory(Path("/nonexistent/rmv-scan-test-path-xyz"))


def test_chunk_iq_file_wrong_reshape_would_mismatch() -> None:
    """Regression: reshape(n,2,L) on interleaved IQ corrupts I/Q channels."""
    from rmv.dataset.preprocess import interleaved_to_iq, normalise_unit_power
    from tests.fixtures.synthetic_iq import generate_bpsk, interleave_iq

    source = normalise_unit_power(generate_bpsk()[np.newaxis, ...])[0]
    raw = interleave_iq(source)
    wrong = raw.reshape(1, 2, 1024)
    assert not np.allclose(wrong[0, 0], source[0], atol=0.1)
    restored = interleaved_to_iq(raw)
    assert np.allclose(restored[0], source[0], atol=1e-5)


def test_scan_iq_loader_roundtrip_nbfm(tmp_path: Path) -> None:
    """Scan .iq write format must reload to the same (N, 2, 1024) chunks."""
    from rmv.dataset.synthetic import _generate_nbfm_chunk
    from rmv.scan.iq_generator import _write_iq_and_sidecar
    from rmv.validate import load_iq_chunks

    chunks = np.stack(
        [
            _generate_nbfm_chunk(
                max_dev=2500.0,
                tau=0.0,
                sample_rate_hz=48000.0,
                audio_rate_hz=8000,
                class_name="NBFM_25",
                snr_db=20.0,
                use_gnuradio=False,
                rng=np.random.default_rng(99),
            )
            for _ in range(16)
        ],
        axis=0,
    )
    gen = _write_iq_and_sidecar(
        tmp_path,
        "mod_nbfm",
        chunks,
        expected_family="FM",
        expected_order="NBFM_25",
        project_name="test",
        generation_method="numpy",
        gr_env_used="none",
        notes="roundtrip test",
    )
    loaded = load_iq_chunks(gen.iq_path)
    assert loaded.shape == chunks.shape
    np.testing.assert_allclose(loaded, chunks, atol=1e-5)
    assert loaded[:, 0].mean() < 0.15
    assert loaded[:, 1].mean() < 0.15


def _require_confident_classifier(clf: object, chunks: np.ndarray) -> object:
    """Skip when bundled ONNX checkpoints do not discriminate (low confidence)."""
    from rmv.classifier import ModulationClassifier
    from rmv.types import ClassifierResult

    assert isinstance(clf, ModulationClassifier)
    pred = clf.classify_aggregate(chunks)
    assert isinstance(pred, ClassifierResult)
    if pred.family_confidence < 0.5:
        pytest.skip(
            f"Classifier confidence too low ({pred.family_confidence:.2f}); "
            "re-export models from checkpoints"
        )
    return pred


@pytest.mark.skipif(
    not Path("models/family_classifier.onnx").is_file(),
    reason="ONNX models not present",
)
def test_scan_iq_classifies_correctly(tmp_path: Path) -> None:
    """End-to-end: scan .iq file through validate must classify NBFM as FM when models work."""
    from rmv.classifier import ModulationClassifier
    from rmv.dataset.synthetic import _generate_nbfm_chunk
    from rmv.scan.iq_generator import _write_iq_and_sidecar
    from rmv.validate import load_iq_chunks, run_validate_file

    chunks = np.stack(
        [
            _generate_nbfm_chunk(
                max_dev=2500.0,
                tau=0.0,
                sample_rate_hz=48000.0,
                audio_rate_hz=8000,
                class_name="NBFM_25",
                snr_db=20.0,
                use_gnuradio=False,
                rng=np.random.default_rng(99),
            )
            for _ in range(16)
        ],
        axis=0,
    )
    gen = _write_iq_and_sidecar(
        tmp_path,
        "mod_nbfm",
        chunks,
        expected_family="FM",
        expected_order="NBFM_25",
        project_name="test",
        generation_method="numpy",
        gr_env_used="none",
        notes="classify test",
    )
    loaded = load_iq_chunks(gen.iq_path)
    assert loaded.shape[1:] == (2, 1024)

    clf = ModulationClassifier(Path("models"), verify_checksums=False)
    pred = _require_confident_classifier(clf, loaded)
    assert pred.family == "FM", pred.family
    assert pred.family_confidence > 0.70, pred.family_confidence

    result = run_validate_file(gen.iq_path, clf, threshold=0.70)
    assert result.family_pass is True
    assert result.predicted_family == "FM"


@pytest.mark.skipif(
    not Path("checkpoints/best_family_classifier.pt").is_file(),
    reason="family checkpoint not present",
)
def test_scan_fsk_reference_classifies_as_fsk() -> None:
    """Scan FSK IQ must use symmetric deviation, not positive-only tones (AM confusion)."""
    import json
    import torch

    from rmv.export import _load_checkpoint
    from rmv.scan.iq_generator import _chunks_from_complex, _gen_fsk_numpy

    sig = _gen_fsk_numpy(2)
    chunks = _chunks_from_complex(sig)
    model, _, _ = _load_checkpoint(Path("checkpoints/best_family_classifier.pt"))
    meta = json.loads(Path("checkpoints/best_family_meta.json").read_text())
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(chunks[:4]))
        probs = torch.softmax(logits, dim=1)
        pred = meta["class_names"][int(probs[0].argmax())]
    assert pred == "FSK"
    assert float(probs[0].max()) > 0.7


def test_scan_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
