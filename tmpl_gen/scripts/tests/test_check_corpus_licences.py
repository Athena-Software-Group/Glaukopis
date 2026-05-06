"""Smoke tests for tmpl_gen/scripts/check_corpus_licences.py
(v13 build-time licence-allowlist gate, v13_plan.txt sec 4.5 + sec 10).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tmpl_gen" / "scripts" / "check_corpus_licences.py"


def _run(corpus_path: Path, report_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(corpus_path),
         "--report", str(report_path), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows))


def test_all_allowlisted_passes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    report = tmp_path / "report.json"
    _write(corpus, [
        {"instruction": "x", "input": "", "output": "y",
         "source": "athena-cti-db-internal", "shortname": "T1"},
        {"instruction": "x", "input": "", "output": "y",
         "source": "misp-galaxy-cc0", "shortname": "T2"},
        {"instruction": "x", "input": "", "output": "y",
         "source": "mitre-attack-custom", "shortname": "T3"},
    ])
    r = _run(corpus, report)
    assert r.returncode == 0, r.stderr
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "ok"
    assert rep["fail_row_count"] == 0
    assert rep["missing_source_field"] == 0


def test_denied_source_fails(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    report = tmp_path / "report.json"
    _write(corpus, [
        {"instruction": "x", "input": "", "output": "y",
         "source": "athena-cti-db-internal", "shortname": "T1"},
        {"instruction": "x", "input": "", "output": "y",
         "source": "crowdstrike-proprietary", "shortname": "TBAD"},
    ])
    r = _run(corpus, report)
    assert r.returncode != 0
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "fail"
    assert "crowdstrike-proprietary" in rep["by_status"]["denied"]


def test_unknown_source_fails(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    report = tmp_path / "report.json"
    _write(corpus, [
        {"instruction": "x", "input": "", "output": "y",
         "source": "some-random-blog", "shortname": "T1"},
    ])
    r = _run(corpus, report)
    assert r.returncode != 0
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "fail"
    assert "some-random-blog" in rep["by_status"]["unknown"]


def test_missing_source_field_fails(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    report = tmp_path / "report.json"
    _write(corpus, [
        {"instruction": "x", "input": "", "output": "y", "shortname": "T1"},
    ])
    r = _run(corpus, report)
    assert r.returncode != 0
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "fail"
    assert rep["missing_source_field"] == 1


def test_alpaca_nc_denied_by_default_allowed_with_flag(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    report = tmp_path / "report.json"
    _write(corpus, [
        {"instruction": "x", "input": "", "output": "y",
         "source": "alpaca-cc-by-nc-4", "shortname": "T1"},
    ])
    r = _run(corpus, report)
    assert r.returncode != 0, "alpaca-cc-by-nc-4 must be denied without --allow-nc"
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "fail"
    assert "alpaca-cc-by-nc-4" in rep["by_status"]["denied-nc"]

    r = _run(corpus, report, "--allow-nc")
    assert r.returncode == 0, r.stderr
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "ok"
    assert "alpaca-cc-by-nc-4" in rep["by_status"]["allowed-nc"]
