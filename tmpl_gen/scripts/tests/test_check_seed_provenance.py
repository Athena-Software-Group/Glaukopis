"""Smoke tests for tmpl_gen/scripts/check_seed_provenance.py
(v13 build-time seed-provenance gate, v13_plan.txt sec 4.5 + sec 10.3).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tmpl_gen" / "scripts" / "check_seed_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_seed_provenance", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _sha256(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def _make_seed(tmp_path: Path, name: str, content: bytes,
               provenance_text: str) -> tuple[Path, Path, str]:
    seed = tmp_path / name
    seed.write_bytes(content)
    pv = tmp_path / f"{name}.PROVENANCE.txt"
    pv.write_text(provenance_text)
    return seed, pv, _sha256(seed)


def test_passes_against_real_repo_seeds(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0, r.stderr
    rep = json.loads(report.read_text())
    assert rep["outcome"] == "ok"
    assert rep["failed_seeds"] == 0
    assert rep["total_seeds"] >= 3


def test_check_one_passes_for_valid_seed(tmp_path: Path) -> None:
    mod = _load_module()
    seed, pv, sha = _make_seed(tmp_path, "ok.bin", b"hello world",
                               "Licence: CC0 1.0 Public Domain")
    res = mod.check_one({
        "name": "ok", "path": str(seed.relative_to(tmp_path)),
        "provenance": str(pv.relative_to(tmp_path)),
        "expected_sha256": sha, "licence_keywords": ["CC0 1.0"],
    }, repo_root=tmp_path)
    assert res["issues"] == []
    assert res["actual_sha256"] == sha


def test_check_one_flags_missing_provenance(tmp_path: Path) -> None:
    mod = _load_module()
    seed = tmp_path / "x.bin"; seed.write_bytes(b"abc")
    res = mod.check_one({
        "name": "x", "path": "x.bin", "provenance": "x.PROVENANCE.txt",
        "expected_sha256": _sha256(seed), "licence_keywords": ["Anything"],
    }, repo_root=tmp_path)
    assert any("PROVENANCE.txt missing" in i for i in res["issues"])


def test_check_one_flags_sha_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    seed, pv, _ = _make_seed(tmp_path, "y.bin", b"abc",
                             "Licence: CC0 1.0 Public Domain")
    res = mod.check_one({
        "name": "y", "path": "y.bin", "provenance": "y.bin.PROVENANCE.txt",
        "expected_sha256": "0" * 64, "licence_keywords": ["CC0 1.0"],
    }, repo_root=tmp_path)
    assert any("SHA-256 mismatch" in i for i in res["issues"])


def test_check_one_flags_missing_licence_keyword(tmp_path: Path) -> None:
    mod = _load_module()
    seed, pv, sha = _make_seed(tmp_path, "z.bin", b"abc",
                               "Some text without any allowlisted keyword.")
    res = mod.check_one({
        "name": "z", "path": "z.bin", "provenance": "z.bin.PROVENANCE.txt",
        "expected_sha256": sha,
        "licence_keywords": ["MITRE ATT&CK Terms of Use", "CC0 1.0"],
    }, repo_root=tmp_path)
    assert any("no licence keyword" in i for i in res["issues"])


def test_check_one_flags_missing_seed_file(tmp_path: Path) -> None:
    mod = _load_module()
    res = mod.check_one({
        "name": "missing", "path": "nope.bin",
        "provenance": "nope.PROVENANCE.txt",
        "expected_sha256": "0" * 64, "licence_keywords": ["x"],
    }, repo_root=tmp_path)
    assert any("seed file missing" in i for i in res["issues"])


def test_real_repo_seeds_have_expected_keywords() -> None:
    """Defence-in-depth: every registered SEED in the real script must
    surface at least one of its licence_keywords in its on-disk
    PROVENANCE.txt (a regression here would reach the build gate, but
    catching it at unit-test time is faster)."""
    mod = _load_module()
    for seed in mod.SEEDS:
        pv = REPO_ROOT / seed["provenance"]
        assert pv.exists(), f"PROVENANCE.txt missing for {seed['name']}"
        text = pv.read_text()
        hits = [kw for kw in seed["licence_keywords"] if kw in text]
        assert hits, (
            f"no licence keyword from {seed['licence_keywords']} found "
            f"in {pv}; PROVENANCE.txt must declare licence")
