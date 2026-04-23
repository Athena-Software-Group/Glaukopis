#!/usr/bin/env python3
"""Per-format parsers that turn raw/ artifacts into normalized {"text", "meta"} docs.

Dispatches on the `parser` key of each source entry. Each parser takes a
list of raw file paths and yields dicts:

    {"text": str, "meta": {"source": str, "id": str, ...}}

Parsers are defensive: bad records are logged and skipped rather than
raising, so a single corrupt NVD entry or malformed HTML doesn't abort
the corpus build. Parsed output is written as JSONL to
cache/parsed/<source>.jsonl by build_corpus.py.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator
from xml.etree import ElementTree as ET


# ---------- STIX (MITRE ATT&CK, CAPEC) ----------

def _stix_objects(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    for obj in bundle.get("objects", []):
        yield obj


def parse_stix(paths: list[Path], source: str) -> Iterator[dict]:
    """Render ATT&CK/CAPEC STIX objects as self-contained natural-language docs.

    Each technique/group/software/mitigation becomes one doc. The doc
    embeds the external reference id (e.g. T1059.001, G0016, M1026) so
    the leak filter in process.py can detect benchmark overlap.
    """
    for path in paths:
        try:
            for obj in _stix_objects(path):
                text, meta_id = _stix_render(obj)
                if not text:
                    continue
                # STIX dates are ISO 8601; 'modified' is the last semantic
                # change to the object, which is the best proxy for "when
                # did this entry reach its current content" for CPT freshness.
                date = (obj.get("modified") or obj.get("created") or "")[:10]
                yield {
                    "text": text,
                    "meta": {
                        "source": source,
                        "id": meta_id,
                        "type": obj.get("type", ""),
                        "date": date,
                    },
                }
        except Exception as e:  # noqa: BLE001
            print(f"[parse:stix] {path}: {e}", file=sys.stderr)


_STIX_INTERESTING = {
    "attack-pattern", "intrusion-set", "malware", "tool",
    "course-of-action", "campaign", "threat-actor",
}


def _external_id(obj: dict) -> str:
    for ref in obj.get("external_references", []) or []:
        src = (ref.get("source_name") or "").lower()
        if src in {"mitre-attack", "mitre-mobile-attack", "mitre-ics-attack", "capec"}:
            return ref.get("external_id") or ""
    return ""


def _stix_render(obj: dict) -> tuple[str, str]:
    if obj.get("revoked") or obj.get("x_mitre_deprecated"):
        return "", ""
    otype = obj.get("type", "")
    if otype not in _STIX_INTERESTING:
        return "", ""
    name = (obj.get("name") or "").strip()
    desc = (obj.get("description") or "").strip()
    if not name or not desc:
        return "", ""
    ext_id = _external_id(obj)
    aliases = obj.get("aliases") or obj.get("x_mitre_aliases") or []
    platforms = obj.get("x_mitre_platforms") or []
    tactics = [
        (p.get("phase_name") or "").replace("-", " ")
        for p in obj.get("kill_chain_phases", []) or []
        if (p.get("kill_chain_name") or "").startswith("mitre-")
    ]
    header_bits = [f"{ext_id} {name}" if ext_id else name]
    if aliases:
        header_bits.append(f"Aliases: {', '.join(a for a in aliases if a and a != name)}")
    if tactics:
        header_bits.append(f"Tactics: {', '.join(tactics)}")
    if platforms:
        header_bits.append(f"Platforms: {', '.join(platforms)}")
    header = " | ".join(header_bits)
    return f"{header}\n\n{desc}\n", ext_id or name


# ---------- NVD CVE ----------

def parse_cve_nvd(paths: list[Path], source: str) -> Iterator[dict]:
    """Render NVD 2.0 feed CVE records as short docs: id, desc, CWE, CVSS, refs."""
    for path in paths:
        try:
            data = _load_json_maybe_gz(path)
        except Exception as e:  # noqa: BLE001
            print(f"[parse:cve_nvd] {path}: {e}", file=sys.stderr)
            continue
        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve") or {}
            cid = cve.get("id") or ""
            if not cid:
                continue
            desc = ""
            for d in cve.get("descriptions", []) or []:
                if d.get("lang") == "en":
                    desc = (d.get("value") or "").strip()
                    break
            if not desc:
                continue
            cwes = []
            for w in cve.get("weaknesses", []) or []:
                for dd in w.get("description", []) or []:
                    if dd.get("lang") == "en":
                        cwe = (dd.get("value") or "").strip()
                        if cwe and cwe not in cwes:
                            cwes.append(cwe)
            cvss_parts = []
            for k in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                for m in (cve.get("metrics") or {}).get(k, []) or []:
                    cd = m.get("cvssData") or {}
                    sev = cd.get("baseSeverity") or ""
                    score = cd.get("baseScore")
                    vec = cd.get("vectorString") or ""
                    if score is not None:
                        cvss_parts.append(f"{k} {sev} {score} {vec}".strip())
                    break  # one metric per version
            header = f"{cid}"
            if cwes:
                header += f" | {', '.join(cwes)}"
            if cvss_parts:
                header += f" | {'; '.join(cvss_parts)}"
            date = (cve.get("published") or cve.get("lastModified") or "")[:10]
            yield {
                "text": f"{header}\n\n{desc}\n",
                "meta": {"source": source, "id": cid, "type": "cve", "date": date},
            }


def _load_json_maybe_gz(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- CISA KEV ----------

def parse_kev(paths: list[Path], source: str) -> Iterator[dict]:
    for path in paths:
        try:
            data = _load_json_maybe_gz(path)
        except Exception as e:  # noqa: BLE001
            print(f"[parse:kev] {path}: {e}", file=sys.stderr)
            continue
        for v in data.get("vulnerabilities", []):
            cve = v.get("cveID") or ""
            name = v.get("vulnerabilityName") or ""
            vendor = v.get("vendorProject") or ""
            product = v.get("product") or ""
            desc = v.get("shortDescription") or ""
            action = v.get("requiredAction") or ""
            ransomware = v.get("knownRansomwareCampaignUse") or ""
            if not cve or not desc:
                continue
            text = (
                f"{cve} | KEV: {vendor} {product} -- {name}\n\n"
                f"{desc}\n\nRequired mitigation: {action}\n"
                f"Known ransomware association: {ransomware}\n"
            )
            date = (v.get("dateAdded") or "")[:10]
            yield {
                "text": text,
                "meta": {"source": source, "id": cve, "type": "kev", "date": date},
            }


# ---------- CWE XML ----------

def parse_cwe_xml(paths: list[Path], source: str) -> Iterator[dict]:
    ns = {"cwe": "http://cwe.mitre.org/cwe-7"}
    for path in paths:
        if path.suffix != ".xml":
            continue
        try:
            tree = ET.parse(path)
        except Exception as e:  # noqa: BLE001
            print(f"[parse:cwe_xml] {path}: {e}", file=sys.stderr)
            continue
        root = tree.getroot()
        for w in root.findall(".//cwe:Weakness", ns):
            wid = w.get("ID") or ""
            name = w.get("Name") or ""
            desc_el = w.find("cwe:Description", ns)
            ext_el = w.find("cwe:Extended_Description", ns)
            desc = (desc_el.text or "").strip() if desc_el is not None else ""
            ext = _flatten(ext_el) if ext_el is not None else ""
            if not wid or not name or not desc:
                continue
            body = desc + (f"\n\n{ext}" if ext else "")
            yield {
                "text": f"CWE-{wid} {name}\n\n{body}\n",
                "meta": {"source": source, "id": f"CWE-{wid}", "type": "cwe"},
            }


def _flatten(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


# ---------- HTML (CISA advisories, blogs, DFIR Report) ----------

def parse_html_trafilatura(paths: list[Path], source: str) -> Iterator[dict]:
    try:
        import trafilatura
    except ImportError:
        print("[parse:html] trafilatura not installed; run pip install -r cpt/requirements.txt", file=sys.stderr)
        return
    for path in paths:
        try:
            html = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            print(f"[parse:html] read {path}: {e}", file=sys.stderr)
            continue
        text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=False)
        if not text or len(text) < 400:
            continue
        yield {"text": text + "\n", "meta": {"source": source, "id": path.stem, "type": "html"}}


# ---------- PDF (vendor threat reports) ----------

def parse_pdf_pymupdf(paths: list[Path], source: str) -> Iterator[dict]:
    try:
        import pymupdf  # type: ignore
    except ImportError:
        print("[parse:pdf] pymupdf not installed; run pip install -r cpt/requirements.txt", file=sys.stderr)
        return
    for path in paths:
        if path.suffix.lower() != ".pdf":
            continue
        try:
            doc = pymupdf.open(path)
            chunks = [page.get_text("text") for page in doc]
            doc.close()
        except Exception as e:  # noqa: BLE001
            print(f"[parse:pdf] {path}: {e}", file=sys.stderr)
            continue
        text = "\n".join(c for c in chunks if c and c.strip())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 800:
            continue
        yield {"text": text + "\n", "meta": {"source": source, "id": path.stem, "type": "pdf"}}


# ---------- Sigma YAML ----------

def parse_sigma_yaml(paths: list[Path], source: str) -> Iterator[dict]:
    import yaml
    # For git-cloned sources we receive a single repo dir; walk to find .yml files.
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.yml")))
        elif p.suffix in {".yml", ".yaml"}:
            files.append(p)
    for f in files:
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        title = (doc.get("title") or "").strip()
        desc = (doc.get("description") or "").strip()
        if not title or not desc:
            continue
        tags = ", ".join(doc.get("tags") or []) or ""
        logsrc = doc.get("logsource") or {}
        logsrc_s = ", ".join(f"{k}={v}" for k, v in logsrc.items() if v)
        body = f"Sigma rule: {title}\nLog source: {logsrc_s}\nTags: {tags}\n\n{desc}\n"
        yield {"text": body, "meta": {"source": source, "id": doc.get("id") or title, "type": "sigma"}}


# ---------- dispatch ----------

DISPATCH = {
    "stix": parse_stix,
    "cve_nvd": parse_cve_nvd,
    "kev": parse_kev,
    "cwe_xml": parse_cwe_xml,
    "capec_xml": parse_stix,  # CAPEC is served as STIX here; XML variant can be added later
    "html_trafilatura": parse_html_trafilatura,
    "pdf_pymupdf": parse_pdf_pymupdf,
    "sigma_yaml": parse_sigma_yaml,
}


def parse_source(name: str, spec: dict[str, Any], paths: list[Path]) -> Iterator[dict]:
    parser = spec.get("parser")
    if parser not in DISPATCH:
        raise ValueError(f"Unknown parser '{parser}' for source '{name}'")
    yield from DISPATCH[parser](paths, name)
