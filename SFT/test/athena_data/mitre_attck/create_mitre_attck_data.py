import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import requests
from openai import OpenAI
from tqdm import tqdm

from pipelines.data_loader import (
    load_api_key,
    load_yaml,
    parse_date,
    within_inclusive,
)

EXAMPLES = """Example 1:
attack-pattern-description: Adversaries may abuse Windows Management Instrumentation (WMI) to execute malicious commands and payloads.
detail-attack-scenario: An adversary remotely connected to a compromised workstation and used Windows Management Instrumentation (WMI) to execute a command that launched a malicious script. The script was delivered and run entirely through WMI without requiring the attacker to interact with the system's graphical interface.

Example 2:
attack-pattern-description: Adversaries may attach filters to a network socket to monitor then activate backdoors used for persistence or command and control.
detail-attack-scenario: An attacker gained access to a compromised server and attached a packet filter to a network socket to quietly observe inbound traffic. After monitoring activity for several hours, the adversary triggered a backdoor listener on the same socket to establish remote control of the system.

Example 3:
attack-pattern-description: Adversaries may use utilities to compress and/or encrypt collected data prior to exfiltration.
detail-attack-scenario: An attacker gathered sensitive project files from a compromised server and used a command-line utility to compress them into a single archive. To further secure the stolen data, the adversary applied password-based encryption to the archive before preparing it for transfer.
"""

MITRE_SRC_NAMES = ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack")

logger = logging.getLogger(__name__)


def ensure_attack_bundle(url: str, dest: Path) -> Path:
    logger.info("Downloading ATT&CK bundle from %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    dest.write_bytes(r.content)
    logger.info("Bundle saved to %s", dest)
    return dest


def external_id(obj: Dict[str, Any]) -> str:
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") in MITRE_SRC_NAMES and ref.get("external_id"):
            return ref["external_id"]
    return ""


def first_sentence(text: str, max_len: int = 400) -> str:
    text = (text or "").strip().replace("\r", " ").replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = parts[0] if parts else text
    return out[:max_len]


def normalize_platforms(obj: Dict[str, Any]) -> List[str]:
    return list(obj.get("x_mitre_platforms", []) or [])


def normalize_tactics(obj: Dict[str, Any]) -> List[str]:
    phases = obj.get("kill_chain_phases", []) or []
    return [p.get("phase_name", "") for p in phases if p.get("phase_name")]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_utc_iso(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# =========================
# GPT-5 helpers
# =========================
def gpt5_generate_scenario(
    client: OpenAI,
    model: str,
    desc: str,
    reasoning_effort: str,
    verbosity: str,
    timeout_s: int,
    max_retries: int,
    backoff_base: float,
) -> str:
    prompt = (
        "You are an expert cyber threat analyst. Create a brief 2-3 sentence of specific attack scenario "
        "based solely on the attack pattern description below."
        "Ensure the scenario reflects only this pattern and does not add extra techniques. "
        "Do not discuss ramifications or provide commentary. "
        "Return only the scenario text with no preface or postscript.\n\n"
        " You can learn from the three examples provided below:\n\n"
        f"{EXAMPLES}\n\n"
        f"attack-pattern-description: {desc}\n"
    )
    combined_input = f"[SYSTEM]\nGenerate only the scenario text.\n\n[USER]\n{prompt}"

    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=combined_input,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": verbosity},
                timeout=timeout_s,
            )
            out = (getattr(resp, "output_text", None) or "").strip()
            out = re.sub(r"^\s*(Scenario:|Example:|detail-attack-scenario:)\s*", "", out, flags=re.I)
            return out
        except Exception as e:
            transient = any(
                x in str(e)
                for x in (
                    "timed out",
                    "Timeout",
                    "429",
                    "Rate limit",
                    "502",
                    "503",
                    "504",
                    "Temporary",
                    "Connection reset",
                    "RemoteDisconnected",
                )
            )
            if attempt < max_retries and transient:
                time.sleep(backoff_base ** attempt)
                continue
            return ""


# =========================
# Core builder
# =========================
def build_scenarios(cfg: Dict[str, Any], client: OpenAI) -> Tuple[List[Dict[str, Any]], Path]:
    attack_url = cfg.get("attack_url")
    cache_path = Path(cfg.get("cache_path"))
    data_dir = Path(cfg.get("data_dir"))
    min_desc_chars = int(cfg.get("min_desc_chars", 0))
    max_items = int(cfg.get("max_items", 0))
    seed = int(cfg.get("seed", 1337))
    sleep_between = float(cfg.get("sleep_between_gpt_calls", 0.5))
    start_time = cfg.get("start_time")
    end_time = cfg.get("end_time")

    gpt_cfg = cfg.get("gpt", {})
    model = str(gpt_cfg.get("model"))
    reasoning_effort = str(gpt_cfg.get("reasoning_effort", "minimal"))
    verbosity = str(gpt_cfg.get("verbosity", "low"))
    timeout_s = int(gpt_cfg.get("timeout_s", 90))
    max_retries = int(gpt_cfg.get("max_retries", 3))
    backoff_base = float(gpt_cfg.get("backoff_base", 2.0))

    logger.info("Fetching ATT&CK data")
    bundle_path = ensure_attack_bundle(attack_url, cache_path)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objs = data.get("objects", [])

    attack_patterns: Dict[str, Dict[str, Any]] = {}
    mitigations: Dict[str, Dict[str, Any]] = {}
    for o in objs:
        t = o.get("type")
        if t not in ("attack-pattern", "course-of-action"):
            continue
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        eid = external_id(o)
        if not eid:
            continue
        if t == "attack-pattern" and eid.startswith("T"):
            attack_patterns[o["id"]] = o
        elif t == "course-of-action" and eid.startswith("M"):
            mitigations[o["id"]] = o

    tech_to_mits: Dict[str, set] = {}
    for o in objs:
        if o.get("type") != "relationship":
            continue
        if o.get("relationship_type") != "mitigates":
            continue
        src = o.get("source_ref")
        tgt = o.get("target_ref")
        if src in mitigations and tgt in attack_patterns:
            mid = external_id(mitigations[src])
            tid = external_id(attack_patterns[tgt])
            if not mid or not tid:
                continue
            tech_to_mits.setdefault(tid, set()).add(mid)

    start_dt = parse_date(start_time) if start_time else None
    end_dt = parse_date(end_time) if end_time else None
    if start_dt and start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt and end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    logger.info("Selecting candidate attack patterns")
    candidates = []
    for ap in attack_patterns.values():
        tid = external_id(ap)
        if not tid:
            continue
        ts = ap.get("modified") or ap.get("created") or ""
        if start_dt and end_dt and not within_inclusive(ts, start_dt, end_dt):
            continue
        mset = tech_to_mits.get(tid, set())
        if not mset:
            continue
        desc = ap.get("description", "") or ""
        if len(desc) < min_desc_chars:
            continue
        candidates.append((ap, tid, mset, ts))

    rng = random.Random(seed)
    if max_items and len(candidates) > max_items:
        candidates = rng.sample(candidates, max_items)

    logger.info("Generating scenarios with GPT-5 for %d candidates", len(candidates))
    rows: List[Dict[str, Any]] = []
    for ap, tid, mset, ts_raw in tqdm(candidates, desc="Scenarios"):
        desc = ap.get("description", "") or ""
        short_desc = first_sentence(desc, max_len=400)
        tname = ap.get("name", "")
        plats = normalize_platforms(ap)
        tactics = normalize_tactics(ap)
        platform_hint = plats[0] if plats else "Enterprise"
        scenario = gpt5_generate_scenario(
            client,
            model,
            desc,
            reasoning_effort,
            verbosity,
            timeout_s,
            max_retries,
            backoff_base,
        ) or short_desc
        time.sleep(sleep_between)
        gen_ts = to_utc_iso(datetime.now(timezone.utc))
        obj_ts_norm = to_utc_iso(parse_date(ts_raw)) if parse_date(ts_raw) else ts_raw
        rows.append(
            {
                "technique_id": tid,
                "technique_name": tname,
                "platform": platform_hint,
                "tactics": tactics,
                "description": short_desc,
                "scenario": scenario,
                "mitigations": sorted(list(mset)),
                "timestamp": obj_ts_norm,
                "metadata": {
                    "generated": gen_ts,
                    "attack_bundle_source": str(bundle_path),
                },
            }
        )

    today = datetime.now(timezone.utc).date().isoformat()
    out_path = data_dir / f"mitre_attck_{today}.jsonl"
    write_jsonl(out_path, rows)
    logger.info("Wrote %d scenario records -> %s", len(rows), out_path)
    return rows, out_path


def build_rms_tasks(records: List[Dict[str, Any]], prompt_template: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    k = int(cfg.get("num_questions", 0))
    seed = int(cfg.get("seed", 1337))
    rng = random.Random(seed)
    pool = records[:]
    if k and len(pool) > k:
        pool = rng.sample(pool, k)
    logger.info("Building RMS tasks")
    tasks = []
    for r in tqdm(pool, desc="RMS"):
        env = r.get("platform", "Enterprise")
        mits = r.get("mitigations", [])
        count = len(mits)
        scenario = r.get("scenario", "")
        tasks.append(
            {
                "technique_id": r.get("technique_id"),
                "description": r.get("description"),
                "scenario": scenario,
                "prompt": prompt_template.format(env, count, scenario),
                "answer": ", ".join(mits),
            }
        )
    return tasks


def build_ate_tasks(records: List[Dict[str, Any]], prompt_template: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    k = int(cfg.get("num_questions", 0))
    seed = int(cfg.get("seed", 1337))
    rng = random.Random(seed)
    pool = records[:]
    if k and len(pool) > k:
        pool = rng.sample(pool, k)
    logger.info("Building ATE tasks")
    tasks = []
    for r in tqdm(pool, desc="ATE"):
        env = r.get("platform", "Enterprise")
        scenario = r.get("scenario", "")
        tid = r.get("technique_id")
        tasks.append(
            {
                "technique_id": tid,
                "description": r.get("description"),
                "scenario": scenario,
                "prompt": prompt_template.format(env, scenario),
                "answer": tid,
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MITRE ATT&CK scenario data and benchmarks")
    parser.add_argument("--config", default="athena_data/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # Quiet noisy libraries
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        log = logging.getLogger(name)
        log.setLevel(logging.WARNING)  # or logging.ERROR for total silence
        log.propagate = False

    logger.info("Loading config from %s", args.config)
    cfg = load_yaml(args.config)
    prompts = load_yaml(cfg["COMMON"]["task_prompt_file"])

    api_key = load_api_key("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    mitre_cfg = cfg.get("MITRE_ATTCK", {})
    logger.info("Building scenarios")
    records, _ = build_scenarios(mitre_cfg, client)

    date_tag = datetime.now().strftime("%Y%m%d")
    rms_prompt = prompts["RMS"]
    rms_cfg = cfg.get("RMS", {})
    rms_tasks = build_rms_tasks(records, rms_prompt, rms_cfg)
    rms_out = Path(rms_cfg.get("output_path", "benchmark_data/athena_bench/athena-cti-rms.jsonl"))
    rms_out = rms_out.with_name(f"{rms_out.stem}_{date_tag}{rms_out.suffix}")
    if rms_out:
        write_jsonl(Path(rms_out), rms_tasks)
        logger.info("Wrote %d RMS tasks -> %s", len(rms_tasks), rms_out)

    ate_prompt = prompts["ATE"]
    ate_cfg = cfg.get("ATE", {})
    ate_tasks = build_ate_tasks(records, ate_prompt, ate_cfg)
    ate_out = Path(ate_cfg.get("output_path", "benchmark_data/athena_bench/athena-cti-ate.jsonl"))
    ate_out = ate_out.with_name(f"{ate_out.stem}_{date_tag}{ate_out.suffix}")
    if ate_out:
        write_jsonl(Path(ate_out), ate_tasks)
        logger.info("Wrote %d ATE tasks -> %s", len(ate_tasks), ate_out)


if __name__ == "__main__":
    main()
