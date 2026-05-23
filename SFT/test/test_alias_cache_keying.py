#!/usr/bin/env python3
# Alias-keyed response cache regression check. Run from repo root:
#     python SFT/test/test_alias_cache_keying.py
# Exits 0 on success, 1 on any failure. No pytest dependency.
#
# Covers the systemic fix in 18e3430 that decoupled the on-disk cache
# slot from the HF repo id. Two failure modes this guards against:
#   1. Drift between pipelines.models.alias_to_safe_name and the bash
#      ${MODEL_NAME//\//_} convention in run_benchmark.sh.
#   2. Regression of the migration script's ambiguity detection (an HF
#      repo with >1 alias must NOT be auto-renamed).

import ast
import importlib.util
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "SFT" / "test"
MODELS_PY = BENCH / "pipelines" / "models.py"
RUN_BENCH_SH = BENCH / "utils" / "run_benchmark.sh"
MIGRATE_PY = BENCH / "utils" / "migrate_response_cache_to_alias_keyed.py"

PASSED, FAILED = [], []


def check(label, ok, detail=""):
    (PASSED if ok else FAILED).append(label)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))


def _extract_func_source(py_path, fname):
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return ast.get_source_segment(py_path.read_text(encoding="utf-8"), node)
    raise RuntimeError(f"{fname} not found in {py_path}")


def _load_migrate_module():
    spec = importlib.util.spec_from_file_location("migrate_cache", MIGRATE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1. alias_to_safe_name contract (extract via AST to avoid the torch import).
print("\n1. pipelines.models.alias_to_safe_name contract")
src = _extract_func_source(MODELS_PY, "alias_to_safe_name")
ns = {}
exec(src, ns)
fn = ns["alias_to_safe_name"]
cases = [
    ("qwen3-30b-a3b-thinking-2507-vllm", "qwen3-30b-a3b-thinking-2507-vllm"),
    ("qwen3-30b-a3b-thinking-2507-no-think-vllm",
     "qwen3-30b-a3b-thinking-2507-no-think-vllm"),
    ("athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm",
     "athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm"),
    ("Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen_Qwen3-30B-A3B-Thinking-2507"),
    ("asg-ai/athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse",
     "asg-ai_athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse"),
    ("gpt5", "gpt5"),
]
for inp, want in cases:
    got = fn(inp)
    check(f"alias_to_safe_name({inp!r}) == {want!r}", got == want, f"got {got!r}")

# 2. Shell convention in run_benchmark.sh must match the Python helper.
print("\n2. shell <-> python sync (run_benchmark.sh)")
sh_src = RUN_BENCH_SH.read_text(encoding="utf-8")
check("SAFE_NAME=${MODEL_NAME//\\//_} present in run_benchmark.sh",
      'SAFE_NAME="${MODEL_NAME//\\//_}"' in sh_src)
check("resolve_resp_file keys off ${SAFE_NAME}",
      'responses/${SAFE_NAME}/' in sh_src,
      "cache base path must use SAFE_NAME, not DISPLAY_NAME")
check("filename component uses ${SAFE_NAME}_response",
      '_${SAFE_NAME}_response.' in sh_src,
      "embedded filename component must use SAFE_NAME")

# 3. Migration script helpers.
print("\n3. migrate_response_cache_to_alias_keyed helpers")
mig = _load_migrate_module()
check("_alias_safe matches alias_to_safe_name",
      mig._alias_safe("Qwen/Foo") == fn("Qwen/Foo") == "Qwen_Foo")
check("_hf_safe replaces only '/'", mig._hf_safe("a/b") == "a_b"
      and mig._hf_safe("a-b") == "a-b")

fixture_mapping = {
    "qwen3-30b-a3b-thinking-2507-vllm":          "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-30b-a3b-thinking-2507-no-think-vllm": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "deephat-7b":                                "DeepHat/DeepHat-V1-7B",
    "gpt5":                                      "gpt-5",
}
idx = mig._build_reverse_index(fixture_mapping)
check("reverse index flags ambiguous HF repo",
      sorted(idx["Qwen_Qwen3-30B-A3B-Thinking-2507"]) == [
          "qwen3-30b-a3b-thinking-2507-no-think-vllm",
          "qwen3-30b-a3b-thinking-2507-vllm",
      ])
check("reverse index records unambiguous HF repo as single-alias list",
      idx["DeepHat_DeepHat-V1-7B"] == ["deephat-7b"])
check("reverse index records api-style ids verbatim (no slash)",
      idx["gpt-5"] == ["gpt5"])

# 4. End-to-end migration on a temp fixture: unambiguous renames, ambiguous skips.
print("\n4. migration end-to-end (temp fixture)")
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    # Unambiguous: DeepHat_DeepHat-V1-7B/ -> deephat-7b/
    src_unamb = root / "DeepHat_DeepHat-V1-7B" / "athena-mcq"
    src_unamb.mkdir(parents=True)
    f1 = src_unamb / "athena-mcq_all_v1_DeepHat_DeepHat-V1-7B_response.jsonl"
    f1.write_text("{}\n")
    # Ambiguous: Qwen_Qwen3-30B-A3B-Thinking-2507/ (2 aliases) -- must stay.
    src_amb = root / "Qwen_Qwen3-30B-A3B-Thinking-2507" / "mmlu-pro"
    src_amb.mkdir(parents=True)
    f2 = src_amb / "mmlu-pro_all_v1_Qwen_Qwen3-30B-A3B-Thinking-2507_response.csv"
    f2.write_text("h\n")

    fake_models = root / "models.py"
    fake_models.write_text(f"model_mapping = {fixture_mapping!r}\n")
    rc = mig.main(["--responses-root", str(root),
                   "--models-py", str(fake_models), "--apply"])
    check("migrate.main returns 0", rc == 0)
    check("unambiguous dir renamed to alias", (root / "deephat-7b").is_dir())
    check("unambiguous source dir removed",
          not (root / "DeepHat_DeepHat-V1-7B").exists())
    renamed = root / "deephat-7b" / "athena-mcq" / \
        "athena-mcq_all_v1_deephat-7b_response.jsonl"
    check("embedded filename component rewritten to alias",
          renamed.is_file(), f"expected {renamed}")
    check("ambiguous dir left in place", src_amb.parent.is_dir())
    check("ambiguous file left in place (untouched)", f2.is_file())

# 5. Spot-check a benchmark file actually imports + uses the helper.
print("\n5. benchmark integration spot-checks")
for bench_file in ["mmlu_pro.py", "athena_mcq.py", "cybermetric.py",
                   "cti_mcq.py", "cybersoceval_malware.py"]:
    text = (BENCH / "benchmarks" / bench_file).read_text(encoding="utf-8")
    check(f"{bench_file} imports alias_to_safe_name",
          "alias_to_safe_name" in text)
    check(f"{bench_file} does NOT use model_mapping for display name",
          "model_mapping.get(model_name" not in text
          and "model_mapping[model_name]" not in text)

print(f"\n{'=' * 60}")
print(f"PASS: {len(PASSED)}   FAIL: {len(FAILED)}")
if FAILED:
    print("Failed checks:")
    for label in FAILED:
        print(f"  - {label}")
    sys.exit(1)
print("All alias-cache-keying checks passed.")
sys.exit(0)
