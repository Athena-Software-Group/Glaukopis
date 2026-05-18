# v21 — v18.1 strict reproducibility experiment

`v21` is a forked clone of the `v18.1` (Qwen2.5-14B) three-stage chain
intended to verify that the `v18.1-core` peak is bit-reproducible from
byte-identical template + gate inputs. It exists because both `v19-core`
(58.5) and `v20-core` (57.9) regressed against `v18.1-core` (58.9) under
launcher comments asserting a "byte-identical recipe", which leaves
either silent template/gate drift or data-build non-determinism as the
remaining suspect. v21 forces both candidate causes to identity vs
`v18.1` and re-benchmarks.

See [`v21_plan.txt`](v21_plan.txt) for the full §1–§6 plan, including
the §5 sign-off gate (v21 axis-by-axis result must land within ±1.5 pp
of `v18.1-core` for the recipe to be declared bit-reproducible).

## What is byte-identical vs v18.1

| Artefact (v21) | Source (v18.1) | Verified by |
|---|---|---|
| `Sophia-CTI-Templates-v21.txt`     | `05112026/Sophia-CTI-Templates-v18.1.txt` | `shasum` |
| `Sophia-CTI-Templates-v21_taa.txt` | `05092026/Sophia-CTI-Templates-v16.txt`   | `shasum` |
| `Sophia-CTI-Templates-v21_cse.txt` | `05102026/Sophia-CTI-Templates-v17.1.txt` | `shasum` |
| `v21_row_count_gate.json`          | `05112026/v18_1_row_count_gate.json`      | `shasum` |
| `v21_taa_row_count_gate.json`      | `05092026/v16_row_count_gate.json`        | `shasum` |
| `v21_cse_row_count_gate.json`      | `05102026/v17_1_row_count_gate.json`      | `shasum` |

## What changes vs v18.1

Only paperwork and naming. No `Count:`, `Shuffle:`, `--max-samples`,
`--lr`, `--cutoff`, or gate-floor change:

- Date stamp `05_11` → `05_18`, vintage dir `05112026` → `05182026`
- Dataset names `ift_data_2026_05_11_v18p1_*` → `ift_data_2026_05_18_v21_*`
- Build dirs `_v18p1{,_taa,_cse}_build/` → `_v21_{core,taa,cse}_build/`
  (v19/v20 per-stage convention; v18.1 collapsed Core into `_v18p1_build/`)
- HF push targets `…-v18-1-{core,taa,cse}` → `…-v21-{core,taa,cse}`

## How to run

```bash
# 1. Build the v21 data (one shot per stage). Same args as v18.1.
mkdir -p _v21_core_build/triples _v21_taa_build/triples _v21_cse_build/triples
# Core (count_limit=2500, count_max=1500 verbatim from v18.1 README-18-1)
nohup bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21.txt \
    _v21_core_build/triples \
    SFT/data/ift_data_2026_05_18_v21_core.raw.json \
    2500 1500 > _v21_core_build/build.log 2>&1 &
echo "PID=$!" > _v21_core_build/build.pid
nohup bash _v21_core_build/watcher.sh > _v21_core_build/watcher.log 2>&1 &
# TAA (count_limit=10, count_max=3500 verbatim from v18 README)
nohup bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_taa.txt \
    _v21_taa_build/triples \
    SFT/data/ift_data_2026_05_18_v21_taa.raw.json \
    10 3500 > _v21_taa_build/build.log 2>&1 &
echo "PID=$!" > _v21_taa_build/build.pid
nohup bash _v21_taa_build/watcher.sh > _v21_taa_build/watcher.log 2>&1 &
# CSE (count_limit=10, count_max=3500 verbatim from v18 README)
nohup bash tmpl_gen/data_generation/make_dataset.sh \
    tmpl_gen/templates/05182026/Sophia-CTI-Templates-v21_cse.txt \
    _v21_cse_build/triples \
    SFT/data/ift_data_2026_05_18_v21_cse.raw.json \
    10 3500 > _v21_cse_build/build.log 2>&1 &
echo "PID=$!" > _v21_cse_build/build.pid
nohup bash _v21_cse_build/watcher.sh > _v21_cse_build/watcher.log 2>&1 &

# 2. Train (sequential; each stage chains off the prior stage's HF push)
cd SFT/autotrain
./run_sft_qwen25_14b_v21_core.sh
./run_sft_qwen25_14b_v21_plus_taa.sh
./run_sft_qwen25_14b_v21_final.sh

# 3. Benchmark (vLLM aliases registered in SFT/test/pipelines/models.py)
cd ../test
utils/run_benchmark.sh athena-cti-sft-qwen25-14b-v21-core-vllm \
    --suite athena --version 1
```

## Provenance

Forked 2026-05-18. See [`v21_plan.txt`](v21_plan.txt) §1 for the
regression context and §5 for the falsification matrix.
