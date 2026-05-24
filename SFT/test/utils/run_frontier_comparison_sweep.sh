#!/bin/bash

# Frontier model comparison sweep: runs four hosted models against the
# same CyberMetric (2000 + 10000), CyberSOCEval (malware + threat-intel),
# and MMLU-Pro task set back-to-back. Mirrors the gpt5.5 sweep scope so
# the resulting rows in the per-model summary tables line up directly.
#
# Pure HTTP (OpenAI / Gemini / HuggingFace Inference Providers) — no
# local GPU or vLLM is touched, so this is safe to run on the same host
# as an active SFT training job (only CPU + RAM footprint).
#
# Models exercised (sequential to avoid shared OpenAI/Gemini quota
# contention and to keep logs readable):
#   1. gemini-3-flash            -> gemini-3-flash-preview
#   2. deepseek-v4-pro-hf        -> deepseek-ai/DeepSeek-V4-Pro (HF Router)
#   3. gpt5.2 --reasoning-effort high  -> gpt-5.2 (high effort)
#   4. deepseek-v3.2-exp-hf      -> deepseek-ai/DeepSeek-V3.2-Exp (HF Router)
#
# Per-model task chain (also sequential per task to free conn pools and
# isolate per-task summary writes):
#   --suite cybermetric --cybermetric-size 2000,10000
#   --suite cybersoceval         (malware + threat-intel)
#   --suite mmlu-pro
#
# Usage:
#   conda activate ctibench
#   nohup bash run_frontier_comparison_sweep.sh > frontier_sweep_$(date +%s).log 2>&1 &
#
# Flags:
#   --rows N             cap each task at N rows (smoke-test override)
#   --batch N            in-model concurrency (default 16; HF Router can
#                        be unstable above 16 for the deepseek family)
#   --no-overwrite       resume mode; existing rows are preserved
#   --models "a b c"     override the model list (space-separated aliases;
#                        loses per-model reasoning-effort defaults — pass
#                        --reasoning-effort separately if needed)
#   --skip-cybermetric   drop the CyberMetric step (per model)
#   --skip-cybersoceval  drop the CyberSOCEval step (per model)
#   --skip-mmlu-pro      drop the MMLU-Pro step (per model)
#   --dry-run            print the run_benchmark.sh invocations only
#
# Environment:
#   OPENAI_API_KEY       required when the list contains any gpt* alias
#   GEMINI_API_KEY       required when the list contains any gemini* alias
#   HUGGINGFACE_TOKEN    required when the list contains any *-hf alias
#   (SFT/.env is auto-sourced by _load_dotenv.sh so the canonical bench
#   location is honored without a manual export.)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_BENCH="${SCRIPT_DIR}/run_benchmark.sh"

# shellcheck source=_load_dotenv.sh
source "${SCRIPT_DIR}/_load_dotenv.sh"

ROWS=""
BATCH="16"
OVERWRITE=1
DRY_RUN=0
MODELS_OVERRIDE=""
SKIP_CYBERMETRIC=0
SKIP_CYBERSOCEVAL=0
SKIP_MMLU_PRO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows)              ROWS="$2"; shift 2 ;;
        --batch)             BATCH="$2"; shift 2 ;;
        --no-overwrite)      OVERWRITE=0; shift ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --models)            MODELS_OVERRIDE="$2"; shift 2 ;;
        --skip-cybermetric)  SKIP_CYBERMETRIC=1; shift ;;
        --skip-cybersoceval) SKIP_CYBERSOCEVAL=1; shift ;;
        --skip-mmlu-pro)     SKIP_MMLU_PRO=1; shift ;;
        -h|--help)           sed -n '3,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Per-row recipe: "<alias>|<reasoning-effort or empty>". gpt5.2 is the
# only entry that ships a non-empty effort by default (the user asked
# for the high-effort variant for the frontier comparison).
declare -a RUNS=(
    "gemini-3-flash|"
    "deepseek-v4-pro-hf|"
    "gpt5.2|high"
    "deepseek-v3.2-exp-hf|"
)
if [[ -n "${MODELS_OVERRIDE}" ]]; then
    RUNS=()
    for m in ${MODELS_OVERRIDE}; do RUNS+=("${m}|"); done
fi

# Pre-flight: only require keys for families actually present in RUNS.
need_openai=0; need_gemini=0; need_hf=0
for spec in "${RUNS[@]}"; do
    alias_only="${spec%%|*}"
    case "${alias_only}" in
        gpt*)        need_openai=1 ;;
        gemini*)     need_gemini=1 ;;
        *-hf)        need_hf=1 ;;
    esac
done
missing=()
[[ ${need_openai} -eq 1 && -z "${OPENAI_API_KEY:-}"    ]] && missing+=("OPENAI_API_KEY")
[[ ${need_gemini} -eq 1 && -z "${GEMINI_API_KEY:-}"    ]] && missing+=("GEMINI_API_KEY")
[[ ${need_hf}     -eq 1 && -z "${HUGGINGFACE_TOKEN:-}" ]] && missing+=("HUGGINGFACE_TOKEN")
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[FAIL] missing env var(s): ${missing[*]}" >&2
    echo "       Export them or add to SFT/.env." >&2
    exit 2
fi

stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="${SCRIPT_DIR}/frontier_sweep_${stamp}.log"

echo "=== Frontier comparison sweep (${stamp}) ==="
echo "  models : ${RUNS[*]}"
echo "  rows   : ${ROWS:-all}"
echo "  batch  : ${BATCH}"
echo "  ovwt   : ${OVERWRITE}"
echo "  log    : ${LOG_FILE}"
echo

cd "${BENCH_DIR}" || { echo "[FAIL] cd ${BENCH_DIR}" >&2; exit 2; }

overall=0
for spec in "${RUNS[@]}"; do
    IFS='|' read -r alias effort <<< "${spec}"
    label="${alias}${effort:+ (effort=${effort})}"
    echo "########## MODEL: ${label} ##########" | tee -a "${LOG_FILE}"

    common_args=(--version 1 --batch "${BATCH}")
    [[ -n "${ROWS}"      ]] && common_args+=(--rows "${ROWS}")
    [[ ${OVERWRITE} -eq 1 ]] && common_args+=(--overwrite --yes)
    [[ -n "${effort}"    ]] && common_args+=(--reasoning-effort "${effort}")

    # tag:invocation pairs, evaluated in order. Skipping a step here
    # drops the row entirely so the rc[] map stays compact.
    declare -a tasks=()
    [[ ${SKIP_CYBERMETRIC}  -eq 0 ]] && tasks+=("cybermetric:--suite cybermetric --cybermetric-size 2000,10000")
    [[ ${SKIP_CYBERSOCEVAL} -eq 0 ]] && tasks+=("cybersoceval:--suite cybersoceval")
    [[ ${SKIP_MMLU_PRO}     -eq 0 ]] && tasks+=("mmlu-pro:--suite mmlu-pro")

    for entry in "${tasks[@]}"; do
        IFS=':' read -r tag suite_args <<< "${entry}"
        echo "----- ${alias} / ${tag} -----" | tee -a "${LOG_FILE}"
        echo "  cmd: bash ${RUN_BENCH} ${alias} ${suite_args} ${common_args[*]}" | tee -a "${LOG_FILE}"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "  [dry-run] not invoking" | tee -a "${LOG_FILE}"
            continue
        fi
        set +e
        # suite_args intentionally unquoted: contains multiple tokens
        # (e.g. "--suite cybermetric --cybermetric-size 2000,10000")
        # that must word-split into separate argv entries.
        bash "${RUN_BENCH}" "${alias}" ${suite_args} "${common_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
        rc=${PIPESTATUS[0]}
        set -e
        echo "  exit: ${rc}" | tee -a "${LOG_FILE}"
        [[ ${rc} -ne 0 ]] && overall=${rc}
    done
    echo | tee -a "${LOG_FILE}"
done

echo "=== sweep complete (overall exit ${overall}) ===" | tee -a "${LOG_FILE}"
exit ${overall}
