#!/bin/bash

# SSH-resilient wrapper around run_sft_qwen3_30b_a3b_thinking_v21_chain.sh.
# Launches the multi-stage chain (TAA -> CSE -> Recalibrate, optionally
# --include-core) under `nohup setsid` with stdin closed and stdout/stderr
# redirected to a timestamped log under SFT/logs/, so the run survives an
# SSH disconnect (no SIGHUP delivered) and detaches from the controlling
# terminal (no SIGTTIN/SIGTTOU on background tty reads). PID is written
# to a sibling .pid file so the whole torchrun tree can be killed cleanly
# via the recorded process group.
#
# This is the multi-stage analogue of run_sft_qwen3_30b_a3b_thinking_v21_recal_32b.nohup.sh;
# the chain orchestrator itself shells out to each stage launcher, but
# wrapping the top-level orchestrator in nohup/setsid is sufficient to
# keep every torchrun tree alive across SSH disconnects (each stage
# launcher inherits the detached process group).
#
# All flags after this script's own (--log-dir / --log-file / --no-tail)
# are forwarded verbatim to the inner chain (e.g. --start-stage cse,
# --include-core, --offload, --skip-eval, --dry-run). Caller must
# already have the train conda env active (e.g. `conda activate llm-sft`)
# -- the wrapper does not source conda.
#
# Usage:
#   conda activate llm-sft
#   bash SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_chain.nohup.sh
#   # ... or with chain passthrough flags:
#   bash SFT/autotrain/run_sft_qwen3_30b_a3b_thinking_v21_chain.nohup.sh \
#       --include-core --report-to wandb
#
# Options (consumed by this wrapper; everything else is forwarded):
#   --log-dir DIR    override SFT/logs as the log destination
#   --log-file PATH  override the full log file path (implies --log-dir's parent)
#   --no-tail        don't `tail -F` the log after launch; just print PID + paths

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INNER="${SCRIPT_DIR}/run_sft_qwen3_30b_a3b_thinking_v21_chain.sh"

if [[ ! -x "${INNER}" ]]; then
    if [[ -f "${INNER}" ]]; then
        chmod +x "${INNER}"
    else
        echo "[FAIL] inner launcher missing: ${INNER}" >&2
        exit 2
    fi
fi

LOG_DIR="${SFT_DIR}/logs"
LOG_FILE=""
TAIL_AFTER=1
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --log-file)  LOG_FILE="$2"; LOG_DIR="$(dirname "$2")"; shift 2 ;;
        --no-tail)   TAIL_AFTER=0; shift ;;
        -h|--help)   sed -n '3,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           PASSTHROUGH+=("$1"); shift ;;
    esac
done

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE="${LOG_DIR}/sft_qwen3_30b_a3b_thinking_2507_v21_chain_${TIMESTAMP}.log"
fi
PID_FILE="${LOG_FILE%.log}.pid"

# setsid puts the inner bash into a fresh session + process group so SSH
# disconnect cannot deliver SIGHUP to it (nohup also masks SIGHUP, belt &
# suspenders). stdin redirected from /dev/null so torchrun / DeepSpeed
# can never block on a tty read. disown drops the job from this shell's
# job table so an immediate `exit` from the login shell doesn't tear it
# down via the shell's hangup hook. Recording the PID and the process
# group id (PGID) separately because `kill -- -PGID` is the clean
# teardown path for a torchrun tree (kills all worker ranks at once).
echo "=== launching v21 chain under nohup setsid ==="
echo "  inner script : ${INNER}"
echo "  forwarded    : ${PASSTHROUGH[*]:-<none>}"
echo "  log file     : ${LOG_FILE}"
echo "  pid file     : ${PID_FILE}"
echo

nohup setsid bash "${INNER}" "${PASSTHROUGH[@]}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
INNER_PID=$!
disown "${INNER_PID}" 2>/dev/null || true

# setsid makes the inner bash a session leader, so its PGID == its PID.
# Record both for clarity (kill -- -${INNER_PID} == `pkill -g ${INNER_PID}`).
PGID="${INNER_PID}"
cat > "${PID_FILE}" <<EOF
PID=${INNER_PID}
PGID=${PGID}
LOG=${LOG_FILE}
STARTED=${TIMESTAMP}
INNER=${INNER}
ARGS=${PASSTHROUGH[*]:-}
EOF

# Brief grace for the inner bash to print its banner so the user sees
# the first lines before the wrapper exits or attaches `tail`.
sleep 2

if ! kill -0 "${INNER_PID}" 2>/dev/null; then
    echo "[FAIL] inner process exited immediately. Tail of log:" >&2
    tail -n 40 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo "=== detached; v21 chain running as PID ${INNER_PID} (PGID ${PGID}) ==="
echo
echo "  follow live  :  tail -F ${LOG_FILE}"
echo "  check alive  :  kill -0 ${INNER_PID} && echo running || echo dead"
echo "  graceful stop:  kill -TERM -- -${PGID}     # whole chain + active stage's torchrun tree"
echo "  hard stop    :  kill -KILL -- -${PGID}"
echo "  pid record   :  ${PID_FILE}"
echo

if [[ ${TAIL_AFTER} -eq 1 ]]; then
    echo "=== attaching tail -F (Ctrl-C detaches; the chain keeps running) ==="
    echo
    # `exec` so the wrapper's pid becomes tail; Ctrl-C only kills tail,
    # not the disowned setsid'd bash. -F handles log rotation if anything
    # ever rotates it (LF / DeepSpeed don't, but cheap safety).
    exec tail -F "${LOG_FILE}"
fi
