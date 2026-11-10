#!/bin/bash
# Cross-model single-pass comparison: score both sides of each comparison on the same stack as the 9 Nov round.
# Usage: score_queue_ports.sh GROUP [GROUP...]
# A finished run leaves ~/logs/port_score_<name>.done and a rerun skips it. Any failure stops the queue.
source /root/miniconda3/bin/activate vllm
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
export BENCH_CONDA_ENV=ctibench
cd eval/utils || exit 1
Q=~/logs/port_score_queue.log
WRAP=serve_and_bench_v21_single_stage_port.sh
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > ~/logs/port_score_queue.status; exit 1; }

# The previous run's server must have released the GPUs before the next one loads.
gpus_free() {
    local i used
    for i in $(seq 1 40); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "${used:-99999}" -lt 2000 ] && return 0
        sleep 3
    done
    return 1
}

# Get the weights first, with a timeout and retries. A stalled Hub download otherwise uses up the server's start limit.
fetch() {
    local repo=$1 i
    for i in 1 2 3 4 5 6; do
        timeout 600 python ~/prefetch.py "$repo" >> ~/logs/port_score_prefetch.log 2>&1 && return 0
        echo "$(stamp) retry download $repo after attempt $i" >> "$Q"
    done
    return 1
}

score() {
    local name=$1 stem=$2 check=$3 alias="$2-vllm"
    if [ -f ~/logs/port_score_$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; return 0; fi
    fetch "asg-ai/$stem" || stop "$name download"
    gpus_free || stop "$name gpus not free"
    echo "$(stamp) START $name" >> "$Q"
    MODEL_ALIAS="$alias" bash "$WRAP" > ~/logs/port_score_$name.log 2>&1
    local rc=$?
    echo "$(stamp) END $name rc=$rc" >> "$Q"
    [ $rc -eq 0 ] || stop "$name rc=$rc"
    python ~/check_scores.py "$alias" $check >> "$Q" 2>&1 || stop "$name completeness"
    touch ~/logs/port_score_$name.done
    echo "$(stamp) DONE $name" >> "$Q"
}

D=$(date -u +%Y%m%d)
R=~/Glaukopis/SFT/eval/responses
[ -f $R/eval_stack_${D}_vllm.txt ] || pip freeze > $R/eval_stack_${D}_vllm.txt 2>/dev/null
[ -f $R/eval_stack_${D}_ctibench.txt ] || conda run -n ctibench pip freeze > $R/eval_stack_${D}_ctibench.txt 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/logs/port_score_hardware_${D}.txt

C=athena-cti-sft
Q3=$C-qwen3-30b-a3b-thinking-2507-v21
for group in "$@"; do
    case "$group" in
        foundation_8b)
            score foundation_layered $C-foundation-8b-v21-recalibrate ""
            score foundation_single  $C-foundation-8b-v21-single-stage "" ;;
        llama31_8b)
            score llama_layered $C-llama31-8b-v21-recalibrate ""
            score llama_single  $C-llama31-8b-v21-single-stage "" ;;
        qwen25_14b_layered)
            score qwen14_layered $C-qwen25-14b-v21-recalibrate "" ;;
        qwen25_14b_single)
            score qwen14_single $C-qwen25-14b-v21-single-stage "" ;;
        qwen3_layered)
            score qwen3_layered $Q3-cse "--no-compare" ;;
        qwen3_single)
            score qwen3_single $Q3-single-stage "--no-compare" ;;
        foundation_8b_2xh200)
            score foundation_2xh200 $C-foundation-8b-v21-2xh200-recalibrate "" ;;
        llama31_8b_2xh200)
            score llama_2xh200 $C-llama31-8b-v21-2xh200-recalibrate "" ;;
        *) stop "unknown group $group" ;;
    esac
done
echo "all_ok groups=$* finished=$(stamp)" > ~/logs/port_score_queue.status
