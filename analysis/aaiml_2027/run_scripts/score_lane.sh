#!/bin/bash
# Parallel scoring lane (11 Nov 2026): scores checkpoints on a fixed GPU pair and vLLM port so two lanes share the 4xH200.
# Same serving and bench stack as score_queue_ports.sh (--tp 2, vllm env, ctibench bench client); only GPUs and port differ.
# Usage: score_lane.sh LANE GPUS PORT WAIT_STATUS_FILE|- "name|stem|check" ...
LANE=$1 GPUS=$2 PORT=$3 WAITF=$4; shift 4
export HOME=/home
export PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER VIRTUAL_ENV PYTHONPATH
Q=~/logs/score_lane_$LANE.log
ST=~/logs/score_lane_$LANE.status
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > "$ST"; exit 1; }
echo "$(stamp) lane $LANE gpus=$GPUS port=$PORT wait=$WAITF items=$*" >> "$Q"
if [ "$WAITF" != - ]; then
    for i in $(seq 1 720); do [ -f "$WAITF" ] && break; sleep 30; done
    [ -f "$WAITF" ] || stop "timed out waiting for $WAITF"
    grep -q all_ok "$WAITF" || stop "upstream not ok: $(cat $WAITF)"
fi
source /root/miniconda3/bin/activate vllm
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
export BENCH_CONDA_ENV=ctibench CUDA_VISIBLE_DEVICES=$GPUS VLLM_BASE_URL=http://localhost:$PORT/v1
cd eval/utils || exit 1
WRAP=serve_and_bench_v21_single_stage_port.sh
[ "$PORT" = 8000 ] || WRAP=serve_and_bench_v21_single_stage_port_p$PORT.sh
[ -f "$WRAP" ] || stop "no wrapper $WRAP"
py=$(conda run -n ctibench bash -c "command -v python" 2>/dev/null | tail -1)
echo "$(stamp) bench python: $py" >> "$Q"
[ "$py" = /root/miniconda3/envs/ctibench/bin/python ] || stop "bench python is $py"
gpus_free() {
    local i used
    for i in $(seq 1 40); do
        used=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "${used:-99999}" -lt 2000 ] && return 0
        sleep 3
    done
    return 1
}
fetch() {
    local repo=$1 i
    for i in 1 2 3 4 5 6; do
        timeout 600 python ~/prefetch.py "$repo" >> ~/logs/port_score_prefetch.log 2>&1 && return 0
        echo "$(stamp) retry download $repo after attempt $i" >> "$Q"
    done
    return 1
}
for item in "$@"; do
    IFS="|" read -r name stem check <<< "$item"
    alias="$stem-vllm"
    if [ -f ~/logs/port_score_lane_$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; continue; fi
    fetch "asg-ai/$stem" || stop "$name download"
    gpus_free || stop "$name gpus $GPUS not free"
    echo "$(stamp) START $name ($alias)" >> "$Q"
    MODEL_ALIAS="$alias" bash "$WRAP" > ~/logs/port_score_$name.log 2>&1
    rc=$?
    echo "$(stamp) END $name rc=$rc" >> "$Q"
    [ $rc -eq 0 ] || stop "$name rc=$rc"
    python ~/check_scores.py "$alias" $check >> "$Q" 2>&1 || stop "$name completeness"
    touch ~/logs/port_score_lane_$name.done
    echo "$(stamp) DONE $name" >> "$Q"
done
echo "all_ok lane=$LANE finished=$(stamp)" > "$ST"
