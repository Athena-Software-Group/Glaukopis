#!/bin/bash
# AAIML scoring on 2xH200: same harness and serving as the published runs (--tp 2, 32K context, MMLU-Pro on).
# Order: the anchor first, then the six new checkpoints, G3 last because it is the only run that could be long.
# The scorer runs in resume mode, so a rerun keeps finished rows. A finished run leaves ~/logs/score_<name>.done
# and a rerun skips it. Any failure stops the queue.
source /root/miniconda3/bin/activate vllm
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
export BENCH_CONDA_ENV=ctibench
cd eval/utils || exit 1
Q=~/logs/score_queue.log
P=athena-cti-sft-qwen25-32b-v21
WRAP=serve_and_bench_qwen25_32b_v21_single_stage.sh
G3=athena-cti-sft-qwen3-30b-a3b-thinking-2507-v21-cse-vllm
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > ~/logs/score_queue.status; exit 1; }

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

score() {
    local name=$1 alias=$2 check=$3
    shift 3
    if [ -f ~/logs/score_$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; return 0; fi
    gpus_free || stop "$name gpus not free"
    echo "$(stamp) START $name" >> "$Q"
    "$@" > ~/logs/score_$name.log 2>&1
    local rc=$?
    echo "$(stamp) END $name rc=$rc" >> "$Q"
    [ $rc -eq 0 ] || stop "$name rc=$rc"
    python ~/check_scores.py "$alias" $check >> "$Q" 2>&1 || stop "$name completeness"
    touch ~/logs/score_$name.done
    echo "$(stamp) DONE $name" >> "$Q"
}

# Get the weights first, with a timeout and retries. A stalled Hub download otherwise uses up the server's start limit.
fetch() {
    local repo=$1 i
    for i in 1 2 3 4 5 6; do
        timeout 600 python ~/prefetch.py "$repo" >> ~/logs/score_prefetch.log 2>&1 && return 0
        echo "$(stamp) retry download $repo after attempt $i" >> "$Q"
    done
    return 1
}

qwen25() {
    local name=$1 suffix=$2
    [ -f ~/logs/score_$name.done ] || fetch "asg-ai/$P-$suffix" || stop "$name download"
    score "$name" "$P-$suffix-vllm" "" env MODEL_ALIAS="$P-$suffix-vllm" bash "$WRAP"
}

D=$(date -u +%Y%m%d)
pip freeze > ~/Glaukopis/SFT/eval/responses/eval_stack_${D}_vllm.txt 2>/dev/null
conda run -n ctibench pip freeze > ~/Glaukopis/SFT/eval/responses/eval_stack_${D}_ctibench.txt 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/logs/score_hardware_${D}.txt

qwen25 anchor recal-32b
qwen25 g1 single-stage
qwen25 g2a skiptaa-cse
qwen25 g2b skiptaa-recal-32b
qwen25 g4s1 recal-32b-seed1
qwen25 g4s2 recal-32b-seed2
qwen25 g4s3 recal-32b-seed3
[ -f ~/logs/score_g3.done ] || fetch "asg-ai/${G3%-vllm}" || stop "g3 download"
score g3 "$G3" "--no-mmlu --no-compare" bash serve_and_bench_qwen3_30b_a3b_thinking_2507_v21_cse_thinking_on.sh
echo "all_ok finished=$(stamp)" > ~/logs/score_queue.status
