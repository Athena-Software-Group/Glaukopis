#!/bin/bash
# Single-pass controls for the four other v21 models. Usage: train_queue_ports.sh KEY [KEY...]
# After each run: keep its training records, verify the Hugging Face upload by hash, then remove the
# local folder. A finished job leaves ~/logs/port_<key>.done, so a rerun skips it. Any failure stops the queue.
source /root/miniconda3/bin/activate llm-sft
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
cd autotrain || exit 1
Q=~/logs/port_queue.log
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > ~/logs/port_queue.status; exit 1; }

job() {
    local key=$1 safe=$2 slug=$3 min_gb=$4 min_gpus=$5
    local name=port_$key dir=/home/Glaukopis/SFT/saves/$safe/full/port_single_stage_$key
    local repo=asg-ai/athena-cti-sft-$slug-v21-single-stage
    if [ -f ~/logs/$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; return 0; fi
    local gpus; gpus=$(nvidia-smi -L | wc -l)
    [ "$gpus" -ge "$min_gpus" ] || stop "$name needs $min_gpus GPUs, found $gpus"
    local resume=""; [ -d "$dir" ] && ls -d "$dir"/checkpoint-* > /dev/null 2>&1 && resume="--resume"
    echo "$(stamp) START $name gpus=$gpus $resume" >> "$Q"
    ./run_sft_v21_single_stage_port.sh --model-key "$key" --no-offload --resumable $resume --output-dir "$dir" > ~/logs/$name.log 2>&1
    local rc=$?
    echo "$(stamp) END $name rc=$rc" >> "$Q"
    [ $rc -eq 0 ] || stop "$name rc=$rc"
    mkdir -p ~/logs/runs/$name
    for f in trainer_log.jsonl trainer_state.json all_results.json train_results.json eval_results.json train_config.json train.log; do
        cp -p "$dir/$f" ~/logs/runs/$name/ 2>/dev/null
    done
    cp -p ~/logs/$name.log ~/logs/runs/$name/launcher.log
    python ~/verify_hf_ports.py "$dir" "$repo" --min-gb "$min_gb" --prune >> "$Q" 2>&1 || stop "$name upload verification"
    touch ~/logs/$name.done
    echo "$(stamp) DONE $name" >> "$Q"
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/logs/port_hardware_$(date -u +%Y%m%d_%H%M).txt
pip freeze > ~/logs/port_train_env_freeze_$(date -u +%Y%m%d_%H%M).txt 2>/dev/null

for key in "$@"; do
    case "$key" in
        foundation_8b)      job "$key" fdtn-ai_Foundation-Sec-8B-Instruct foundation-8b 15 2 ;;
        llama31_8b)         job "$key" meta-llama_Llama-3.1-8B-Instruct llama31-8b 15 2 ;;
        qwen25_14b)         job "$key" Qwen_Qwen2.5-14B-Instruct qwen25-14b 26 4 ;;
        qwen3_30b_thinking) job "$key" Qwen_Qwen3-30B-A3B-Thinking-2507 qwen3-30b-a3b-thinking-2507 55 4 ;;
        *) stop "unknown key $key" ;;
    esac
done
echo "all_ok keys=$* finished=$(stamp)" > ~/logs/port_queue.status
