#!/bin/bash
# Hardware control: retrain the layered v21 chain on 2xH200, the hardware of the 8B single-pass runs.
# Usage: control_chain_2x.sh KEY [KEY...]   (foundation_8b, llama31_8b)
# Same stage launchers, shards and effective batches as the May 8-GPU chain. New Hugging Face repos (-v21-2xh200-*) and
# local parents, so the published checkpoints are neither read nor overwritten. A finished stage leaves
# ~/logs/ctrl_<key>_<stage>.done and a rerun skips it. Any failure stops the queue.
source /root/miniconda3/bin/activate llm-sft
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
cd autotrain || exit 1
Q=~/logs/ctrl_queue.log
MIN_FREE_GB=330
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > ~/logs/ctrl_queue.status; exit 1; }
resume_flag() { ls -d "$1"/checkpoint-* > /dev/null 2>&1 && echo --resume; }

# stage NAME OUTPUT_DIR COMMAND...: run one stage, keep its records, drop its checkpoints (the final weights stay).
stage() {
    local name=$1 dir=$2
    shift 2
    if [ -f ~/logs/$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; return 0; fi
    echo "$(stamp) START $name" >> "$Q"
    "$@" > ~/logs/$name.log 2>&1
    local rc=$?
    echo "$(stamp) END $name rc=$rc" >> "$Q"
    [ $rc -eq 0 ] || stop "$name rc=$rc"
    ls "$dir"/*.safetensors > /dev/null 2>&1 || stop "$name left no weights in $dir"
    mkdir -p ~/logs/runs/$name
    for f in trainer_log.jsonl trainer_state.json all_results.json train_results.json eval_results.json train_config.json train.log; do
        cp -p "$dir/$f" ~/logs/runs/$name/ 2>/dev/null
    done
    cp -p ~/logs/$name.log ~/logs/runs/$name/launcher.log
    rm -rf "$dir"/checkpoint-*
    touch ~/logs/$name.done
    echo "$(stamp) DONE $name" >> "$Q"
}

chain() {
    local key=$1 safe=$2 slug=$3
    local n=ctrl_$key root=/home/Glaukopis/SFT/saves/$safe/full/ctrl2x_$key hf=asg-ai/athena-cti-sft-$slug-v21-2xh200 L=./run_sft_${key}_v21
    local A=$root/phase_a B=$root/phase_b T=$root/taa C=$root/cse R=$root/recalibrate
    if [ -f ~/logs/$n.done ]; then echo "$(stamp) SKIP $n" >> "$Q"; return 0; fi
    [ "$(nvidia-smi -L | wc -l)" -eq 2 ] || stop "$n must run on exactly 2 GPUs"
    if [ ! -f ~/logs/${n}_phase_a.done ]; then
        local free; free=$(df --output=avail -BG /home | tail -n 1 | tr -dc 0-9)
        [ "$free" -ge $MIN_FREE_GB ] || stop "$n only ${free} GB free on /home, needs $MIN_FREE_GB"
    fi
    mkdir -p "$root"
    stage ${n}_phase_a "$A" bash "$L"_core.sh --phase a --phase-a-dir "$A" --phase-b-dir "$B" --repo-id $hf-core --no-offload $(resume_flag "$A")
    stage ${n}_phase_b "$B" bash "$L"_core.sh --phase b --phase-a-dir "$A" --phase-b-dir "$B" --repo-id $hf-core --no-offload $(resume_flag "$B")
    [ -f ~/logs/${n}_taa.done ] || rm -rf "$T"
    stage ${n}_taa "$T" bash "$L"_plus_taa.sh --base-model "$B" --output-dir "$T" --repo-id $hf-taa --no-offload
    [ -f ~/logs/${n}_cse.done ] || rm -rf "$C"
    stage ${n}_cse "$C" bash "$L"_final.sh --base-model "$T" --output-dir "$C" --repo-id $hf-cse --no-offload
    [ -f ~/logs/${n}_recalibrate.done ] || rm -rf "$R"
    stage ${n}_recalibrate "$R" bash "$L"_recalibrate.sh --base-model "$C" --output-dir "$R" --repo-id $hf-recalibrate --no-offload --eff-batch 8
    python ~/verify_hf_ports.py "$B" $hf-core --min-gb 15 --prune >> "$Q" 2>&1 || stop "$n core upload verification"
    python ~/verify_hf_ports.py "$T" $hf-taa --min-gb 15 --prune >> "$Q" 2>&1 || stop "$n taa upload verification"
    python ~/verify_hf_ports.py "$C" $hf-cse --min-gb 15 --prune >> "$Q" 2>&1 || stop "$n cse upload verification"
    python ~/verify_hf_ports.py "$R" $hf-recalibrate --min-gb 15 --prune >> "$Q" 2>&1 || stop "$n recalibrate upload verification"
    rm -rf "$A"
    rmdir "$root"
    touch ~/logs/$n.done
    echo "$(stamp) DONE $n" >> "$Q"
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/logs/ctrl_hardware_$(date -u +%Y%m%d_%H%M).txt
pip freeze > ~/logs/ctrl_train_env_freeze_$(date -u +%Y%m%d_%H%M).txt 2>/dev/null

for key in "$@"; do
    case "$key" in
        foundation_8b) chain "$key" fdtn-ai_Foundation-Sec-8B-Instruct foundation-8b ;;
        llama31_8b)    chain "$key" meta-llama_Llama-3.1-8B-Instruct llama31-8b ;;
        *) stop "unknown key $key" ;;
    esac
done
echo "all_ok keys=$* finished=$(stamp)" > ~/logs/ctrl_queue.status
