#!/bin/bash
# Hardware control: retrain the layered v21 chain on 4xH200, the hardware of the Qwen2.5-14B and -32B single-pass runs.
# Usage: control_chain_4x.sh KEY [KEY...]   (qwen25_14b, qwen25_32b)
# Same stage launchers, shards and effective batches as the May chains (16, 8, 16, 16, 8), with --no-offload like the
# single passes. New Hugging Face repos (-v21-4xh200-*) and local parents, so the published checkpoints are neither read
# nor overwritten. These launchers save weights-only checkpoints, which cannot be resumed, so a rerun restarts any
# unfinished stage from its parent. A finished stage leaves ~/logs/ctrl_<key>_<stage>.done and a rerun skips it.
# Any failure stops the queue.
source /root/miniconda3/bin/activate llm-sft
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
cd autotrain || exit 1
Q=~/logs/ctrl_queue.log
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > ~/logs/ctrl_queue.status; exit 1; }

# stage NAME OUTPUT_DIR COMMAND...: run one stage, keep its records, drop its checkpoints (the final weights stay).
stage() {
    local name=$1 dir=$2
    shift 2
    if [ -f ~/logs/$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; return 0; fi
    rm -rf "$dir"
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

# release DIR REPO MIN_GB: check the uploaded weights match the local ones, then delete the local copy. A missing
# directory was released on an earlier run.
release() {
    [ -d "$1" ] || return 0
    python ~/verify_hf_ports.py "$1" "$2" --min-gb "$3" --prune >> "$Q" 2>&1 || stop "$2 upload verification"
}

# chain KEY SAFE_DIR SLUG MIN_FREE_GB MIN_UPLOAD_GB LAST_LAUNCHER LAST_NAME [LAST_ARGS...]
chain() {
    local key=$1 safe=$2 slug=$3 min_free=$4 min_gb=$5 last=$6 last_name=$7
    shift 7
    local n=ctrl_$key root=/home/Glaukopis/SFT/saves/$safe/full/ctrl4x_$key hf=asg-ai/athena-cti-sft-$slug-v21-4xh200 L=./run_sft_${key}_v21
    local A=$root/phase_a B=$root/phase_b T=$root/taa C=$root/cse R=$root/${last_name//-/_}
    if [ -f ~/logs/$n.done ]; then echo "$(stamp) SKIP $n" >> "$Q"; return 0; fi
    [ "$(nvidia-smi -L | wc -l)" -eq 4 ] || stop "$n must run on exactly 4 GPUs"
    if [ ! -f ~/logs/${n}_phase_a.done ]; then
        local free; free=$(df --output=avail -BG /home | tail -n 1 | tr -dc 0-9)
        [ "$free" -ge "$min_free" ] || stop "$n only ${free} GB free on /home, needs $min_free"
    fi
    mkdir -p "$root"
    # Each parent is released as soon as its child stage is done, so at most one parent, two checkpoints and one
    # final model are on disk at a time (about 260 GB at 32B).
    stage ${n}_phase_a "$A" bash "$L"_core.sh --phase a --phase-a-dir "$A" --phase-b-dir "$B" --repo-id $hf-core --no-offload
    stage ${n}_phase_b "$B" bash "$L"_core.sh --phase b --phase-a-dir "$A" --phase-b-dir "$B" --repo-id $hf-core --no-offload
    rm -rf "$A"
    stage ${n}_taa "$T" bash "$L"_plus_taa.sh --base-model "$B" --output-dir "$T" --repo-id $hf-taa --no-offload
    release "$B" $hf-core $min_gb
    stage ${n}_cse "$C" bash "$L"_final.sh --base-model "$T" --output-dir "$C" --repo-id $hf-cse --no-offload
    release "$T" $hf-taa $min_gb
    stage ${n}_$last_name "$R" bash "$L"_$last.sh --base-model "$C" --output-dir "$R" --repo-id $hf-$last_name --no-offload "$@"
    release "$C" $hf-cse $min_gb
    release "$R" $hf-$last_name $min_gb
    rmdir "$root"
    touch ~/logs/$n.done
    echo "$(stamp) DONE $n" >> "$Q"
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/logs/ctrl_hardware_$(date -u +%Y%m%d_%H%M).txt
pip freeze > ~/logs/ctrl_train_env_freeze_$(date -u +%Y%m%d_%H%M).txt 2>/dev/null

for key in "$@"; do
    case "$key" in
        # Layer 4 targets effective batch 4 below 8 GPUs; May ran 8.
        qwen25_14b) chain "$key" Qwen_Qwen2.5-14B-Instruct qwen25-14b 200 25 recalibrate recalibrate --eff-batch 8 ;;
        # Recal-32B already targets effective batch 8 (2 accumulation steps on 4 GPUs).
        qwen25_32b) chain "$key" Qwen_Qwen2.5-32B-Instruct qwen25-32b 300 60 recal_32b recal-32b ;;
        *) stop "unknown key $key" ;;
    esac
done
echo "all_ok keys=$* finished=$(stamp)" > ~/logs/ctrl_queue.status
