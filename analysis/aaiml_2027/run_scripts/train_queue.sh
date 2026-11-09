#!/bin/bash
# AAIML control runs on 4xH200: same v21 shards and harness, no offload, effective batch preserved.
# After each run: keep its training records, verify the Hugging Face upload by hash, then remove the
# local folder. A finished job leaves ~/logs/<name>.done, so a rerun skips it. Any failure stops the queue.
source /root/miniconda3/bin/activate llm-sft
cd ~/Glaukopis/SFT || exit 1
set -a; source ./.env; set +a
cd autotrain || exit 1
S=/home/Glaukopis/SFT/saves/Qwen_Qwen2.5-32B-Instruct/full
P=asg-ai/athena-cti-sft-qwen25-32b-v21
Q=~/logs/queue.log
stamp() { date -u +%FT%TZ; }
stop() { echo "$(stamp) STOP $1" >> "$Q"; echo "failed=$1 at=$(stamp)" > ~/logs/queue.status; exit 1; }

job() {
    local name=$1 dir=$S/$2 repo=$P-$3
    shift 3
    if [ -f ~/logs/$name.done ]; then echo "$(stamp) SKIP $name" >> "$Q"; return 0; fi
    echo "$(stamp) START $name" >> "$Q"
    "$@" --output-dir "$dir" > ~/logs/$name.log 2>&1
    local rc=$?
    echo "$(stamp) END $name rc=$rc" >> "$Q"
    [ $rc -eq 0 ] || stop "$name rc=$rc"
    mkdir -p ~/logs/runs/$name
    for f in trainer_log.jsonl trainer_state.json all_results.json train_results.json eval_results.json train_config.json train.log; do
        cp -p "$dir/$f" ~/logs/runs/$name/ 2>/dev/null
    done
    cp -p ~/logs/$name.log ~/logs/runs/$name/launcher.log
    python ~/verify_hf.py "$dir" "$repo" --prune >> "$Q" 2>&1 || stop "$name upload verification"
    touch ~/logs/$name.done
    echo "$(stamp) DONE $name" >> "$Q"
}

job g2b g2b_skiptaa_recal skiptaa-recal-32b ./run_sft_qwen25_32b_v21_recal_32b.sh --no-offload \
    --base-model $P-skiptaa-cse --repo-id $P-skiptaa-recal-32b
job g1 g1_single_stage single-stage ./run_sft_qwen25_32b_v21_single_stage.sh --resumable --no-offload
for s in 1 2 3; do
    job g4s$s g4_seed$s recal-32b-seed$s ./run_sft_qwen25_32b_v21_recal_32b.sh --no-offload --seed $s
done
echo "all_ok finished=$(stamp)" > ~/logs/queue.status
