#!/usr/bin/env bash
#
# Reclaim disk on the SFT/benchmark host.
#
# Targets the four largest accumulators we have seen fill /home and
# /workspace mid-run (ENOSPC on training save, ENOSPC on jsonl write):
#   1. SFT/saves/<model>/<run>/checkpoint-* and global_step* dirs
#   2. SFT/wandb/run-* dirs
#   3. HF hub cache snapshots (/workspace/.cache/huggingface/hub/*)
#   4. .partial / .sanitize.tmp / .bak scratch files left behind
#
# Default mode is dry-run (lists what would be deleted, frees nothing).
# Pass --apply to actually remove. Pass --yes to skip per-section prompts.
#
# Usage:
#   ./cleanup_disk.sh                  # dry-run, all sections, prompts
#   ./cleanup_disk.sh --apply          # delete, prompt per section
#   ./cleanup_disk.sh --apply --yes    # delete, no prompts (non-interactive)
#   ./cleanup_disk.sh --apply --keep-latest-checkpoint
#                                      # keep newest checkpoint-N per run dir
#   ./cleanup_disk.sh --hf-cache-keep "Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.1-8B-Instruct"
#                                      # never touch these HF snapshots

set -euo pipefail

APPLY=0
ASSUME_YES=0
KEEP_LATEST_CKPT=0
HF_CACHE_KEEP=""
SFT_ROOT="${SFT_ROOT:-/home/Glaukopis/SFT}"
HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        --keep-latest-checkpoint) KEEP_LATEST_CKPT=1; shift ;;
        --hf-cache-keep) HF_CACHE_KEEP="$2"; shift 2 ;;
        --sft-root) SFT_ROOT="$2"; shift 2 ;;
        --hf-cache-dir) HF_HUB_CACHE="$2"; shift 2 ;;
        -h|--help) sed -n '1,/^set -e/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

confirm() {
    [[ $ASSUME_YES -eq 1 ]] && return 0
    read -r -p "  proceed with this section? [y/N] " ans
    [[ "$ans" =~ ^[yY]$ ]]
}

do_rm() {
    local target="$1"
    if [[ $APPLY -eq 1 ]]; then
        rm -rf -- "$target"
    fi
}

human_size() {
    du -sh "$1" 2>/dev/null | awk '{print $1}'
}

# Sum sizes of paths as kilobytes; portable across GNU/BSD coreutils.
sum_kb() {
    local kb=0 part
    for p in "$@"; do
        [[ -e "$p" ]] || continue
        part=$(du -sk "$p" 2>/dev/null | awk '{print $1}')
        [[ -n "$part" ]] && kb=$((kb + part))
    done
    echo "$kb"
}

# Pretty-print a kilobyte total.
kb_human() {
    local kb=$1
    if   [[ $kb -lt 1024 ]];        then echo "${kb}K"
    elif [[ $kb -lt 1048576 ]];     then awk -v k="$kb" 'BEGIN{printf "%.1fM\n", k/1024}'
    else                                 awk -v k="$kb" 'BEGIN{printf "%.1fG\n", k/1024/1024}'
    fi
}

banner() {
    echo
    echo "=========================================================="
    echo "  $1"
    echo "=========================================================="
}

mode_str="DRY-RUN"; [[ $APPLY -eq 1 ]] && mode_str="APPLY"
banner "cleanup_disk.sh — mode: $mode_str"
df -h "$SFT_ROOT" "$HF_HUB_CACHE" 2>/dev/null | sort -u

# --- Section 1: SFT training checkpoints ---------------------------------
banner "1. SFT checkpoints under $SFT_ROOT/saves/"
saves_dir="$SFT_ROOT/saves"
if [[ -d "$saves_dir" ]]; then
    ckpt_dirs=()
    while IFS= read -r d; do ckpt_dirs+=("$d"); done < <(find "$saves_dir" -maxdepth 4 -type d \
        \( -name 'checkpoint-*' -o -name 'global_step*' \) 2>/dev/null | sort)
    if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
        echo "  (none found)"
    else
        if [[ $KEEP_LATEST_CKPT -eq 1 ]]; then
            # For each parent dir (the per-run dir), keep the checkpoint with
            # the highest trailing integer (e.g. checkpoint-1821 over
            # checkpoint-200, global_step9000 over global_step100). Done in
            # Python for portable numeric extraction + grouping.
            keep_set=$(printf '%s\n' "${ckpt_dirs[@]}" | python3 -c "
import sys, os, re
groups = {}
for line in sys.stdin:
    p = line.rstrip()
    if not p: continue
    parent = os.path.dirname(p)
    m = re.search(r'(\d+)', os.path.basename(p))
    n = int(m.group(1)) if m else -1
    if parent not in groups or groups[parent][0] < n:
        groups[parent] = (n, p)
for _, p in groups.values(): print(p)
")
            new_list=()
            for d in "${ckpt_dirs[@]}"; do
                grep -qxF "$d" <<<"$keep_set" || new_list+=("$d")
            done
            ckpt_dirs=("${new_list[@]}")
            echo "  (--keep-latest-checkpoint: keeping newest checkpoint per run)"
        fi
        total_kb=$(sum_kb "${ckpt_dirs[@]}")
        for d in "${ckpt_dirs[@]}"; do
            echo "  $(human_size "$d")  $d"
        done
        printf '  ----\n  total: %s across %d dir(s)\n' \
            "$(kb_human "$total_kb")" "${#ckpt_dirs[@]}"
        if [[ $APPLY -eq 1 ]] && confirm; then
            for d in "${ckpt_dirs[@]}"; do do_rm "$d"; done
            echo "  removed."
        fi
    fi
fi

# --- Section 2: wandb run dirs -------------------------------------------
banner "2. wandb runs under $SFT_ROOT/wandb/"
wandb_dir="$SFT_ROOT/wandb"
if [[ -d "$wandb_dir" ]]; then
    wandb_runs=()
    while IFS= read -r d; do wandb_runs+=("$d"); done < <(find "$wandb_dir" -mindepth 1 -maxdepth 1 -type d \
        \( -name 'run-*' -o -name 'offline-run-*' \) 2>/dev/null | sort)
    if [[ ${#wandb_runs[@]} -eq 0 ]]; then
        echo "  (none found)"
    else
        for d in "${wandb_runs[@]}"; do echo "  $(human_size "$d")  $d"; done
        echo "  ---- $(human_size "$wandb_dir") total under $wandb_dir"
        if [[ $APPLY -eq 1 ]] && confirm; then
            for d in "${wandb_runs[@]}"; do do_rm "$d"; done
            echo "  removed."
        fi
    fi
fi

# --- Section 3: HF hub cache snapshots -----------------------------------
banner "3. HF hub cache snapshots under $HF_HUB_CACHE"
if [[ -d "$HF_HUB_CACHE" ]]; then
    hf_models=()
    while IFS= read -r d; do hf_models+=("$d"); done < <(find "$HF_HUB_CACHE" -mindepth 1 -maxdepth 1 -type d \
        -name 'models--*' 2>/dev/null | sort)
    keep_norm=$(echo "$HF_CACHE_KEEP" | tr ' ' '\n' | sed 's|/|--|g; s|^|models--|' | sort -u)
    for d in "${hf_models[@]}"; do
        base=$(basename "$d")
        marker=""
        if grep -qx "$base" <<<"$keep_norm" 2>/dev/null; then marker=" [KEEP]"; fi
        echo "  $(human_size "$d")  $base$marker"
    done
    if [[ $APPLY -eq 1 ]] && confirm; then
        for d in "${hf_models[@]}"; do
            base=$(basename "$d")
            grep -qx "$base" <<<"$keep_norm" 2>/dev/null && continue
            do_rm "$d"
        done
        echo "  removed (kept entries marked [KEEP])."
    fi
fi

# --- Section 4: stray scratch / partial / backup files --------------------
banner "4. Scratch files (.partial, .sanitize.tmp, .bak, *.log.old)"
scratch=()
while IFS= read -r f; do scratch+=("$f"); done < <(find "$SFT_ROOT" \( -name '*.partial' -o -name '*.sanitize.tmp' \
    -o -name '*.bak' -o -name '*.log.old' \) -type f 2>/dev/null | sort)
if [[ ${#scratch[@]} -eq 0 ]]; then
    echo "  (none found)"
else
    for f in "${scratch[@]}"; do echo "  $(human_size "$f")  $f"; done
    if [[ $APPLY -eq 1 ]] && confirm; then
        for f in "${scratch[@]}"; do [[ $APPLY -eq 1 ]] && rm -f -- "$f"; done
        echo "  removed."
    fi
fi

banner "Final disk state"
df -h "$SFT_ROOT" "$HF_HUB_CACHE" 2>/dev/null | sort -u
[[ $APPLY -eq 0 ]] && echo "(dry-run; pass --apply to actually delete)"
