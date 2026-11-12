#!/bin/bash
# Rerun of the Qwen3 single-pass scoring and the per-layer gap scoring (11 Nov 2026).
# The first attempt ran the bench client with the llm-sft python (no evaluate module), because conda run
# replaces only the base prefix in PATH and an env bin ahead of it wins. Start from a clean PATH and check first.
export HOME=/home
export PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER VIRTUAL_ENV PYTHONPATH
L=~/logs/score_rerun.log
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "$(stamp) FAIL $*" >> "$L"; echo "failed=$* at=$(stamp)" > ~/logs/score_layers.status; exit 1; }
py=$(conda run -n ctibench bash -c "command -v python" 2>/dev/null | tail -1)
echo "$(stamp) bench python: $py" >> "$L"
[ "$py" = /root/miniconda3/envs/ctibench/bin/python ] || fail "bench python is $py"
conda run -n ctibench python -c "import evaluate" || fail "evaluate missing in ctibench"
free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
[ "$free" -ge 250 ] || fail "only ${free} GB free"
echo "$(stamp) BEGIN qwen3_single" >> "$L"
bash ~/score_queue_ports.sh qwen3_single || fail "qwen3_single"
echo "$(stamp) BEGIN layer_breakdown" >> "$L"
bash ~/score_queue_ports.sh layer_breakdown || fail "layer_breakdown"
echo "all_ok finished=$(stamp)" > ~/logs/score_layers.status
echo "$(stamp) DONE" >> "$L"
