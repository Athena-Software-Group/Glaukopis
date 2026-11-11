#!/bin/bash
# Per-layer breakdown scoring (11 Nov 2026), after pipeline_qwen3_4x.sh has trained and scored the Qwen3 single pass:
# the Qwen2.5-14B Layer-3 (CSE) checkpoint and the Qwen3 Core, +TAA and Layer-4 checkpoints with thinking on.
# Log ~/logs/score_layers.log; ~/logs/score_layers.status appears when it ends (all_ok or failed=...).
L=~/logs/score_layers.log
stamp() { date -u +%FT%TZ; }
fail() { echo "failed=$1 at=$(stamp)" > ~/logs/score_layers.status; echo "$(stamp) STOP $1" >> "$L"; exit 1; }
rm -f ~/logs/score_layers.status
echo "$(stamp) WAIT for pipeline_qwen3_4x" >> "$L"
until [ -f ~/logs/pipeline_qwen3_4x.status ]; do sleep 60; done
grep -q "^all_ok" ~/logs/pipeline_qwen3_4x.status || fail "pipeline_qwen3_4x did not finish cleanly: $(cat ~/logs/pipeline_qwen3_4x.status)"
free=$(df --output=avail -BG /home | tail -n 1 | tr -dc 0-9)
[ "$free" -ge 250 ] || fail "only ${free} GB free on /home, need 250 for four downloads"
echo "$(stamp) BEGIN score_queue_ports.sh layer_breakdown" >> "$L"
bash ~/score_queue_ports.sh layer_breakdown
rc=$?
echo "$(stamp) FINISH rc=$rc" >> "$L"
[ $rc -eq 0 ] || fail "score_queue_ports.sh layer_breakdown rc=$rc"
echo "all_ok finished=$(stamp)" > ~/logs/score_layers.status
