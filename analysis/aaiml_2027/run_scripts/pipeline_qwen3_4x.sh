#!/bin/bash
# Qwen3-30B-A3B-Thinking single pass on 4xH200, after pipeline_4x.sh has finished the Qwen2.5-14B single pass.
# Log ~/logs/pipeline_qwen3_4x.log; ~/logs/pipeline_qwen3_4x.status appears when it ends (all_ok or failed=...).
L=~/logs/pipeline_qwen3_4x.log
stamp() { date -u +%FT%TZ; }
fail() { echo "failed=$1 at=$(stamp)" > ~/logs/pipeline_qwen3_4x.status; echo "$(stamp) STOP $1" >> "$L"; exit 1; }
step() {
    echo "$(stamp) BEGIN $*" >> "$L"
    "$@"
    local rc=$?
    echo "$(stamp) FINISH $* rc=$rc" >> "$L"
    [ $rc -eq 0 ] || fail "$* rc=$rc"
}
rm -f ~/logs/pipeline_qwen3_4x.status
echo "$(stamp) WAIT for pipeline_4x (Qwen2.5-14B single pass)" >> "$L"
until [ -f ~/logs/pipeline_4x.status ]; do sleep 60; done
grep -q '^all_ok' ~/logs/pipeline_4x.status || fail "pipeline_4x did not finish cleanly: $(cat ~/logs/pipeline_4x.status)"
# Resumable Qwen3 checkpoints peak near 480 GB, plus the 61 GB base download.
free=$(df --output=avail -BG /home | tail -n 1 | tr -dc 0-9)
[ "$free" -ge 560 ] || fail "only ${free} GB free on /home, Qwen3 needs 560"
step bash ~/train_queue_ports.sh qwen3_30b_thinking
step bash ~/score_queue_ports.sh qwen3_single
echo "all_ok finished=$(stamp)" > ~/logs/pipeline_qwen3_4x.status
