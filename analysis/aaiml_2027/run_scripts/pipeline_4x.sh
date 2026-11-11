#!/bin/bash
# 4xH200 round: Qwen2.5-14B single pass, then the layered chains for Qwen2.5-14B and -32B retrained on the same hardware,
# each scored on the 9 Nov stack as soon as it is trained. Each queue skips finished work, so a rerun resumes.
# Log ~/logs/pipeline_4x.log; ~/logs/pipeline_4x.status appears when the pipeline ends (all_ok or failed=...).
L=~/logs/pipeline_4x.log
stamp() { date -u +%FT%TZ; }
step() {
    echo "$(stamp) BEGIN $*" >> "$L"
    "$@"
    local rc=$?
    echo "$(stamp) FINISH $* rc=$rc" >> "$L"
    if [ $rc -ne 0 ]; then echo "failed=$* at=$(stamp)" > ~/logs/pipeline_4x.status; exit 1; fi
}
rm -f ~/logs/pipeline_4x.status
step bash ~/train_queue_ports.sh qwen25_14b
step bash ~/score_queue_ports.sh qwen25_14b_single
step bash ~/control_chain_4x.sh qwen25_14b
step bash ~/score_queue_ports.sh qwen25_14b_4xh200
step bash ~/control_chain_4x.sh qwen25_32b
step bash ~/score_queue_ports.sh qwen25_32b_4xh200
echo "all_ok finished=$(stamp)" > ~/logs/pipeline_4x.status
