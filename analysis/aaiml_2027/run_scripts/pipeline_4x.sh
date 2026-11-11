#!/bin/bash
# 4xH200 round: Qwen2.5-14B single pass, trained and then scored on the 9 Nov stack. The Qwen3 single pass follows in
# pipeline_qwen3_4x.sh. Each queue skips finished work, so a rerun resumes.
# Log ~/logs/pipeline_4x.log; ~/logs/pipeline_4x.status appears when the pipeline ends (all_ok or failed=...).
# As first launched on 11 Nov this also ran layered retrains of Qwen2.5-14B and -32B; those steps were cancelled
# (made no-ops) before they started, because the study takes the single pass as its baseline.
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
echo "all_ok finished=$(stamp)" > ~/logs/pipeline_4x.status
