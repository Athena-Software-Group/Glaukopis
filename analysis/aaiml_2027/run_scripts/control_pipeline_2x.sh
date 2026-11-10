#!/bin/bash
# Hardware control, after the cross-model pipeline on the same 2xH200 box. Waits for that pipeline to report all_ok, so a
# failed step can be fixed and rerun first. Each queue skips finished jobs, so a rerun picks up where this stopped.
L=~/logs/ctrl_pipeline.log
stamp() { date -u +%FT%TZ; }
step() {
    echo "$(stamp) BEGIN $*" >> "$L"
    bash "$@"
    local rc=$?
    echo "$(stamp) FINISH $* rc=$rc" >> "$L"
    if [ $rc -ne 0 ]; then echo "failed=$* at=$(stamp)" > ~/logs/ctrl_pipeline.status; exit 1; fi
}
rm -f ~/logs/ctrl_pipeline.status
echo "$(stamp) WAIT for the cross-model pipeline" >> "$L"
until grep -q '^all_ok' ~/logs/port_pipeline.status 2>/dev/null; do sleep 60; done
step ~/control_chain_2x.sh foundation_8b
step ~/score_queue_ports.sh foundation_8b_2xh200
step ~/control_chain_2x.sh llama31_8b
step ~/score_queue_ports.sh llama31_8b_2xh200
echo "all_ok finished=$(stamp)" > ~/logs/ctrl_pipeline.status
