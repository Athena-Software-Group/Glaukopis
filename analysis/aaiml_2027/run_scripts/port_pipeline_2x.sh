#!/bin/bash
# Everything from the cross-model single-pass round that fits on 2xH200, in order. Stops at the first failure.
# Each queue skips finished jobs, so a rerun picks up where this stopped.
L=~/logs/port_pipeline.log
stamp() { date -u +%FT%TZ; }
step() {
    echo "$(stamp) BEGIN $*" >> "$L"
    bash "$@"
    local rc=$?
    echo "$(stamp) FINISH $* rc=$rc" >> "$L"
    if [ $rc -ne 0 ]; then echo "failed=$* at=$(stamp)" > ~/logs/port_pipeline.status; exit 1; fi
}
rm -f ~/logs/port_pipeline.status
step ~/train_queue_ports.sh foundation_8b
step ~/score_queue_ports.sh foundation_8b
step ~/train_queue_ports.sh llama31_8b
step ~/score_queue_ports.sh llama31_8b
step ~/score_queue_ports.sh qwen25_14b_layered
step ~/score_queue_ports.sh qwen3_layered
echo "all_ok finished=$(stamp)" > ~/logs/port_pipeline.status
