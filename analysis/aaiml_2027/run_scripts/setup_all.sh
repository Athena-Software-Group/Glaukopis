#!/bin/bash
# Reinstall the training and scoring environments after a resize, then pin scoring to the May 2026 package versions.
export PATH=/root/miniconda3/bin:$PATH
S=~/setup_all.status
rm -f $S
cd ~/Glaukopis/SFT/utils || exit 1
LIGER_SPEC="liger-kernel==0.8.0" BNB_SPEC="bitsandbytes==0.49.2" ./setup.sh --cuda cu128 --mode train --no-disk-cleanup </dev/null > ~/logs/setup_train_4x.log 2>&1; t=$?
echo "train=$t at=$(date -u +%FT%TZ)" >> $S
VLLM_SPEC="vllm==0.21.0" ./setup.sh --cuda cu128 --mode vllm --no-disk-cleanup </dev/null > ~/logs/setup_vllm_4x.log 2>&1; v=$?
echo "vllm=$v at=$(date -u +%FT%TZ)" >> $S
bash ~/pin_may.sh > ~/logs/pin_may_4x.log 2>&1; p=$?
echo "pin=$p at=$(date -u +%FT%TZ)" >> $S
echo "done" >> $S
