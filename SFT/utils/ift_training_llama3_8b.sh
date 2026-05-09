#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

# For gated models, authenticate with one of:
# 1) export HF_TOKEN=hf_xxx
# 2) hf auth login or huggingface-cli login 

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --do_eval True \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset train \
    --eval_dataset validation \
    --cutoff_len 2048 \
    --learning_rate 1e-04 \
    --num_train_epochs 2.0 \
    --max_samples 150000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_ratio 0.05 \
    --packing False \
    --enable_thinking False \
    --report_to wandb \
    --output_dir saves/Llama-3.1-8B-Instruct/lora/train_${TIMESTAMP} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 18000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target all \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 16
