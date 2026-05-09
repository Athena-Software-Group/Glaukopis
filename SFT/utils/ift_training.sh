#!/bin/bash
set -e

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --do_eval True \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ift_data \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 150000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --report_to wandb \
    --output_dir saves/Qwen2.5-14B-Instruct/lora/train_${TIMESTAMP} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --val_size 0.2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 4
