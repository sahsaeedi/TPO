#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

OUTPUT_DIR="./tpo"
DATASET_NAME_OR_PATH=""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
     run_tpo.py  \
    --model_name_or_path meta-llama/meta-Llama-3-8B \
    --tokenizer_name meta-llama/meta-Llama-3-8B  \
    --is_three_preference true \
    --beta 0.01  \
    --tpo_alpha 1  \
    --do_train  \
    --bf16   \
    --attn_implementation flash_attention_2 \
    --learning_rate 5.0e-7 \
    --gradient_accumulation_steps 1  \
    --lr_scheduler_type cosine  \
    --optim adamw_torch  \
    --warmup_ratio 0.1   \
    --save_steps 100  \
    --log_level info   \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1  \
    --evaluation_strategy steps   \
    --save_total_limit 1  \
    --logging_strategy steps \
    --logging_steps 10   \
    --output_dir $OUTPUT_DIR  \
    --num_train_epochs 1  \
    --max_length 1024   \
    --max_prompt_length 512 \
    --seed 42  \
    --overwrite_output_dir \
    --report_to none \
    --local_dataset \
    --dataset_name_or_path $DATASET_NAME_OR_PATH