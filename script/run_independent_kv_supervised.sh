# CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=""
# python  src/train_bash.py \
deepspeed --num_gpus=8  src/train_bash.py \
    --stage sft \
    --independent_kv_type supervised \
    --model_name_or_path /data/models/Llama-2-7b-hf \
    --do_train \
    --dataset wikihop_gpt_correct \
    --template default \
    --finetuning_type full \
    --output_dir /data/models/Llama-2-7b-wikihop_gpt_correct_test \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 10 \
    --max_source_length 4096 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --deepspeed config/deepspeed.json
