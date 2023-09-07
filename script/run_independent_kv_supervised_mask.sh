# CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=""
# python  src/train_bash.py \
deepspeed --num_gpus=8  src/train_bash.py \
    --stage sft \
    --independent_kv_type supervised \
    --model_name_or_path /data/models/Llama-2-13b-hf \
    --do_train \
    --dataset wikihop_gpt_correct \
    --position_emb_type parallel \
    --ref_num 5 \
    --ref_length 256 \
    --instruction_length 12 \
    --template default \
    --finetuning_type full \
    --output_dir /data/models/Llama-2-13b-wikihop_gpt_correct-newmask-PEparallelv2-reflen256 \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 10 \
    --max_source_length 2048 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
    --deepspeed config/deepspeed.json