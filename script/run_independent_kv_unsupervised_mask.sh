deepspeed --num_gpus=8  src/train_bash.py \
    --stage pt \
    --independent_kv_type unsupervised \
    --model_name_or_path /data/models/Llama-2-13b-hf \
    --do_train \
    --dataset wiki_demo_2 \
    --position_emb_type serial \
    --ref_num 16 \
    --ref_length 128 \
    --template default \
    --finetuning_type full \
    --output_dir /data/models/Llama-2-13b-wiki-newmask-serial-test \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 10 \
    --max_source_length 2048 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --deepspeed config/deepspeed.json
